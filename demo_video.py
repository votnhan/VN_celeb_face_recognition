import cv2
import os
import argparse
import glob
import time
import pandas as pd
import numpy as np
import face_alignment
import torch
import models as model_md 
from pathlib import Path
from utils import read_image, read_json
from demo_image import find_embedding, identify_person, \
                        draw_boxes_on_image, load_model_classify, \
                            get_face_from_boxes, align_face, move_landmark_to_box
from imgaug import augmenters as iaa
from align_face import alignment, center_point_dict
from data_loader import transforms_default

def seq_detection_and_alignment(np_image, detect_model, embedding_model, fa_model,
                            classify_model, device, label2name_df, target_fs, 
                            center_point, box_requirements, threshold):
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    boxes, _ = detect_model.inference(rgb_image, landmark=False)
    if len(boxes) > 0:
        list_face, face_idx = get_face_from_boxes(np_image, boxes)
        aligned_face_list = []
        new_face_idx = []
        for idx, face in enumerate(list_face):
            dst = align_face(face, fa_model)
            if dst is not None:
                aligned_face = alignment(face, center_point, dst, target_fs[0], 
                                target_fs[1])
                rgb_alg_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_face_list.append(rgb_alg_face)
                new_face_idx.append(idx)

        remain_idx = [face_idx[x] for x in new_face_idx]
        
        if len(remain_idx) > 0:
            tf_list = []
            for face in aligned_face_list:
                tf_face = transforms_default(face)
                tf_list.append(tf_face)

            aligned_faces_tf = torch.stack(tf_list, dim=0)
            chosen_boxes = [boxes[x] for x in remain_idx]
            embeddings = find_embedding(aligned_faces_tf.to(device), embedding_model)
            names = identify_person(embeddings, classify_model, label2name_df, threshold)
            np_image_recog = draw_boxes_on_image(np_image, chosen_boxes, names)
            return np_image_recog, names, chosen_boxes
        
        return np_image, None, None
    else:
        return np_image, None, None

def parallel_detection_and_alignment(np_image, detect_model, embedding_model, fa_model,
                            classify_model, device, label2name_df, target_fs, 
                            center_point, threshold):
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    boxes, _, landmarks = detect_model.inference(rgb_image, landmark=True)
    if len(boxes) > 0:
        list_face, face_idx = get_face_from_boxes(np_image, boxes)
        if len(face_idx) > 0:
            aligned_face_list = []
            chosen_boxes = [boxes[x] for x in face_idx]
            chosen_landmarks = [landmarks[x] for x in face_idx]

            for idx, face in enumerate(list_face):
                moved_landmark = move_landmark_to_box(chosen_boxes[idx], chosen_landmarks[idx])
                aligned_face = alignment(face, center_point, moved_landmark, target_fs[0], target_fs[1])
                rgb_alg_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_face_list.append(rgb_alg_face)

            tf_list = []
            for face in aligned_face_list:
                tf_face = transforms_default(face)
                tf_list.append(tf_face)

            aligned_faces_tf = torch.stack(tf_list, dim=0)
            embeddings = find_embedding(aligned_faces_tf.to(device), embedding_model)
            names = identify_person(embeddings, classify_model, label2name_df, threshold)
            np_image_recog = draw_boxes_on_image(np_image, chosen_boxes, names)
            return np_image_recog, names, chosen_boxes
        
        else:
            return np_image, None, None
    else:
        return np_image, None, None


def export_video_face_recognition(output_frame_dir, fps, output_path):
    container_path = Path(output_frame_dir)
    image_files = glob.glob(str(container_path / '*'))
    n_images = len(image_files)
    size = None
    first_img_path = container_path / 'frame_{}.png'.format(1)
    first_img = cv2.imread(str(first_img_path))
    height, width, channels = first_img.shape
    size = (width, height)
    out_writer = cv2.VideoWriter(output_path, 
                    cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(1, n_images + 1):
        img_path = container_path / 'frame_{}.png'.format(i)
        img = cv2.imread(str(img_path))
        out_writer.write(img)

    out_writer.release()
    print('Save exported video in {} ...'.format(output_path))
        

def main(args, detect_model, embedding_model, classify_model, fa_model, device, 
            label2name_df, target_fs, center_point):
    
    if not os.path.exists(args.output_frame):
        os.makedirs(args.output_frame)

    print('Method: {}'.format(args.inference_method))
    if args.inference_method == 'seq_fd_vs_aln':
        box_requirements = {
            'min_dim': args.min_dim_box,
            'box_ratio': args.box_ratio
        }
    
    cap = cv2.VideoCapture(args.video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = []
    start_time = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        time_in_video = count / fps
        print('Processing for frame: {}, time: {:.2f} s'.format(count, 
                    time_in_video))
        if args.inference_method == 'seq_fd_vs_aln':
            recognized_img, names, _ = seq_detection_and_alignment(frame, detect_model, 
                                    embedding_model, fa_model, classify_model, 
                                    device, label2name_df, target_fs, 
                                    center_point, box_requirements, args.recog_threshold)
        elif args.inference_method == 'par_fd_vs_aln':
            recognized_img, names, _ = parallel_detection_and_alignment(frame, detect_model, 
                                    embedding_model, fa_model, classify_model, 
                                    device, label2name_df, target_fs, 
                                    center_point, args.recog_threshold)
        else:
            print('Do not support {} method.'.format(args.args.inference_method))
            break

        if args.save_frame_recognized != '':
            image_name = 'frame_{}.png'.format(count)
            image_path = os.path.join(args.output_frame, image_name)
            cv2.imwrite(image_path, recognized_img)
        
        if names is None:
            names = []
        
        tracker.append((time_in_video, str(names)))
    
    end_time = time.time()
    processed_time = end_time - start_time
    fps_process = int(count / processed_time)
    tracked_df = pd.DataFrame(data=tracker, columns=['Time', 'Names'])
    tracked_df.to_csv(args.output_tracker, index=False)
    cap.release()
    print('Saved tracker file in {} ...'.format(args.output_tracker))
    print('FPS for recognition face: {}'.format(fps_process))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face \
                    recognition on a video')

    args_parser.add_argument('-fs', '--face_size', default=160, type = int)
    args_parser.add_argument('-mfs', '--min_face_size', default=50, type=int)
    args_parser.add_argument('-i', '--video_path', default='video.mp4', type=str)
    args_parser.add_argument('-o', '--output_frame', default='output_frame', 
                                type=str)
    args_parser.add_argument('-ot', '--output_tracker', default='tracker.csv', 
                                type=str)
    args_parser.add_argument('-m', '--classify_model', default='model_best.pth', 
                                type=str)
    args_parser.add_argument('-l2n', '--label2name', default='label2name.csv', 
                                type=str)
    args_parser.add_argument('-w', '--pre_trained_emb', default='vggface2', 
                                type=str)
    args_parser.add_argument('-dv', '--device', default='GPU', type=str) 
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-ov', '--output_video', default='', type=str)
    args_parser.add_argument('-fps', '--fps_video', default=25.0, type=float)
    args_parser.add_argument('-sfr', '--save_frame_recognized', default='', 
                                    type=str)
    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-enc', '--encoder', default='InceptionResnetV1', 
                                type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/embedding/iresnet100_enc.json', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, type=int)
    args_parser.add_argument('--inference_method', default='seq_fd_vs_aln', type=str)
    args_parser.add_argument('--min_dim_box', default=50, type=int)
    args_parser.add_argument('--box_ratio', default=2.0, type=float)
    args_parser.add_argument('--recog_threshold', default=0.0, type=float)

    args = args_parser.parse_args()

    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'

    # Prepare 3 models, database for label to name
    label2name_df = pd.read_csv(args.label2name)
    
    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()

    # face alignment model
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                flip_input=False, device=device)

    # face embedding model
    enc_args = read_json(args.encoder_args)
    emb_model = getattr(model_md, args.encoder)(**enc_args).to(device)

    # classify from embedding model
    classify_model = model_md.MLPModel(args.input_dim_emb, args.num_classes)
    load_model_classify(args.classify_model, classify_model)
    classify_model = classify_model.to(device)

    # center point, face size after alignment
    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]

    main(args, detection_md, emb_model, classify_model, fa_model, device, 
            label2name_df, target_fs, center_point)

    if args.output_video != '':
        export_video_face_recognition(args.output_frame, args.fps_video, 
            args.output_video)
