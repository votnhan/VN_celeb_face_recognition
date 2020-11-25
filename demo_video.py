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
from utils import read_image, read_json, load_pickle
from demo_image import  draw_boxes_on_image, load_model_classify, \
                        get_face_from_boxes, align_face, \
                        move_landmark_to_box, recognize_celeb, \
                        recognize_emotion, draw_emotions, \
                        sequential_detect_and_align, \
                        parallel_detect_and_align
from imgaug import augmenters as iaa
from align_face import alignment, center_point_dict
from data_loader import transforms_default, trans_emotion_inf


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
    
    if args.recog_emotion:
        idx2etag = load_pickle(args.etag2idx_file)['idx2key']
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args).to(device)
    
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
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.inference_method == 'seq_fd_vs_aln':
            recognized_img, names = frame, None
            alg_face_list, chosen_boxes = sequential_detect_and_align(rgb_image, 
                                            detection_md, center_point, 
                                            target_fs, box_requirements,False)
    
        elif args.inference_method == 'par_fd_vs_aln':
            recognized_img, names = frame, None
            alg_face_list, chosen_boxes = parallel_detect_and_align(rgb_image, 
                                        detection_md, center_point, target_fs, 
                                        False)
        else:
            print('Do not support {} method.'.format(args.args.inference_method))
            break

        if len(chosen_boxes) > 0:
            names = recognize_celeb(alg_face_list, chosen_boxes, device, 
                                    emb_model, classify_model, 
                                    transforms_default, 
                                    label2name_df, args.recog_threshold)
            np_image_recog = draw_boxes_on_image(frame, chosen_boxes, names)

            if args.recog_emotion:
                map_func = np.vectorize(lambda x: idx2etag[x])
                emotions, probs = recognize_emotion(alg_face_list, device, 
                                        emt_model, trans_emotion_inf, map_func, 
                                        args.topk_emotions)
                np_image_recog = draw_emotions(np_image_recog, chosen_boxes, 
                                    emotions, probs)
            recognized_img = np_image_recog

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
    args_parser.add_argument('--recog_emotion', action='store_true')
    args_parser.add_argument('-emt', '--emotion', default='resnet_2branch_50', 
                                type=str)
    args_parser.add_argument('-emtargs', '--emotion_args', 
                                default='cfg/emotion/resnet50_2_branch.json', 
                                type=str)
    args_parser.add_argument('-t2i', '--etag2idx_file', 
                        default='meta_data/emotion_recognition/etag2idx.pkl.keep', 
                        type=str)
    args_parser.add_argument('--topk_emotions', default=6, type=int)

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

    # emotion model (if need)
    emt_model = None
    if args.recog_emotion:
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args).to(device)

    # center point, face size after alignment
    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]

    main(args, detection_md, emb_model, classify_model, fa_model, device, 
            label2name_df, target_fs, center_point)

    if args.output_video != '':
        export_video_face_recognition(args.output_frame, args.fps_video, 
            args.output_video)
