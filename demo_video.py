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
from utils import read_image, read_json, load_pickle, \
                    convert_sec_to_max_time_quantity, append_log_to_file
from demo_image import  draw_boxes_on_image, load_model_classify, \
                        get_face_from_boxes, align_face, \
                        move_landmark_to_box, recognize_celeb, \
                        recognize_emotion, draw_emotions, \
                        sequential_detect_and_align, \
                        parallel_detect_and_align
from imgaug import augmenters as iaa
from pre_process import alignment, center_point_dict
from data_loader import transforms_default, trans_emotion_inf
from db import CelebDB, EmotionDB


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
        

def main(args, detect_model, embedding_model, classify_models, fa_model, device, 
            celeb_db, target_fs, center_point):
    
    if not os.path.exists(args.output_frame):
        os.makedirs(args.output_frame)

    print('Method: {}'.format(args.inference_method))
    if args.inference_method == 'seq_fd_vs_aln':
        box_requirements = {
            'min_dim': args.min_dim_box,
            'box_ratio': args.box_ratio
        }
    
    
    # emotion model (if need)
    if args.recog_emotion:
        # Database for emotion
        emotion_db = EmotionDB(args.etag2idx_file)
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args).to(device)

    # Create tracker file 
    df_columns = ['Time', 'Names', 'Frame_idx', 'Bboxes']
    if args.recog_emotion:
        df_columns.append('Emotion')

    # Overwrite old tracker file
    with open(args.output_tracker, 'w') as tracker_file:
        tracker_file.write('')

    append_log_to_file(args.output_tracker, df_columns)    
    
    # Data structure for statistic algorithm
    cap = cv2.VideoCapture(args.video_path)
    count = 0
    processed_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()
    frames_queue, frames_info = [], []
    end_video = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            end_video = True
            
        count += 1
        time_in_video = count / fps
        if not end_video:
            frames_queue.append(frame)
            frames_info.append([time_in_video, count])

        if (len(frames_queue) !=  args.n_frames) and not end_video:
            continue

        processed_frame += len(frames_queue)

        if (processed_frame % args.log_step) == 0:
            hms_time = convert_sec_to_max_time_quantity(time_in_video)
            print('Processing for frame: {}, time: {}'.format(count, 
                        hms_time))
        
        rgb_images = []
        for frame in frames_queue:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)
        
        if args.inference_method == 'seq_fd_vs_aln':
            bth_alg_faces, bth_chosen_boxes = sequential_detect_and_align(rgb_images, 
                                            detection_md, center_point, 
                                            target_fs, box_requirements,False)
    
        elif args.inference_method == 'par_fd_vs_aln':
            bth_alg_faces, bth_chosen_boxes, bth_chosen_faces = parallel_detect_and_align(rgb_images, 
                                            detection_md, center_point, 
                                            target_fs, False)
        else:
            print('Do not support {} method.'.format(args.args.inference_method))
            break


        bth_names = recognize_celeb(bth_alg_faces, device, 
                                emb_model, classify_models, 
                                transforms_default, 
                                celeb_db, args.recog_threshold)

        np_image_recogs = []
        for idx, names in enumerate(bth_names):
            if len(names) > 0:
                img_recog = draw_boxes_on_image(frames_queue[idx], 
                                bth_chosen_boxes[idx], names)
            else:
                img_recog = frames_queue[idx]
            np_image_recogs.append(img_recog)

        if args.recog_emotion:
            bth_emotions, bth_probs = recognize_emotion(bth_chosen_faces, device, 
                                    emt_model, trans_emotion_inf, emotion_db,
                                    args.topk_emotions)
            for idx, (emotions, probs) in enumerate(zip(bth_emotions, bth_probs)):
                draw_emotions(np_image_recogs[idx], bth_chosen_boxes[idx], 
                                emotions, probs)

        if args.save_frame_recognized != '':
            for idx, recog_img in enumerate(np_image_recogs):
                image_name = 'frame_{}.png'.format(frames_info[idx][1])
                image_path = os.path.join(args.output_frame, image_name)
                cv2.imwrite(image_path, recog_img)

        logged_rows = []
        for idx, names in enumerate(bth_names):
            bboxes = bth_chosen_boxes[idx]
            row = [str(frames_info[idx][0]), '"' + str(names) + '"', 
                    str(frames_info[idx][1])]

            if len(bboxes) == 0:
                scaled_bboxes = []
            else:
                h, w, _ = frames_queue[idx].shape
                scale = np.array([w, h, w, h])
                scaled_bboxes = [list(x / scale) for x in bboxes]
            
            row.append('"' + str(scaled_bboxes) + '"')
            
            if args.recog_emotion:
                emotions = bth_emotions[idx]
                emotions_list = []
                if len(bboxes) > 0 :
                    for i in range(emotions.shape[0]):
                        emotions_list.append(list(emotions[i]))
                
                row.append('"' + str(emotions_list) + '"')

            str_row = ','.join(row) + '\n'
            logged_rows.append(str_row)

        str_logged_rows = ''.join(logged_rows)
        with open(args.output_tracker, 'a') as tracker_file:
            tracker_file.write(str_logged_rows)

        # Destroy queue !
        frames_queue = []
        frames_info = []

        # Check end video 
        if end_video:
            break        

    end_time = time.time()
    processed_time = end_time - start_time
    fps_process = int(processed_frame / processed_time)
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
    args_parser.add_argument('-m', '--classify_model', nargs='+', type=str)
    args_parser.add_argument('-l2n', '--label2name', default='label2name.csv', 
                                type=str)
    args_parser.add_argument('--alias2main_id', default='alias2main_id.json', 
                                type=str)
    args_parser.add_argument('-w', '--pre_trained_emb', default='vggface2', 
                                type=str)
    args_parser.add_argument('-dv', '--device', default='cuda:0', type=str) 
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-ov', '--output_video', default='', type=str)
    args_parser.add_argument('-fps', '--fps_video', default=25.0, type=float)
    args_parser.add_argument('-sfr', '--save_frame_recognized', action='store_true')
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
    args_parser.add_argument('--log_step', default=100, type=int)
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
    args_parser.add_argument('--n_frames', default=16, type=int)

    args = args_parser.parse_args()

    device = args.device

    # Database for label to name
    celeb_db = CelebDB(args.label2name, args.alias2main_id)
    
    # Prepare 3 models, 
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
    cls_model_paths = list(args.classify_model)
    classify_models = []
    for path in cls_model_paths:
        classify_model = model_md.MLPModel(args.input_dim_emb, args.num_classes)
        load_model_classify(path, classify_model)
        classify_model.to(device)
        classify_models.append(classify_model)

    # center point, face size after alignment
    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]

    main(args, detection_md, emb_model, classify_models, fa_model, device, 
            celeb_db, target_fs, center_point)

    if args.output_video != '':
        export_video_face_recognition(args.output_frame, args.fps_video, 
            args.output_video)
