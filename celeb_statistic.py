import cv2
import os
import argparse
import glob
import time
import ast
import pandas as pd
import numpy as np
import face_alignment
import torch
import pafy
import math
import models as model_md 
from collections import Counter
from pathlib import Path
from itertools import groupby
from utils import read_image, read_json, write_json, load_pickle, append_log_to_file
from demo_image import find_embedding, identify_person, \
                        draw_boxes_on_image, load_model_classify, \
                        get_face_from_boxes, align_face, \
                        move_landmark_to_box, recognize_celeb, \
                        recognize_emotion, draw_emotions, \
                        sequential_detect_and_align, \
                        parallel_detect_and_align

from imgaug import augmenters as iaa
from align_face import alignment, center_point_dict
from data_loader import transforms_default, trans_emotion_inf
from utils import convert_sec_to_max_time_quantity


def export_json_stat_dynamic_itv(tracker_df, output_js_path, n_intervals, 
                        n_appear=4, unknown_name='Unknown'):
    n_rows = len(tracker_df['Time'])
    interval_counter = 0
    dict_track = {}
    n_rows_in_itv = n_rows // n_intervals
    remain_rows = n_rows % n_intervals
    for i in range(n_intervals):
        interval_counter += 1
        start_range = i*n_rows_in_itv
        end_range = (i+1)*n_rows_in_itv
        if i == n_intervals - 1:
            end_range += remain_rows
        df_for_itv = tracker_df.iloc[start_range: end_range]
        final_bboxes_dict, start_itv, end_itv = find_celeb_infor_in_interval(df_for_itv, 
                                                    unknown_name, n_appear)
        dict_track[str(interval_counter)] = {
            "interval": (start_itv, end_itv),
            "celebrities": final_bboxes_dict
        }
    
    write_json(output_js_path, dict_track, log=True)


def export_json_stat_fixed_itv(tracker_df, output_js_path, n_rows_in_itv, 
                        n_appear=4, unknown_name='Unknown'):
    n_rows = len(tracker_df['Time'])
    interval_counter = 0
    dict_track = {}
    n_intervals = math.ceil(n_rows / n_rows_in_itv)
    for i in range(n_intervals):
        interval_counter += 1
        start_range = i*n_rows_in_itv
        end_range = (i+1)*n_rows_in_itv
        if end_range > n_rows:
            end_range = n_rows
        df_for_itv = tracker_df.iloc[start_range: end_range]
        final_bboxes_dict, start_itv, end_itv = find_celeb_infor_in_interval(df_for_itv, 
                                                    unknown_name, n_appear)
        dict_track[str(interval_counter)] = {
            "interval": (start_itv, end_itv),
            "celebrities": final_bboxes_dict
        }
    
    write_json(output_js_path, dict_track, log=True)


def find_celeb_infor_in_interval(df_for_itv, unknown_name, n_appear):
    bboxes_dict = {}
    zip_obj = zip(df_for_itv['Names'], df_for_itv['Bboxes'], 
                df_for_itv['Time'], df_for_itv['Emotion'])
    for names_str, bboxes_str, time_s, emotions in zip_obj:
        time_s = float(time_s)
        hms_time = convert_sec_to_max_time_quantity(time_s)
        list_names = ast.literal_eval(names_str)
        list_bboxes = ast.literal_eval(bboxes_str)
        list_emotions = ast.literal_eval(emotions)
        for name, bbox, emotion in zip(list_names, list_bboxes, list_emotions):
            bbox_item = {
                        'time': hms_time,
                        'bbox': bbox,
                        'emotions': emotion
                        }
            if name not in bboxes_dict:
                bboxes_dict[name] = [bbox_item]
            else:
                bboxes_dict[name] += [bbox_item]

    final_bboxes_dict = {}
    for k, v in bboxes_dict.items():
        if (k != unknown_name) and (len(v) >= n_appear):
            final_bboxes_dict[k] = v
            
    start_itv = convert_sec_to_max_time_quantity(float(df_for_itv['Time'].iloc[0]))
    end_itv = convert_sec_to_max_time_quantity(float(df_for_itv['Time'].iloc[-1]))
    return final_bboxes_dict, start_itv, end_itv


def main(args, detect_model, embedding_model, classify_model, fa_model, device, 
            label2name_df, target_fs, center_point, frame_idxes):
    
    if not os.path.exists(args.output_frame):
        os.makedirs(args.output_frame)

    if args.inference_method == 'seq_fd_vs_aln':
        box_requirements = {
            'min_dim': args.min_dim_box,
            'box_ratio': args.box_ratio
        }

    # emotion model (if need)
    if args.recog_emotion:
        idx2etag = load_pickle(args.etag2idx_file)['idx2key']
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args).to(device)

    # Create threshold
    if args.local_thresholds != '':
      print('Using local thresholds !')
      threshold = read_json(args.local_thresholds)
    else:
      print('Using global a threshold !')
      threshold = {}
      for i in range(args.num_classes):
          threshold[str(i)] = args.recog_threshold

    # Create tracker file 
    df_columns = ['Time', 'Names', 'Frame_idx']
    if args.track_bbox:
        df_columns.append('Bboxes')
    if args.recog_emotion:
        df_columns.append('Emotion')
    
    # Overwrite old tracker file
    with open(args.output_tracker, 'w') as tracker_file:
        tracker_file.write('')

    append_log_to_file(args.output_tracker, df_columns)

    # Index video on Youtube
    if args.youtube_video:
        pafy_obj = pafy.new(args.video_path)
        play = pafy_obj.getbest(preftype="mp4")
        if play is None:
            print('This Youtube video did not support mp4 format !')
            return 
        print('Video resolution: {}, video format: {}'.format(play.resolution, 
                play.extension))
        video_path = play.url
    else:
        video_path = args.video_path
    
    # Data structure for statistic algorithm
    cap = cv2.VideoCapture(video_path)
    count = 0
    processed_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()
    frames_queue, frames_info = [], []
    end_video = False
    
    # Process video !
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            end_video = True
        
        count += 1
        if_process = False
        for idx in frame_idxes:
            if count % fps == idx:
                if_process = True
                break

        if (not if_process) and not end_video:
            continue

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
                                            detection_md, center_point, target_fs, 
                                            box_requirements, False)
                
        elif args.inference_method == 'par_fd_vs_aln':
            bth_alg_faces, bth_chosen_boxes = parallel_detect_and_align(rgb_images, 
                                        detection_md, center_point, target_fs, 
                                        False)
        else:
            print('Do not support {} method.'.format(args.args.inference_method))
            break


        bth_names = recognize_celeb(bth_alg_faces, device, emb_model, classify_model, 
                    transforms_default, label2name_df, threshold)

        np_image_recogs = []
        for idx, names in enumerate(bth_names):
            if len(names) > 0:
                img_recog = draw_boxes_on_image(frames_queue[idx], 
                                bth_chosen_boxes[idx], names)
            else:
                img_recog = frames_queue[idx]
            np_image_recogs.append(img_recog)

        if args.recog_emotion:
            map_func = np.vectorize(lambda x: idx2etag[x])
            bth_emotions, bth_probs = recognize_emotion(bth_alg_faces, device, 
                                    emt_model, trans_emotion_inf, map_func ,
                                    args.topk_emotions)
            for idx, (emotions, probs) in enumerate(zip(bth_emotions, bth_probs)):
                draw_emotions(np_image_recogs[idx], bth_chosen_boxes[idx], 
                                emotions, probs)

        if args.save_frame_recognized:
            for idx, recog_img in enumerate(np_image_recogs):
                image_name = 'frame_{}.png'.format(frames_info[idx][1])
                image_path = os.path.join(args.output_frame, image_name)
                cv2.imwrite(image_path, recog_img)
        
        logged_rows = []
        for idx, names in enumerate(bth_names):
            bboxes = bth_chosen_boxes[idx]
            row = [str(frames_info[idx][0]), '"' + str(names) + '"', 
                    str(frames_info[idx][1])]
            if args.track_bbox:
                if bboxes is None:
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
    tracked_df = pd.read_csv(args.output_tracker)
    return tracked_df


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
    args_parser.add_argument('-dv', '--device', default='GPU', type=str) 
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-sfr', '--save_frame_recognized', 
                                action='store_true')
    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-enc', '--encoder', default='InceptionResnetV1', 
                                type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/embedding/iresnet100_enc.json', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    args_parser.add_argument('-jst', '--json_tracker', default='tracker.json', 
                                type=str)
    args_parser.add_argument('-fidx','--frame_idxes', nargs='+', type=int)
    args_parser.add_argument('-ign', '--ignored_name', default='Unknown', 
                                type=str)
    args_parser.add_argument('-nvi', '--n_video_intervals', default=5, type=int)
    args_parser.add_argument('-tap', '--n_time_appear', default=8, type=int)
    args_parser.add_argument('--statistic_mode', default='dynamic_itv', type=str, 
                                help='dynamic_itv or fixed_itv')
    args_parser.add_argument('--time_an_interval', default=5, type=int)
    args_parser.add_argument('--inference_method', default='seq_fd_vs_aln', type=str)
    args_parser.add_argument('--min_dim_box', default=50, type=int)
    args_parser.add_argument('--box_ratio', default=2.0, type=float)
    args_parser.add_argument('--log_step', default=100, type=int)
    args_parser.add_argument('--recog_threshold', default=0.7, type=float)
    args_parser.add_argument('--local_thresholds', default='', type=str)
    args_parser.add_argument('--track_bbox', action='store_true')
    args_parser.add_argument('--youtube_video', action='store_true')
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

    # choose frames for a second
    frame_idxes = list(args.frame_idxes) 
    if not os.path.exists(args.output_tracker):
        print('Create tracker file {}'.format(args.output_tracker))
        tracker_df = main(args, detection_md, emb_model, classify_model, fa_model, device, 
                            label2name_df, target_fs, center_point, frame_idxes)
    else:
        print('Re-use tracker file {}'.format(args.output_tracker))
        tracker_df = pd.read_csv(args.output_tracker)

    # export JSON file for video celebrity indexing 
    print('Statistic mode: {}'.format(args.statistic_mode))
    if args.statistic_mode == 'dynamic_itv':
        export_json_stat_dynamic_itv(tracker_df, args.json_tracker, 
                                        args.n_video_intervals, 
                                        args.n_time_appear, 
                                        args.ignored_name)
    elif args.statistic_mode == 'fixed_itv':
        n_rows_in_itv = args.time_an_interval * len(frame_idxes) * 60
        export_json_stat_fixed_itv(tracker_df, args.json_tracker, n_rows_in_itv, 
                                    args.n_time_appear, args.ignored_name)
    else:
        print('This statistic mode {} is not supported !'.format(args.statistic_mode))
    


