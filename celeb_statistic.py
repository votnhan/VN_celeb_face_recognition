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
import models as model_md 
from collections import Counter
from pathlib import Path
from itertools import groupby
from utils import read_image, read_json, write_json
from demo_image import find_embedding, identify_person, \
                        draw_boxes_on_image, load_model_classify, \
                            get_face_from_boxes, align_face
from imgaug import augmenters as iaa
from align_face import alignment, center_point_dict
from data_loader import transforms_default

def ignore_name_unknown(names_list, bboxes_list, unknown_name):
    chosen_list = []
    for idx, name in enumerate(names_list):
        if name != unknown_name:
            chosen_list.append(idx)
    
    chosen_names = [names_list[x] for x in chosen_list]
    chosen_bboxes = [bboxes_list[x] for x in chosen_list]
    return chosen_bboxes, chosen_names


def recognize_faces_image(np_image, detect_model, embedding_model, fa_model,
                            classify_model, device, label2name_df, target_fs, 
                            center_point, ignored_name='Unknown'):
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    _, boxes = detect_model(rgb_image, extract_face=False)
    if boxes is not None:
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
            embeddings = find_embedding(aligned_faces_tf.to(device), emb_model)
            names = identify_person(embeddings, classify_model, label2name_df)
            after_ignored_bboxes, after_ignore_names = ignore_name_unknown(names, 
                                                        chosen_boxes, ignored_name)
            np_image_recog = draw_boxes_on_image(np_image, after_ignored_bboxes, 
                                after_ignore_names)
            
            return np_image_recog, after_ignore_names
        
        return np_image, None
    else:
        return np_image, None


def export_json_stat(tracker_df, output_js_path, n_video_intervals, n_appear=8):
    interval_counter = 0
    dict_track = {}
    n_rows = len(tracker_df['Time'])
    n_rows_in_itv = n_rows // n_video_intervals

    for i in range(n_video_intervals):
        interval_counter += 1
        start_range = i*n_rows_in_itv
        if (i+1)*n_rows_in_itv > n_rows:
            end_range = n_rows
        else:
            end_range = (i+1)*n_rows_in_itv
        df_for_itv = tracker_df.iloc[start_range: end_range]
        names_for_itv = []
        for names in df_for_itv['Names']:
            list_names = ast.literal_eval(names)
            names_for_itv += list_names
        
        unique_names_for_itv = []
        for k, v in Counter(names_for_itv).items():
            if v >= n_appear:
                unique_names_for_itv.append(k)

        dict_track[str(interval_counter)] = {
            "interval": (df_for_itv['Time'].iloc[0], df_for_itv['Time'].iloc[-1]),
            "celebrities": unique_names_for_itv
        }
    
    write_json(output_js_path, dict_track, log=True)


def main(args, detect_model, embedding_model, classify_model, fa_model, device, 
            label2name_df, target_fs, center_point, frame_idxes):
    
    if not os.path.exists(args.output_frame):
        os.makedirs(args.output_frame)

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
        if_process = False
        for idx in frame_idxes:
            if count % fps == idx:
                if_process = True
                break
        
        if not if_process:
            continue
        
        time_in_video = count / fps
        print('Processing for frame: {}, time: {:.2f} s'.format(count, 
                    time_in_video))
        recognized_img, names = recognize_faces_image(frame, detect_model, 
                                    embedding_model, fa_model, classify_model, 
                                    device, label2name_df, target_fs, 
                                    center_point, args.ignored_name)

        if args.save_frame_recognized:
            image_name = 'frame_{}.png'.format(count)
            image_path = os.path.join(args.output_frame, image_name)
            cv2.imwrite(image_path, recognized_img)
        
        if names is None:
            names = []
        
        tracker.append((time_in_video, str(names), count))
    
    end_time = time.time()
    processed_time = end_time - start_time
    fps_process = int(count / processed_time)
    tracked_df = pd.DataFrame(data=tracker, columns=['Time', 'Names', 'Frame_idx'])
    tracked_df.to_csv(args.output_tracker, index=False)
    cap.release()
    print('Saved tracker file in {} ...'.format(args.output_tracker))
    print('FPS for recognition face: {}'.format(fps_process))
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
    args_parser.add_argument('-enc', '--encoder', default='InceptionResnetV1', 
                                type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/iresnet100_enc.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    args_parser.add_argument('-jst', '--json_tracker', default='tracker.json', 
                                type=str)
    args_parser.add_argument('-fidx','--frame_idxes', nargs='+', type=int)
    args_parser.add_argument('-ign', '--ignored_name', default='Unknown', 
                                type=str)
    args_parser.add_argument('-nvi', '--n_video_intervals', default=5, type=int)
    args_parser.add_argument('-tap', '--time_appear', default=2, type=int)

    args = args_parser.parse_args()

    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'

    # Prepare 3 models, database for label to name
    label2name_df = pd.read_csv(args.label2name)
    
    # face detection model
    mtcnn = model_md.MTCNN(args.face_size, keep_all=True, device=device, 
                    min_face_size=args.min_face_size)
    mtcnn.eval()

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
    tracker_df = main(args, mtcnn, emb_model, classify_model, fa_model, device, 
                        label2name_df, target_fs, center_point, frame_idxes)

    # export JSON file for video celebrity indexing 
    export_json_stat(tracker_df, args.json_tracker, 
                        args.time_appear*len(frame_idxes))

