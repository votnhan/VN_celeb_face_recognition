import sys
sys.path.append('./')
import argparse
import os
import glob
import models as model_md
import cv2
import pandas as pd
import logging
import shutil
from utils import read_json
from pathlib import Path
from demo_image import move_landmark_to_box, alignment
from align_face import center_point_dict
from dotenv import load_dotenv
from logger import get_logger_for_run
from pathlib import Path

load_dotenv()

def find_area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w*h


def find_max_area_box(bboxes):
    max_idx = 0
    max_area = find_area(bboxes[0])
    for idx, box in enumerate(bboxes[1: ]):
        area = find_area(box)
        if area > max_area:
            max_area = area
            max_idx = idx + 1

    return max_idx


def get_face_from_box(bgr_img, box):
    ori_h, ori_w = bgr_img.shape[:2]
    x1 = max(int(box[0]), 0)
    y1 = max(int(box[1]), 0)
    x2 = min(int(box[2] + 1), ori_w)
    y2 = min(int(box[3] + 1), ori_h)
    face = bgr_img[y1:y2, x1:x2, :]
    return face


def crop_and_align_face(model, rgb_image, center_point, target_fs):
    bth_bboxes, _, bth_landmarks = detection_md.inference([rgb_image], landmark=True)
    bboxes = bth_bboxes[0]
    landmarks = bth_landmarks[0]
    aligned_face = None
    if len(bboxes) > 0:
        max_idx = find_max_area_box(bboxes)
        moved_landmark = move_landmark_to_box(bboxes[max_idx], landmarks[max_idx])
        face = get_face_from_box(rgb_image, bboxes[max_idx])
        bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        aligned_face = alignment(bgr_face, center_point, moved_landmark, 
                            target_fs[0], target_fs[1])
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    return aligned_face, len(bboxes)


def crop_face(model, rgb_image):
    bth_bboxes, _ = model.inference([rgb_image], landmark=False)
    bboxes = bth_bboxes[0]
    face = None
    if len(bboxes) > 0:
        max_idx = find_max_area_box(bboxes)
        face = get_face_from_box(rgb_image, bboxes[max_idx])
    
    return face, len(bboxes)


def crop_face_dataset(input_dir, output_dir, detection_md, no_face_file, 
                many_boxes_file, align_params=None):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    n_no_face, many_boxes, total = 0, 0, 0
    img_files = os.listdir(input_dir)
    img_files.sort()
    n_images = len(img_files)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for idx, img_file in enumerate(img_files):
        total += 1
        logger.info('---------{}/{}---------'.format(idx, n_images))
        output_path = str(output_dir / img_file)
        if os.path.exists(output_path):
            continue
        img_path = str(input_dir / img_file)
        logger.info('Processing {}'.format(img_path))
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        if align_params is None:
            face, n_faces = crop_face(detection_md, rgb_img)
        else:
            face, n_faces = crop_and_align_face(detection_md, rgb_img, 
                                align_params['center_point'], align_params['target_fs'])

        if n_faces >  1:
            many_boxes_file.write(img_path + '\n')
            many_boxes += 1
            continue
        elif n_faces < 1:
            no_face_file.write(img_path + '\n')
            n_no_face += 1
            continue

        bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_face)
        logger.info('Finding face for {} is done ...'.format(img_file))

    logger.info('Total images: {}.'.format(total))
    logger.info('No face images: {}.'.format(n_no_face))
    logger.info('Many face images: {}.'.format(many_boxes))


def copy_fail_images(tracker_file, output_dir):
    tracker_file_obj = open(tracker_file, 'r')
    for line in tracker_file_obj.readlines():
        img_path = line[:-1]
        logger.info('Copying image {}'.format(img_path))
        img_file = img_path.split('/')[-1]
        new_path = os.path.join(output_dir, img_file)
        shutil.copy(img_path, new_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face alignment to \
                            specific size by landmarks detection model')
    args_parser.add_argument('-id', '--input_dir', default='data', type=str)
    args_parser.add_argument('-od', '--output_dir', default='crop_output', 
                                type=str)
    args_parser.add_argument('--many_faces_dir', default='many_faces', type=str)
    args_parser.add_argument('--no_face_dir', default='no_face', type=str)
    args_parser.add_argument('--collect_fail_face', action='store_true')
    args_parser.add_argument('-nf', '--no_face_file', default='no_face.txt', 
                                type=str)
    args_parser.add_argument('-mf', '--many_boxes_file', default='many_boxes.txt', 
                                type=str)    

    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('--align', action='store_true')
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    args_parser.add_argument('--output_log', default='crop_log', type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str)
    
    args = args_parser.parse_args()
    args.output_log = os.path.join(args.root_output, args.output_log)
    args.output_dir = os.path.join(args.root_output, args.output_dir)
    logger, log_dir = get_logger_for_run(args.output_log)
    logger.info('Crop faces from folder {}'.format(args.input_dir))
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()

    # tracker file 
    path_log = Path(log_dir) 
    no_face_file = open(str(path_log / args.no_face_file), 'w')
    many_boxes_file = open(str(path_log / args.many_boxes_file) , 'w')

    # face alignment params
    align_params = None
    if args.align:
        target_fs = (args.target_face_size, args.target_face_size)
        center_point = center_point_dict[str(target_fs)]
        align_params = {'center_point': center_point, 'target_fs': target_fs}
        logger.info('Detect and align parallel')

    crop_face_dataset(args.input_dir, args.output_dir, detection_md, no_face_file, 
                many_boxes_file, align_params)

    no_face_file.close()
    many_boxes_file.close()

    if args.collect_fail_face:
        logger.info('Start collecting non-face images')
        args.no_face_dir = os.path.join(args.root_output, args.no_face_dir)
        if not os.path.exists(args.no_face_dir):
            os.makedirs(args.no_face_dir)
        no_face_path = str(path_log / args.no_face_file)
        copy_fail_images(no_face_path, args.no_face_dir)
        logger.info('End collecting non-face images')

        logger.info('Start collecting many face images')
        args.many_faces_dir = os.path.join(args.root_output, args.many_faces_dir)
        if not os.path.exists(args.many_faces_dir):
            os.makedirs(args.many_faces_dir)
        many_faces_path = str(path_log / args.many_boxes_file)
        copy_fail_images(many_faces_path, args.many_faces_dir)
        logger.info('End collecting many face images')

