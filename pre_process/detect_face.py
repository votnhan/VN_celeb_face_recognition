import sys
sys.path.append('./')
import argparse
import os
import logging
import cv2
import glob
import models as model_md
from pathlib import Path
from utils import read_json
from logger import get_logger_for_run
from crop_face import get_face_from_box


def draw_bbox_and_idx(image, box, idx):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(image, str(idx), (box[2], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def detect_faces(detect_model, input_dir):
    img_path_pt = input_dir + '/*'
    img_paths = glob.glob(img_path_pt)
    img_paths.sort()
    rgb_imgs, bgr_imgs = [], []
    for img_path in img_paths:
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_imgs.append(rgb_img)
        bgr_imgs.append(bgr_img)
    
    bth_bboxes, _ = detect_model.inference(rgb_imgs, landmark=False)
    return bth_bboxes, img_paths, bgr_imgs


def visualize_faces_for_folder(detect_model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    bth_bboxes, img_paths, bgr_imgs = detect_faces(detect_model, input_dir)
    n_images = len(img_paths)
    for img_idx, bboxes in enumerate(bth_bboxes):
        logger.info('{}/{}, Visualizing for image {}'.format(img_idx, n_images, 
                        img_paths[img_idx]))
        for box_idx, bbox in enumerate(bboxes):
            draw_bbox_and_idx(bgr_imgs[img_idx], bbox, box_idx)
        
        img_file = img_paths[img_idx].split('/')[-1]
        new_img_path = os.path.join(output_dir, img_file)
        cv2.imwrite(new_img_path, bgr_imgs[img_idx])


def choose_faces_for_folder(detect_model, input_dir, output_dir, face_idxes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger(os.environ['LOGGER_ID'])
    bth_bboxes, img_paths, bgr_imgs = detect_faces(detect_model, input_dir)
    n_images = len(img_paths)
    for img_idx, bboxes in enumerate(bth_bboxes):
        logger.info('{}/{}, Choosing face for image {}'.format(img_idx, n_images, 
                        img_paths[img_idx]))
        choosen_box = bboxes[face_idxes[img_idx]]
        face = get_face_from_box(bgr_imgs[img_idx], choosen_box)
        img_file = img_paths[img_idx].split('/')[-1]
        new_img_path = os.path.join(output_dir, img_file)
        cv2.imwrite(new_img_path, face)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Detect faces and choose face')
    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('--input_dir', default='data', type=str)
    args_parser.add_argument('--vlz_output_dir', default='detect_face_output', 
                                type=str)
    args_parser.add_argument('--face_output_dir', default='face_output', 
                                type=str)
    args_parser.add_argument('--output_log', default='detect_face_log', type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str)
    args_parser.add_argument('--visualize', action='store_true')
    args_parser.add_argument('--choose_faces', action='store_true')
    args_parser.add_argument('--face_idxes', nargs='+', type=int)
    
    args = args_parser.parse_args()
    args.output_log = os.path.join(args.root_output, args.output_log)
    logger, log_dir = get_logger_for_run(args.output_log)
    logger.info('Crop faces from folder {}'.format(args.input_dir))
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))

    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()

    if args.visualize:
        args.vlz_output_dir = os.path.join(args.root_output, args.vlz_output_dir)
        logger.info('Start visualization process')
        visualize_faces_for_folder(detection_md, args.input_dir, args.vlz_output_dir)
        logger.info('End visualization process')

    if args.choose_faces:
        args.face_output_dir = os.path.join(args.root_output, args.face_output_dir)
        logger.info('Start face choosing process')
        choose_faces_for_folder(detection_md, args.input_dir, args.face_output_dir, 
                                    args.face_idxes)
        logger.info('End face choosing process')
