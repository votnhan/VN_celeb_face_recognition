import sys
sys.path.append('./')

import os
import glob
import argparse
import logging
import shutil
import pandas as pd
from logger import get_logger_for_run
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def convert_folder_2_flatten(root_dir, output_dir, wrong_format):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_str = root_dir + '/*/*'
    image_paths = glob.glob(path_str)
    n_images = len(image_paths)
    wrong_format_counter = 0
    for idx, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            continue

        logger.info('-----{}/{}-----'.format(idx, n_images))
        logger.info('Copying file {}'.format(image_path))
        label, image_file = image_path.split('/')[-2: ]
        image_name, ext = image_file.split('.')
        if ext not in ['png', 'jpg', 'jpeg']:
            wrong_format.write(image_path + '\n')
            wrong_format_counter += 1
            continue
        new_image_file = '{}_{}.{}'.format(label, image_name, ext)
        new_img_path = os.path.join(output_dir, new_image_file)
        if not os.path.exists(new_img_path):
            shutil.copyfile(image_path, new_img_path)

    print('Samples wrong format: {}'.format(wrong_format_counter))


def choose_straight_face(flatten_dir, straight_dir):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    path_pt = flatten_dir + '/*'
    img_paths = glob.glob(path_pt)
    img_paths.sort()
    n_images = len(img_paths)
    n_straight_faces = 0
    for idx, img_path in enumerate(img_paths):
        img_file = img_path.split('/')[-1]
        img_name, _ = img_file.split('.')
        if 'a' in img_name.lower():
            new_img_path = os.path.join(straight_dir, img_file)
            logger.info('{}/{}, Copying straight face {}'.format(idx, n_images, 
                            img_path))
            shutil.copy(img_path, new_img_path)
            n_straight_faces += 1

    logger.info('Number of straight faces: {}'.format(n_straight_faces))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Convert folder \
                    structure dataset to flatten dataset')
    
    args_parser.add_argument('--root_dir', default='data', type=str)
    args_parser.add_argument('--output_dir', default='flatten_dataset', type=str)
    args_parser.add_argument('--output_log', default='flatten_log', type=str)
    args_parser.add_argument('--wrong_format', default='wrong_format.txt', 
                                type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str)
    args_parser.add_argument('--choose_straight', action='store_true')
    args_parser.add_argument('--straight_dir', default='straight_faces', type=str)
    
    args = args_parser.parse_args() 
    args.output_log = os.path.join(args.root_output, args.output_log)
    args.output_dir = os.path.join(args.root_output, args.output_dir)
    args.straight_dir = os.path.join(args.root_output, args.straight_dir)

    logger, log_dir = get_logger_for_run(args.output_log)
    logger.info('Flatten folder dataset {}'.format(args.root_dir))
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))
    
    path_log = Path(log_dir)
    wrong_format = open(str(path_log / args.wrong_format), 'w')
    logger.info('Start convert dataset structure')
    convert_folder_2_flatten(args.root_dir, args.output_dir, wrong_format)
    wrong_format.close()
    logger.info('End convert dataset structure')

    if args.choose_straight:
        if not os.path.exists(args.straight_dir):
            os.makedirs(args.straight_dir)
        logger.info('Choosing straight faces !')
        choose_straight_face(args.output_dir, args.straight_dir)
        logger.info('Choosing straight faces is done')

