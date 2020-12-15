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
        shutil.copyfile(image_path, new_img_path)

    print('Samples wrong format: {}'.format(wrong_format_counter))


if __name__ == '__name__':
    args_parser = argparse.ArgumentParser(description='Convert folder \
                    structure dataset to flatten dataset')
    
    args_parser.add_argument('--root_dir', default='data', type=str)
    args_parser.add_argument('--output_dir', default='flatten_dataset', type=str)
    args_parser.add_argument('--output_log', default='flatten_log', type=str)
    args_parser.add_argument('--wrong_format', default='wrong_format.txt', 
                                type=str)
    
    args = args_parser.parse_args() 

    logger, log_dir = get_logger_for_run(args.output_log)
    path_log = Path(log_dir)
    wrong_format = open(str(path_log / args.wrong_format), 'w')
    convert_folder_2_flatten(args.root_dir, args.output_dir, wrong_format)
    wrong_format.close()
