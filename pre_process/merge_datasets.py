import sys
sys.path.append('./')
import os
import glob
import argparse
import shutil
import logging
from dotenv import load_dotenv
from logger import get_logger_for_run

load_dotenv()


def merge_dataset(new_ds_dir, old_ds_dir):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    path_pt = new_ds_dir + '/*'
    img_paths = glob.glob(path_pt)
    img_paths.sort()
    n_images = len(img_paths)
    for idx, img_path in enumerate(img_paths):
        img_file = img_path.split('/')[-1]
        new_img_path = os.path.join(old_ds_dir, img_file)
        if not os.path.exists(new_img_path):
            logger.info('{}/{}, Copying new image {}'.format(idx, n_images, img_path))
            shutil.copy(img_path, new_img_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Merge old and new dataset')
    args_parser.add_argument('--new_ds_dir', default='align_output', type=str)
    args_parser.add_argument('--old_ds_dir', default='main_version', type=str)
    args_parser.add_argument('--output_log', default='merge_log', type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str) 

    args = args_parser.parse_args()

    args.output_log = os.path.join(args.root_output, args.output_log)
    args.new_ds_dir = os.path.join(args.root_output, args.new_ds_dir)
    logger, log_dir = get_logger_for_run(args.output_log)

    logger.info('Start merging 2 datasets')
    merge_dataset(args.new_ds_dir, args.old_ds_dir)
    logger.info('End merging 2 datasets')
    