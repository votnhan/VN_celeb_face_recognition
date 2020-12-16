import pandas as pd
import numpy as np
import argparse
import logging
import os
from pathlib import Path
from utils import read_json, write_json
from dotenv import load_dotenv
from logger import get_logger_for_run

load_dotenv()

def create_file_describe_ds(describe_file, output_file):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    df_label = pd.read_csv(describe_file)
    labels = np.unique(df_label['label'])
    dict_labels = {}
    label_2_seq_dict = {}
    for idx, i in enumerate(labels):
        chosen = df_label['label'] == i
        img_for_i = df_label['image'][chosen]
        dict_labels[str(i)] = list(img_for_i)
        label_2_seq_dict[str(idx)] = str(i)
    
    write_json(output_file, dict_labels)
    logger.info('Created {} for describe VN_celeb ...'.format(output_file))
    return dict_labels, label_2_seq_dict


def split_train_val(desc_file, output_train, output_val, n_samples_val=1):
    dict_labels = read_json(desc_file)
    dict_train, dict_val = {}, {}
    for k, v in dict_labels.items():
        if len(v) > n_samples_val:
            dict_train[k] = v[:-n_samples_val]
            dict_val[k] = v[-n_samples_val:]
        else:
            dict_train[k] = v[:-1]
            dict_val[k] = [v[-1]]

    write_json(output_train, dict_train)
    write_json(output_val, dict_val)
    return dict_train, dict_val


def remap_sequence_key(label_dict):
    remap_dict = {}
    for idx, key in enumerate(label_dict.keys()):
        remap_dict[str(idx)] = label_dict[key]
    return remap_dict


def get_label_from_dataset(root_dir, label_file):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    images = os.listdir(root_dir)
    images.sort()
    label_items = []
    for img_file in images:
        label = img_file.split('_')[0]
        label_items.append((img_file, int(label)))
    
    label_df = pd.DataFrame(data=label_items, columns=['image', 'label'])
    label_df.to_csv(label_file, index=False)
    logger.info('Saved label file {}.'.format(label_file))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Split training \
                    and validation set for VN celeb dataset')
    args_parser.add_argument('-d', '--describe_file', default='train.csv', 
                            help='File describes train images and labels')

    args_parser.add_argument('-o', '--out_dict_labels', default='vn_celeb.json',
                            help='JSON file contains labels and their images')
    args_parser.add_argument('--label_2_seq', default='label_2_seq.json', 
                                type=str)
    args_parser.add_argument('-tr', '--train_file', default='train.json')
    args_parser.add_argument('-v', '--val_file', default='val.json')
    args_parser.add_argument('--n_samples_val', default=1, type=int)
    args_parser.add_argument('--output_log', default='split_log', type=str)   
    args_parser.add_argument('--root_dir', default='data', type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str)

    args = args_parser.parse_args()
    root_output = Path(args.root_output)
    args.describe_file = str(root_output / args.describe_file)
    args.out_dict_labels = str(root_output / args.out_dict_labels)
    args.label_2_seq = str(root_output / args.label_2_seq)
    args.train_file = str(root_output / args.train_file)
    args.val_file = str(root_output / args.val_file)
    args.output_log = str(root_output / args.output_log)

    logger, log_dir = get_logger_for_run(args.output_log)
    logger.info('Split train and val set')
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))

    if not os.path.exists(args.describe_file):
        logger.info('Create celeb description csv file from {}'.format(args.root_dir))
        get_label_from_dataset(args.root_dir, args.describe_file)

    logger.info('Create JSON description of label')
    dict_labels, label_2_seq_dict = create_file_describe_ds(args.describe_file, 
                                        args.out_dict_labels)
    seq_key_dict_labels = remap_sequence_key(dict_labels)

    logger.info('Writing JSON mapping between not sequent ID and sequent ID')
    write_json(args.label_2_seq, label_2_seq_dict, log=True)
    logger.info('Writing JSON describing dataset with sequentially classes')
    write_json(args.out_dict_labels, seq_key_dict_labels, log=True)

    logger.info('Splitting train and val set')
    dict_train, dict_val = split_train_val(args.out_dict_labels, args.train_file, 
                                args.val_file, args.n_samples_val)
     
