import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from utils import read_json, write_json


def create_file_describe_ds(describe_file, output_file):
    df_label = pd.read_csv(describe_file)
    labels = np.unique(df_label['label'])
    dict_labels = {}
    for i in labels:
        chosen = df_label['label'] == i
        img_for_i = df_label['image'][chosen]
        dict_labels[str(i)] = list(img_for_i)
    
    write_json(output_file, dict_labels)
    print('Created {} for describe VN_celeb ...'.format(output_file))
    return dict_labels


def split_train_val(desc_file, output_train, output_val):
    dict_labels = read_json(desc_file)

    dict_train, dict_val = {}, {}
    for k, v in dict_labels.items():
        if len(v) > 1:
            dict_train[k] = v[:-1]
            dict_val[k] = [v[-1]]
        else:
            dict_train[k] = [v[0]]

    write_json(output_train, dict_train)
    write_json(output_val, dict_val)
    return dict_train, dict_val


def remap_sequence_key(label_dict):
    remap_dict = {}
    for idx, key in enumerate(label_dict.keys()):
        remap_dict[str(idx)] = label_dict[key]
    return remap_dict


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Split training \
                    and validation set for VN celeb dataset')
    args_parser.add_argument('-d', '--describe_file', default='train.csv', 
                            help='File describes train images and labels')

    args_parser.add_argument('-o', '--out_dict_labels', default='vn_celeb.json',
                            help='JSON file contains labels and their images')

    args_parser.add_argument('-tr', '--train_file', default='train.json')
    args_parser.add_argument('-v', '--val_file', default='val.json')
    args_parser.add_argument('--remap_key', action='store_true')

    args = args_parser.parse_args()
    
    dict_labels = create_file_describe_ds(args.describe_file, 
                    args.out_dict_labels)
    dict_train, dict_val = split_train_val(args.out_dict_labels, 
                                args.train_file, args.val_file)
    
    if args.remap_key:
        seq_key_dict_labels = remap_sequence_key(dict_labels)
        seq_key_dict_train = remap_sequence_key(dict_train)
        seq_key_dict_val = remap_sequence_key(dict_val)
        write_json('{}_remap.json'.format(args.describe_file.split('.')[0]), 
                    seq_key_dict_labels)
        write_json('{}_remap.json'.format(args.train_file.split('.')[0]), 
                    seq_key_dict_train)
        write_json('{}_remap.json'.format(args.val_file.split('.')[0]), 
                    seq_key_dict_val)       

