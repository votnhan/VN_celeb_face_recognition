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
    
    write_json(output_file, dict_labels, False)
    
    print('Created {} for describe VN_celeb ...'.format(output_file))


def split_train_val(desc_file, output_train, output_val):
    dict_labels = read_json(desc_file)

    dict_train, dict_val = {}, {}
    for k, v in dict_labels.items():
        dict_train[k] = v[:-1]
        dict_val[k] = [v[-1]]

    write_json(output_train, dict_train)
    write_json(output_val, dict_val)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Split training \
                    and validation set for VN celeb dataset')
    args_parser.add_argument('-d', '--describe_file', default='train.csv', 
                            help='File describes train images and labels')

    args_parser.add_argument('-o', '--out_dict_labels', default='vn_celeb.json',
                            help='JSON file contains labels and their images')

    args_parser.add_argument('-tr', '--train_file', default='train.json')
    args_parser.add_argument('-v', '--val_file', default='val.json')

    args = args_parser.parse_args()
    
    create_file_describe_ds(args.describe_file, args.out_dict_labels)
    split_train_val(args.out_dict_labels, args.train_file, args.val_file)

