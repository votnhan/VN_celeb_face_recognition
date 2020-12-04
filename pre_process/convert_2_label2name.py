import argparse
import os
import pandas as pd
from utils import read_json


def convert_xlsx_2_label2name(xlsx_file_path, sheet_name):
    xls_obj = pd.ExcelFile(xlsx_file_path)
    sheet_df = xls_obj.parse(sheet_name)
    label_2_name_df = pd.DataFrame(data=zip(sheet_df['Id'], sheet_df['Label']), 
                            columns=['label', 'name'])
    label_2_name_df.dropna(inplace=True)
    return label_2_name_df


def convert_2_seq_label2name(label_2_name_df, idx2label_dict):
    label2name = []
    for k, v in idx2label_dict.items():
        chosen = label_2_name_df['label'] == v
        name = list(label_2_name_df['name'][chosen])[0]
        label2name.append((int(k), name))

    label2name_df = pd.DataFrame(data=label2name, columns=['label', 'name'])
    return label2name_df


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Convert descriptions file label-name format')
    args_parser.add_argument('--xlsx', default='celeb_descriptions.xlsx', type=str)
    args_parser.add_argument('--sheet_name', default='Sheet1', type=str)
    args_parser.add_argument('--main_label2name', default='label2name_main.txt', 
                                type=str)
    args_parser.add_argument('--seq_label2name', default='label2name_seq.txt', 
                                type=str)                              
    args_parser.add_argument('--idx2label', default='label2name_main.json', 
                                type=str)

    args = args_parser.parse_args()
    label_2_name_df = convert_xlsx_2_label2name(args.xlsx, args.sheet_name)
    label_2_name_df.to_csv(args.output_path, index=False)
    print('Converted xlsx file {} to {} with label-name format'.format(args.xlsx, 
            args.main_label2name))
    
    idx2label_dict = read_json(args.idx2label)
    label_2_name_seq_df = convert_2_seq_label2name(label_2_name_df, idx2label_dict)
    label_2_name_seq_df.to_csv(args.seq_label2name, index=False)
    print('Converted main label2name file {} to sequential label2name {}'.\
            format(args.main_label2name, args.seq_label2name))
