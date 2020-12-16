import sys
sys.path.append('./')
import argparse
import os
import pandas as pd
from logger import get_logger_for_run
from utils import read_json
import logging
from dotenv import load_dotenv

load_dotenv()


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
        chosen = label_2_name_df['label'] == int(v)
        name = list(label_2_name_df['name'][chosen])[0]
        label2name.append((int(k), name))

    label2name_df = pd.DataFrame(data=label2name, columns=['label', 'name'])
    return label2name_df, len(idx2label_dict.keys())


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Convert descriptions file label-name format')
    args_parser.add_argument('--xlsx', default='celeb_descriptions.xlsx', type=str)
    args_parser.add_argument('--sheet_name', default='Sheet1', type=str)
    args_parser.add_argument('--main_label2name', default='label2name_main.txt', 
                                type=str)
    args_parser.add_argument('--seq_label2name', default='label2name_seq.txt', 
                                type=str)                              
    args_parser.add_argument('--alias2main_id', default='alias2main_id.json', 
                                type=str)
    args_parser.add_argument('--output_log', default='label2name_cvt', type=str)
    args_parser.add_argument('--root_output', default='temp_data', type=str)

    args = args_parser.parse_args()
    args.output_log = os.path.join(args.root_output, args.output_log)
    logger, log_dir = get_logger_for_run(args.output_log)
    logger.info('Convert file {} to label2name format'.format(args.xlsx))
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))
    
    args.main_label2name = os.path.join(args.root_output, args.main_label2name)
    label_2_name_df = convert_xlsx_2_label2name(args.xlsx, args.sheet_name)
    label_2_name_df.to_csv(args.main_label2name, index=False)
    logger.info('Converted xlsx file {} to {} with label-name format'.format(args.xlsx, 
            args.main_label2name))
    
    args.alias2main_id = os.path.join(args.root_output, args.alias2main_id)
    args.seq_label2name = os.path.join(args.root_output, args.seq_label2name)
    idx2label_dict = read_json(args.alias2main_id)
    label_2_name_seq_df, n_celeb = convert_2_seq_label2name(label_2_name_df, idx2label_dict)
    label_2_name_seq_df.to_csv(args.seq_label2name, index=False)
    logger.info('Converted main label2name file {} to sequential label2name {}'.\
            format(args.main_label2name, args.seq_label2name))
    logger.info('Number of celebrities now: {}'.format(n_celeb))

