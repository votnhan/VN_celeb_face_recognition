import argparse
import os
import cv2
import subprocess
import celeb_statistic as cl_stat
import pandas as pd
import json
import time
import logging
from utils import read_json, write_json
from logger import get_logger_for_run
from dotenv import load_dotenv

load_dotenv()

def read_basic_command(command_path):
    with open(command_path, 'r') as cmd_file:
        command = cmd_file.read()
    return command


def get_process_specific_params(gpu_idx, base_idx, frames_4_seg, args):
    inp_video = '-i {} \\'.format(args.video_path)
    dev_cmd = '--device cuda:{} \\'.format(gpu_idx)
    start_frame_cmd = '--start_frame {} \\'.format(base_idx)
    frames_4_seg_cmd = '--n_segment_frames {} \\'.format(frames_4_seg)
    temp = args.output_tracker.split('/')
    tracker_name, ext = temp[-1].split('.')
    tk_4_process = '{}_{}.{}'.format(tracker_name, gpu_idx, ext)
    tk_4_process_path = '/'.join(temp[:-1]) + '/' + tk_4_process
    tracker_file_cmd = '--output_tracker {} '.format(tk_4_process_path)
    cmd = '\n'.join([inp_video, dev_cmd, start_frame_cmd, frames_4_seg_cmd, 
                        tracker_file_cmd])
    return cmd, tk_4_process_path


def main(args, n_frames, basic_command):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    temp = list(range(args.n_gpus))
    n_segment_frames = int(n_frames // args.n_gpus)
    n_remain = n_frames % args.n_gpus
    base_idx = [x*n_segment_frames for x in temp]
    frames_4_seg = (args.n_gpus - 1)*[n_segment_frames]
    frames_4_seg.append(n_segment_frames + n_remain) 

    logger.info('Parallel processing starts')
    processes_ref = []
    tracker_files = []
    start_time = time.time()
    for gpu_idx in range(args.n_gpus):
        gpu_cmd, tracker_file = get_process_specific_params(gpu_idx, base_idx[gpu_idx], 
                                            frames_4_seg[gpu_idx], args)
        tracker_files.append(tracker_file)
        main_cmd = basic_command + gpu_cmd
        process = subprocess.Popen(main_cmd, shell=True)
        processes_ref.append(process)

    # Wait for proceses to finish
    output = [p.wait() for p in processes_ref]
    end_time = time.time()
    time_of_process = end_time - start_time
    logger.info('Parallel processing ends')
    logger.info('Time for parallel processing: {:0.2f}'.format(time_of_process))

    logger.info('Collecting tracker files starts')
    tracker_df_list = []
    for idx, tracker_file in enumerate(tracker_files):
        process_tracker_df = pd.read_csv(tracker_file)
        tracker_df_list.append(process_tracker_df)
    tracker_df = pd.concat(tracker_df_list)
    tracker_df.to_csv(args.output_tracker, index=False)
    logger.info('Collecting tracker files ends')
    
    logger.info('Statistic mode: {}'.format(args.statistic_mode))
    dict_track = {}
    if args.statistic_mode == 'dynamic_itv':
        dict_track = cl_stat.export_json_stat_dynamic_itv(tracker_df, args.n_video_intervals, 
                            args.n_time_appear, args.ignored_name)
    elif args.statistic_mode == 'fixed_itv':
        frame_idxes = list(args.frame_idxes)
        n_rows_in_itv = args.time_an_interval * len(frame_idxes) * 60
        dict_track = cl_stat.export_json_stat_fixed_itv(tracker_df, n_rows_in_itv, 
                            args.n_time_appear, args.ignored_name)
    else:
        logger.info('This statistic mode {} is not supported !'.format(args.statistic_mode))
    
    write_json(args.json_tracker, dict_track, log=True)

        

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Multi processes when reading video')
    args_parser.add_argument('-i', '--video_path', default='video.mp4', type=str)
    args_parser.add_argument('--n_gpus', default=8, type=int)
    args_parser.add_argument('--basic_command', type=str)
    args_parser.add_argument('-ot', '--output_tracker', default='tracker.csv', 
                                type=str)
    args_parser.add_argument('-nvi', '--n_video_intervals', default=5, type=int)
    args_parser.add_argument('-tap', '--n_time_appear', default=8, type=int)
    args_parser.add_argument('-ign', '--ignored_name', default='Unknown', 
                                type=str)
    args_parser.add_argument('--time_an_interval', default=5, type=int)
    args_parser.add_argument('-jst', '--json_tracker', default='tracker.json', 
                                type=str)
    args_parser.add_argument('--statistic_mode', default='dynamic_itv', type=str, 
                                help='dynamic_itv or fixed_itv')
    args_parser.add_argument('-fidx','--frame_idxes', nargs='+', type=int)
    args_parser.add_argument('--output_inf_dir', default='output_demo', type=str)

    args = args_parser.parse_args()
    logger, log_dir = get_logger_for_run(args.output_inf_dir)
    args.output_tracker = os.path.join(log_dir, args.output_tracker)
    cap = cv2.VideoCapture(args.video_path)
    n_frames = int(cap.get(7))
    logger.info('Frame count: {}, fps of video: {}'.format(n_frames, cap.get(5)))
    basic_command = read_basic_command(args.basic_command)
    main(args, n_frames, basic_command)
    