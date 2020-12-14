import argparse
import logging
import models as model_md 
import os
import pandas as pd
import requests
import json
import cv2
from datetime import datetime
from logger import setup_logging
from celeb_statistic import export_json_stat_dynamic_itv, export_json_stat_fixed_itv
from utils import read_json, write_json, load_pickle
from demo_image import load_model_classify
from align_face import center_point_dict
from dotenv import load_dotenv
from celeb_statistic import main


# load env
load_dotenv()

def check_free_resource(log_dir, least_memory):
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    temp_file = os.path.join(log_dir, 'gpu_free')
    comm_get_free_mem = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > {}'.format(temp_file)
    os.system(comm_get_free_mem)
    memory_available = [int(x.split()[2]) for x in open(temp_file, 'r').readlines()]
    logger.info('Memory available: {}'.format(memory_available))
    chosen_idx = -1
    for idx, mem in enumerate(memory_available):
        if mem > least_memory:
            chosen_idx = idx
            break
    return chosen_idx


def parse_response(response, str_enc = 'utf-8'):
    decoded_content = response._content.decode(str_enc) 
    dict_content = json.loads(decoded_content)
    return dict_content


def parse_s3_url(s3_url):
    return s3_url


def setup_logger(args):
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    log_dir = os.path.join(args.output_inf_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logging(log_dir)
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    logger.info('Setup logger in directory: {}'.format(log_dir))
    logger.info('Running argument parameters: ')
    for k, v in args.__dict__.items():
        logger.info('--{}: {}'.format(k, v))
    
    return logger, log_dir


def export_json(args, tracker_df, frame_idxes):
    # export JSON file for video celebrity indexing 
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    logger.info('Statistic mode: {}'.format(args.statistic_mode))
    dict_track = {}
    if args.statistic_mode == 'dynamic_itv':
        dict_track = export_json_stat_dynamic_itv(tracker_df, args.n_video_intervals, 
                            args.n_time_appear, args.ignored_name)
    elif args.statistic_mode == 'fixed_itv':
        n_rows_in_itv = args.time_an_interval * len(frame_idxes) * 60
        dict_track = export_json_stat_fixed_itv(tracker_df, n_rows_in_itv, 
                            args.n_time_appear, args.ignored_name)
    else:
        logger.info('This statistic mode {} is not supported !'.format(args.statistic_mode))
    
    return dict_track


def setup_device(detect_model, embedding_model, classify_models, emotion_model,
                    device):
    detect_model.to(device)
    embedding_model.to(device)
    emotion_model.to(device)
    for cls_model in classify_models:
        cls_model.to(device)


def write_JSON_to_DB(data, vod_id):
    store_url = os.environ['API_URL'] + os.environ['STORE_DATA_URL']
    payload = {
        "data": {vod_id: data},
        "job_id": "2",
        "vod_id": vod_id
    }
    response = requests.post(store_url, json=payload)
    dict_content = parse_response(response)
    return dict_content


def while_loop(args, detect_model, embedding_model, classify_models, emotion_model, 
                log_dir):
    # logger
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    # label to name DB (for demostration)
    label2name_df = pd.read_csv(args.label2name)
    # center point, face size after alignment
    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]
    # choose frames for a second
    frame_idxes = list(args.frame_idxes) 
    
    # payload for get job
    job_payload = {
        'type': args.queue_idx
    }
    # get job url
    get_job_url = os.environ['API_URL'] + os.environ['GET_JOB_URL']
    logger.info('Start indexing loop')

    gpu_idx = check_free_resource(log_dir, args.least_memory)
    if gpu_idx >= 0:
        device = 'cuda:{}'.format(gpu_idx)
        setup_device(detect_model, embedding_model, classify_models, emotion_model, 
                        device)
        logger.info('Using device {} !'.format(device))
    else:
        logger.info('Can not find any free GPU')
        return 
    
    response = requests.post(get_job_url, json=job_payload)
    dict_content = parse_response(response)['job']
    if dict_content is None:
        logger.info('Job queue is empty now')
        return 
    
    video_url = dict_content['video_url']
    stream_url = parse_s3_url(video_url)
    
    # check if stream_url work
    cap_check = cv2.VideoCapture(stream_url)
    if not cap_check.isOpened():
        logger.info('Can not stream video from link {} .'.format(video_url))
        return
    else:
        width  = cap_check.get(3)
        height = cap_check.get(4)
        logger.info('Video resolution: {}x{} px'.format(width, height))

    args.video_path = stream_url

    # Set up tracker file
    output_tracker = '{}.csv'.format(dict_content['source_id'])
    args.output_tracker = os.path.join(log_dir, output_tracker)
    
    logger.info('Start indexing video: {}'.format(dict_content['source_id']))
    tracker_df = main(args, detect_model, embedding_model, classify_models, 
                        emotion_model, None, device, label2name_df, target_fs, 
                        center_point, frame_idxes)
    logger.info('End indexing video: {}'.format(dict_content['source_id']))

    # Set up JSON file
    output_json = '{}.json'.format(dict_content['source_id'])
    args.json_tracker =  os.path.join(log_dir, output_json)
    
    logger.info('Start doing statistic on video: {}'.format(dict_content['source_id']))
    dict_track = export_json(args, tracker_df, frame_idxes)
    logger.info('End doing statistic on video: {}'.format(dict_content['source_id']))

    logger.info('Start writing JSON file of: {}'.format(dict_content['source_id']))
    # write_json(args.json_tracker, dict_track, True)
    store_res =  write_JSON_to_DB(dict_track, dict_content['source_id'])
    if store_res['result'] == '1':
        logging.info('-- Writing JSON file to DB successfully')
    else:
        logging.info('-- {}'.format(store_res['message']))
    logger.info('End writing JSON file of: {}'.format(dict_content['source_id']))
    logger.info('Indexing for {} is done ...'.format(dict_content['source_id']))


if __name__ == '__main__':
    # arguments descriptions
    args_parser = argparse.ArgumentParser(description='VN celebrity face and emotion')
    args_parser.add_argument('-m', '--classify_model', nargs='+', type=str)
    args_parser.add_argument('-l2n', '--label2name', default='label2name.csv', 
                                type=str)
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-sfr', '--save_frame_recognized', 
                                action='store_true')
    args_parser.add_argument('-det', '--detection', default='RetinaFace', type=str)
    args_parser.add_argument('-enc', '--encoder', default='iresnet100', type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/embedding/iresnet100_enc.json', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    args_parser.add_argument('-fidx','--frame_idxes', nargs='+', type=int)
    args_parser.add_argument('-ign', '--ignored_name', default='Unknown', 
                                type=str)
    args_parser.add_argument('-nvi', '--n_video_intervals', default=5, type=int)
    args_parser.add_argument('-tap', '--n_time_appear', default=8, type=int)
    args_parser.add_argument('--statistic_mode', default='dynamic_itv', type=str, 
                                help='dynamic_itv or fixed_itv')
    args_parser.add_argument('--time_an_interval', default=5, type=int)
    args_parser.add_argument('--inference_method', default='par_fd_vs_aln', type=str)
    args_parser.add_argument('--min_dim_box', default=50, type=int)
    args_parser.add_argument('--box_ratio', default=2.0, type=float)
    args_parser.add_argument('--log_step', default=100, type=int)
    args_parser.add_argument('--recog_threshold', default=0.7, type=float)
    args_parser.add_argument('--local_thresholds', default='', type=str)
    args_parser.add_argument('--track_bbox', action='store_true')
    args_parser.add_argument('--recog_emotion', action='store_true')
    args_parser.add_argument('-emt', '--emotion', default='resnet_2branch_50', 
                                type=str)
    args_parser.add_argument('-emtargs', '--emotion_args', 
                                default='cfg/emotion/resnet50_2_branch.json', 
                                type=str)
    args_parser.add_argument('-t2i', '--etag2idx_file', 
                        default='meta_data/emotion_recognition/etag2idx.pkl.keep', 
                        type=str)
    args_parser.add_argument('--topk_emotions', default=6, type=int)
    args_parser.add_argument('--n_frames', default=16, type=int)
    args_parser.add_argument('--multi_gpus_idx', nargs='+', type=int)
    args_parser.add_argument('--output_inf_dir', default='output_demo', type=str)
    args_parser.add_argument('--least_memory', default=9500, type=int)
    args_parser.add_argument('--queue_idx', default=2, type=int)

    args = args_parser.parse_args()

    # set up logger for inference
    logger, log_dir = setup_logger(args)
    args.logger_id = os.environ['LOGGER_ID']

    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()
    logger.info('Loading detection model {} done ...'.format(args.detection))

    # face embedding model
    enc_args = read_json(args.encoder_args)
    emb_model = getattr(model_md, args.encoder)(**enc_args)
    logger.info('Loading embedding model {} done ...'.format(args.encoder))

    # emotion model (if need)
    if args.recog_emotion:
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args)
        logger.info('Loading emotion model {} is done ...'.format(args.emotion))

    
    # classify from embedding model
    cls_model_paths = list(args.classify_model)
    classify_models = []
    for path in cls_model_paths:
        classify_model = model_md.MLPModel(args.input_dim_emb, args.num_classes)
        load_model_classify(path, classify_model, os.environ['LOGGER_ID'])
        classify_models.append(classify_model)
    logger.info('Loading mlp models done ...')
    
    # Process loop 
    while_loop(args, detection_md, emb_model, classify_models, emt_model, log_dir)
    line = ['-']*100
    logger.info(''.join(line))
