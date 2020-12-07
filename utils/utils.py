import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import glob
import shutil
import pafy
import pickle as pkl
from PIL import Image
from s3_utils import s3_url_generator


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def read_json(filename):
    with open(filename, 'r') as fp:
        content =  json.load(fp)

    return content

def write_json(filename, content_dict, log=True):
    with open(filename, 'w') as fp:
        json.dump(content_dict, fp, ensure_ascii=False, indent=True)

    if log:
        print('Write json file {}'.format(filename))

def create_folder(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)

def save_pandas_df(data, filename, columns, index=None, use_index=True):
    df = pd.DataFrame(data=data, index=index, columns=columns)
    df.to_csv(filename, index=use_index)

def read_image(image_path):
    image = Image.open(image_path)
    return image

def append_log_to_file(file_path, list_items):
    with open(file_path, 'a') as opened_file:
        line_items = ','.join(list_items)
        opened_file.write(line_items+'\n')
        opened_file.close()

def plot_train_val_loss(log_file, out_file):
    df = pd.read_csv(log_file, index_col='Epoch')
    plt.plot(df['Train_loss'].values, label='Training loss')
    plt.plot(df['Validation_loss'].values, label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(out_file)
    print('Plot train and val loss to {}'.format(out_file))


def convert_sec_to_max_time_quantity(second):
    h = second // 3600
    remain_time = second % 3600
    m = remain_time // 60
    s = remain_time % 60
    return '{}h:{}m:{:.2f}s'.format(h, m, s)


def convert_ds_folder_2_def_structure(root_dir, output_dir, label_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_str = root_dir + '/*/*'
    image_paths = glob.glob(path_str)
    image_paths.sort()
    n_images = len(image_paths)
    label_list = []
    for idx, image_path in enumerate(image_paths):
        print('-----{}/{}-----'.format(idx, n_images))
        print('Copying file {}'.format(image_path))
        label, image_file = image_path.split('/')[-2: ]
        image_name, ext = image_file.split('.')
        new_image_file = '{}_{}.{}'.format(label, image_name, ext)
        new_img_path = os.path.join(output_dir, new_image_file)
        shutil.copyfile(image_path, new_img_path)
        label_list.append((new_image_file, int(label)-1))

    label_df = pd.DataFrame(data=label_list, columns=['image', 'label'])
    label_df.to_csv(label_file, index=False)
    print('Saved label file {}.'.format(label_file))


def convert_id_ds_2_def_structure(root_dir, output_dir, wrong_format):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_str = root_dir + '/*/*'
    image_paths = glob.glob(path_str)
    n_images = len(image_paths)
    wrong_format_counter = 0
    for idx, image_path in enumerate(image_paths):
        if not os.path.isfile(image_path):
            continue
        print('-----{}/{}-----'.format(idx, n_images))
        print('Copying file {}'.format(image_path))
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


def load_pickle(save_file):
    with open(save_file, "rb") as of_:
        return  pkl.load(of_)


def generate_url_video(args, s3_client):
    video_url = ''
    if args.youtube_video:
        pafy_obj = pafy.new(args.youtube_link)
        play = pafy_obj.getbest(preftype="mp4")
        if play is None:
            print('This Youtube video did not support mp4 format !')
            return 
        print('Video resolution: {}, video format: {}'.format(play.resolution, 
                play.extension))
        video_url = play.url
    elif args.s3_video:
        video_infor = read_json(args.s3_video_infor)
        video_url = s3_url_generator(s3_client, video_infor['bucket'], 
                                    video_infor['file_name'])
    else:
        video_url = args.video_path

    return video_url
