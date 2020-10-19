from pathlib import Path
from data_loader import transforms
from PIL import Image
from models import InceptionResnetV1
import os
import numpy as np
import torch
import argparse


def create_batch_images(list_files, batch_size):
    n_files = len(list_files)
    n_batchs = n_files // batch_size
    list_batch_files = []
    for i in range(n_batchs):
        img_files = list_files[i*batch_size: i*batch_size+batch_size]
        list_batch_files.append(img_files)

    list_batch_files.append(list_files[n_batchs*batch_size:])
    return list_batch_files, n_batchs


def create_image_tensors(data_dir_path, list_files, transforms):
    list_tensors = []
    for img_file in list_files:
        img_path = str(data_dir_path / img_file)
        img = Image.open(img_path)
        trans_img = transforms(img)
        list_tensors.append(trans_img)

    tensors = torch.stack(list_tensors, 0)
    return tensors

def save_embeddings(embeddings, list_files, output_dir):
    n_emb = embeddings.shape[0]
    output_dir_path = Path(output_dir)
    for i in range(n_emb):
        img_name = list_files[i].split('.')[0]
        emb_name = '{}.npz'.format(img_name)
        emb_path = str(output_dir_path / emb_name)
        np.savez_compressed(emb_path, embeddings[i])
        print('Save embedding for {} ...'.format(list_files[i]))


def cal_embedding(data_dir, batch_size, model, transforms, output_dir, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    list_files = os.listdir(data_dir)
    list_files.sort()
    data_dir_path = Path(data_dir)
    list_batch_files, n_batchs = create_batch_images(list_files, batch_size)
    for idx, batch_file in enumerate(list_batch_files):
        print('Processing for {}/{} batchs:'.format(idx, n_batchs))
        tensors = create_image_tensors(data_dir_path, batch_file, transforms)
        tensors = tensors.to(device)
        embeddings = model(tensors).detach().cpu().numpy()
        save_embeddings(embeddings, batch_file, output_dir)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='Find embedding vertors \
                    for all images in trainning set')

    args_parser.add_argument('-d', '--data_dir', default='train')
    args_parser.add_argument('-bz', '--batch_size', default=10, type=int)
    args_parser.add_argument('-o', '--output_dir', default='train_embedding')
    args_parser.add_argument('-w', '--pre_trained', default='vggface2')
    args_parser.add_argument('-dv', '--device', default='GPU')

    args = args_parser.parse_args()
    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'

    model = InceptionResnetV1(pretrained=args.pre_trained, device=device)
    cal_embedding(args.data_dir, args.batch_size, model, transforms, 
                    args.output_dir, device)


        





