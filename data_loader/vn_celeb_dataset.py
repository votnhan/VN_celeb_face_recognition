import torch
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms as tf
from PIL import Image
from utils import read_json

class VNCelebDataset(Dataset):
    def __init__(self, data_dir, label_file, transforms=None):
        self.data_dir = Path(data_dir)
        self.label_dict = read_json(label_file)
        self.transforms = transforms
        self.n_samples = sum([len(v) for k, v in self.label_dict.items()])
        self.n_classes = len(self.label_dict.keys())
        self.img_names, self.labels = self._get_list_samples_labels()


    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.labels[index]
        img_path = self.data_dir / img_name
        img = Image.open(str(img_path))
        if self.transforms:
            data_tensor = self.transforms(img)
        else:
            transform = tf.ToTensor()
            data_tensor = transform(img)

        return data_tensor, label, str(img_path)


    def __len__(self):
        return self.n_samples

    def _get_list_samples_labels(self):
        samples, labels = [], []
        for k, v in self.label_dict.items():
            sample_for_cls = deepcopy(v)
            sample_for_cls.sort()
            samples += sample_for_cls
            labels += len(sample_for_cls)*[int(k)]
        
        return samples, labels