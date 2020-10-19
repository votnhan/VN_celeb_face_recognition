import torch
import numpy as np
from .vn_celeb_dataset import VNCelebDataset
from pathlib import Path

class VNCelebEmbDataset(VNCelebDataset):
    def __init__(self, data_dir, label_file, transforms=None):
        super().__init__(data_dir, label_file, transforms)

    def __getitem__(self, index):
        emb_name = self.img_names[index].split('.')[0]
        label = self.labels[index]
        emb_path = self.data_dir / '{}.npz'.format(emb_name)
        emb = np.load(str(emb_path))['arr_0']
        
        if self.transforms:
            data_tensor = self.transforms(emb)
        else:
            data_tensor = torch.from_numpy(emb)

        return data_tensor, label, str(emb_path)
