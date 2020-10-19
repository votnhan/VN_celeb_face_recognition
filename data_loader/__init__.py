import torch
import numpy as np
from torchvision import transforms as tf
from .vn_celeb_dataset import VNCelebDataset
from .vn_celeb_emb_dataset import VNCelebEmbDataset 

fixed_size = 160

def fix_std(img):
  return (img - 127.5) / 128

def to_tensor(arr):
  transposed = np.transpose(arr, (2, 0, 1))
  tensor = torch.from_numpy(transposed)
  return tensor

transforms = tf.Compose([
    tf.Resize(fixed_size),
    np.float32,
    tf.Lambda(fix_std),
    tf.Lambda(to_tensor)
])
