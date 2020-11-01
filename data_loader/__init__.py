import torch
import numpy as np
from torchvision import transforms as tf
from .vn_celeb_dataset import VNCelebDataset
from .vn_celeb_emb_dataset import VNCelebEmbDataset 
from PIL import Image

fixed_size = 160

def fix_std(img):
  return (img - 127.5) / 128

def to_tensor(arr):
  transposed = np.transpose(arr, (2, 0, 1))
  tensor = torch.from_numpy(transposed)
  return tensor

transforms_default = tf.Compose([
    tf.Resize(fixed_size),
    np.float32,
    tf.Lambda(fix_std),
    tf.Lambda(to_tensor)
])

transforms_facenet_aug = tf.Compose([
  tf.RandomRotation(degrees=(-10, 10), resample=Image.BICUBIC),
  tf.RandomCrop(size=fixed_size, padding=2, pad_if_needed=True),
  tf.RandomHorizontalFlip(p=0.5),
  np.float32,
  tf.Lambda(fix_std),
  tf.Lambda(to_tensor) 
])

transforms_dict = {
  'default': transforms_default,
  'facenet_aug': transforms_facenet_aug
}
