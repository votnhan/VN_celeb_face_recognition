import torch
import numpy as np
from torchvision import transforms as tf
from .vn_celeb_dataset import VNCelebDataset
from .vn_celeb_emb_dataset import VNCelebEmbDataset
from imgaug import augmenters as iaa
from PIL import Image

fixed_size = 160
sometimes = lambda aug: iaa.Sometimes(0.8, aug)
rank1_VNceleb_aug_obj = iaa.Sequential([
  iaa.Fliplr(0.5),
	sometimes(
		iaa.OneOf([
			iaa.Grayscale(alpha=(0.0, 1.0)),
			iaa.AddToHueAndSaturation((-20, 20)),
			iaa.Add((-20, 20), per_channel=0.5),
			iaa.Multiply((0.5, 1.5), per_channel=0.5),
			iaa.GaussianBlur((0, 2.0)),
			iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
			iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
			iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
		])
	)
])

def fix_std(img):
  return (img - 127.5) / 128


def to_tensor(arr):
  transposed = np.transpose(arr, (2, 0, 1))
  tensor = torch.from_numpy(transposed)
  return tensor


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 


def rank1_VN_celeb_aug(x):
  arr = np.array(x)
  tf_img = rank1_VNceleb_aug_obj.augment_image(arr)
  whiten_tf_img = prewhiten(tf_img)
  return whiten_tf_img


transforms_default = tf.Compose([
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

transforms_rank1_VNceleb_aug = tf.Compose([
  tf.Lambda(rank1_VN_celeb_aug),
  np.float32,
  tf.Lambda(to_tensor)
])


trans_emotion_inf = tf.Compose([
  # tf.Resize(256),
  # tf.CenterCrop(224),
  tf.Resize(224),
  tf.ToTensor(),
  tf.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
  ])


transforms_dict = {
  'default': transforms_default,
  'facenet_aug': transforms_facenet_aug,
  'rank1_aug': transforms_rank1_VNceleb_aug,
  'emotion_inf': trans_emotion_inf
}
