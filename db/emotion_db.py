import sys
sys.path.append('./')
import pandas as pd
import os
import numpy as np
from utils import load_pickle


class EmotionDB():
    def __init__(self, etag2idx_path):
        self.idx2etag = load_pickle(etag2idx_path)
        self.map_func = np.vectorize(lambda x: '{}:{}'.format(x, self.idx2etag[x]))
