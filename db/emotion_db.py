import sys
sys.path.append('./')
import pandas as pd
import os
import numpy as np
from utils import load_pickle


class EmotionDB():
    def __init__(self, etag2idx_path, emotion_label_path, include_emt=False, 
                    sheet_name='690_emotions'):
        self.set_params(etag2idx_path, emotion_label_path, include_emt=False, 
            sheet_name='690_emotions')

    
    def set_params(self, etag2idx_path, emotion_label_path, include_emt=False, 
                    sheet_name='690_emotions'):
        self.emotion_label_path = emotion_label_path
        self.sheet_name = sheet_name
        self.etag2idx_path = etag2idx_path
        self.include_emt = include_emt

    
    def load_db(self):
        emt_label_xls = pd.ExcelFile(self.emotion_label_path)
        self.emt_label_df = emt_label_xls.parse(self.sheet_name)
        self.idx2etag = load_pickle(self.etag2idx_path)['idx2key']
        if self.include_emt:
            self.map_func = np.vectorize(lambda x: '{}:{}'.format(x, self.idx2etag[x]))
        else:
            self.map_func = np.vectorize(lambda x: str(x))


    def get_full_infor_of_emotion(self, id_emotion):
        chosen = self.emt_label_df['id'] == id_emotion
        english_key = list(self.emt_label_df['emotion'][chosen])[0]
        vn_keys = list(self.emt_label_df['vietnamese'][chosen])[0]
        if pd.isna(vn_keys):
            vn_keys = ''
        return {
            'id': str(id_emotion),
            'english': english_key,
            'vietnamese': vn_keys
        }
