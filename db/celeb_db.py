import sys
sys.path.append('./')
import pandas as pd
import os
from utils import read_json


class CelebDB():
    def __init__(self, label2name_path, alias2main_id_path, include_name=False):
        self.set_attributes(label2name_path, alias2main_id_path, include_name)

    
    def set_attributes(self, label2name_path, alias2main_id_path, include_name=False):
        self.label2name_path = label2name_path
        self.alias2main_id_path = alias2main_id_path
        self.include_name = include_name

    def load_db(self):
        self.label2name_df = pd.read_csv(self.label2name_path)
        self.alias2main_id = read_json(self.alias2main_id_path)
        self.unknown_cls = self.label2name_df['label'].iloc[-1]

    def get_id_of_celeb(self, predictions):
        list_names = []
        name_df = self.label2name_df
        for pred in predictions:
            main_id = int(self.alias2main_id[str(pred)])
            celeb_info = '{}'.format(main_id)
            if self.include_name:
                name = list(name_df['name'][name_df['label'] == main_id])
                if len(name) > 0:
                    celeb_info = '{}:{}'.format(main_id, name[0])
                else:
                    celeb_info = '{}:Unknown'.format(self.unknown_cls)
            list_names.append(celeb_info)
        return list_names

    def get_infor_of_celeb(self, celeb_id):
        chosen = self.label2name_df['label'] == celeb_id
        name = list(self.label2name_df['name'][chosen])[0]
        return {
            'id': str(celeb_id),
            'name': name
        }
