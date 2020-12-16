import sys
sys.path.append('./')
import pandas as pd
import os
from utils import read_json


class CelebDB():
    def __init__(self, label2name_path, alias2main_id_path):
        self.label2name_df = pd.read_csv(label2name_path)
        self.alias2main_id = read_json(alias2main_id_path)


    def get_id_of_celeb(self, predictions):
        list_names = []
        name_df = self.label2name_df
        for pred in predictions:
            main_id = int(self.alias2main_id[pred])
            name = list(name_df['name'][name_df['label'] == main_id])
            if len(name) > 0:
                id_name = '{}:{}'.format(main_id, name)
                list_names.append(id_name)
            else:
                list_names.append('Unknown')
        return list_names
