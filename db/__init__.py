from .celeb_db import CelebDB
from .emotion_db import EmotionDB

celeb_db = CelebDB('', '')
emotion_db = EmotionDB('', '')


def set_params_celeb_db(label2name_path, alias2main_id_path, include_name):
    celeb_db.set_attributes(label2name_path, alias2main_id_path, include_name)
    celeb_db.load_db()


def set_params_emotion_db(etag2idx_path, emotion_label_path, include_emt=False, 
                            sheet_name='690_emotions'):
    emotion_db.set_params(etag2idx_path, emotion_label_path, include_emt, 
                            sheet_name)
    emotion_db.load_db()
