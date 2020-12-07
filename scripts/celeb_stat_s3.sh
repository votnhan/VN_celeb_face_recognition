python celeb_statistic.py --s3_video --s3_video_infor cfg/s3_storage/video_1.json \
-ot video_1.csv -m ../weights/face_recognition/checkpoint-epoch9.pth \
-l2n meta_data/face_recognition/label2name_1020_cls.txt -nc 1020 -enc iresnet100 \
-jst video_1_2.json -fidx 1 6 11 16 -ign Unknown -nvi 8 -det RetinaFace \
-dargs cfg/detection/retina_face.json --inference_method par_fd_vs_aln \
--log_step 100 --recog_threshold 0.7 -tap 8 --track_bbox --topk_emotions 6 \
--recog_emotion --statistic_mode dynamic_itv \
--n_frames 120