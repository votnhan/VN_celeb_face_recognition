python celeb_statistic.py -i ../videos_test/MotNamMoiBinhAn.webm \
-ot tracker_1.csv -m ../weights/face_recognition/checkpoint-epoch9.pth \
-l2n meta_data/face_recognition/label2name_1020_cls.txt -nc 1020 -enc iresnet100 \
-jst tracker_1.json -fidx 1 6 11 16 -ign Unknown -det RetinaFace \
-dargs cfg/detection/retina_face.json --inference_method par_fd_vs_aln \
--log_step 100 --recog_threshold 0.7 -tap 8 --track_bbox --topk_emotions 6 \
--recog_emotion --statistic_mode fixed_itv --time_an_interval 1 \
--n_frames 120
