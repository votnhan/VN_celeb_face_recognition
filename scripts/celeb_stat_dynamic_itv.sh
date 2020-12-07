python celeb_statistic.py -i ../videos_test/TranThanhHienHoErik.mp4 \
-ot tracker_tthher_3.csv -m ../weights/face_recognition/mlp_1020_celeb.pth \
-l2n meta_data/face_recognition/label2name_1020_cls.txt -nc 1020 -enc iresnet100 \
-jst tracker_tthher_3.json -fidx 1 6 11 16 -ign Unknown -nvi 8 -det RetinaFace \
-dargs cfg/detection/retina_face.json --inference_method par_fd_vs_aln \
--log_step 100 --recog_threshold 0.7 -tap 8 --track_bbox --topk_emotions 6 \
--recog_emotion --statistic_mode dynamic_itv \
--n_frames 120
