python celeb_statistic.py -i https://s3.vieon.vn/ott-vod-2020/11/17/rap-viet-v2-2020-s01-ep01-124a610bd99821ed9da72539a26c16e1/video/avc1/zbqrpd/profile.m3u8 \
-ot tracker_rap_viet_2.csv -m ../weights/face_recognition/checkpoint-epoch14.pth \
-l2n meta_data/face_recognition/label2name_52_cls.txt -nc 52 -enc iresnet100 \
-jst tracker_rap_viet_2.json -fidx 1 6 11 16 -ign Unknown -det RetinaFace \
-dargs cfg/detection/retina_face.json --inference_method par_fd_vs_aln \
--log_step 100 --recog_threshold 0.7 -tap 8 --track_bbox --topk_emotions 6 \
--recog_emotion --statistic_mode fixed_itv --time_an_interval 5 \
--n_frames 120
