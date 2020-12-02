import argparse
import os
import glob
import models as model_md
import cv2
import pandas as pd
from utils import read_json
from pathlib import Path


def get_face_from_box(bgr_img, box):
    ori_h, ori_w = bgr_img.shape[:2]
    x1 = max(int(box[0]), 0)
    y1 = max(int(box[1]), 0)
    x2 = min(int(box[2] + 1), ori_w)
    y2 = min(int(box[3] + 1), ori_h)
    face = bgr_img[y1:y2, x1:x2, :]
    return face


def crop_face(input_dir, output_dir, detection_md, unknown_file, many_boxes_file, 
                label_file):
    n_no_face, many_boxes, total = 0, 0, 0
    img_files = os.listdir(input_dir)
    img_files.sort()
    n_images = len(img_files)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    label_list = []
    for idx, img_file in enumerate(img_files):
        total += 1
        print('---------{}/{}---------'.format(idx, n_images))
        output_path = str(output_dir / img_file)
        if os.path.exists(output_path):
            continue
        img_path = str(input_dir / img_file)
        print('Processing {}'.format(img_path))
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        bboxes, _ = detection_md.inference(rgb_img, landmark=False)
        
        if len(bboxes) > 1:
            many_boxes_file.write(img_path + '\n')
            many_boxes += 0
        elif len(bboxes) < 1:
            unknown_file.write(img_path + '\n')
            n_no_face += 1
            continue

        face = get_face_from_box(bgr_img, bboxes[0])
        cv2.imwrite(output_path, face)
        print('Finding face for {} is done ...'.format(img_file))
        label = img_file.split('_')[0]
        label_list.append((img_file, int(label)))

    label_df = pd.DataFrame(data=label_list, columns=['image', 'label'])
    label_df.to_csv(label_file, index=False)
    print('Saved label file {}.'.format(label_file))

    print('Total images: {}.'.format(total))
    print('No face images: {}.'.format(n_no_face))
    print('Many face images: {}.'.format(many_boxes))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face alignment to \
                            specific size by landmarks detection model')
    args_parser.add_argument('-id', '--input_dir', default='test', type=str)
    args_parser.add_argument('-od', '--output_dir', default='test_aligned', 
                                type=str)
    args_parser.add_argument('-nf', '--un_face_file', default='unknown.txt', 
                                type=str)
    args_parser.add_argument('-mf', '--many_boxes_file', default='many_boxes.txt', 
                                type=str)    

    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('--label_file', default='VN_celeb.csv', type=str)
    
    args = args_parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()

    unknown_file = open(args.un_face_file, 'w')
    many_boxes_file = open(args.many_boxes_file, 'w')

    crop_face(args.input_dir, args.output_dir, detection_md, unknown_file, 
                many_boxes_file, args.label_file)
    unknown_file.close()
    many_boxes_file.close()
