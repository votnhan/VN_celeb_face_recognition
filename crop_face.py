import argparse
import os
import glob
import models as model_md
import cv2
import pandas as pd
from utils import read_json
from pathlib import Path
from demo_image import move_landmark_to_box, alignment
from align_face import center_point_dict


def find_area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w*h


def find_max_area_box(bboxes):
    max_idx = 0
    max_area = find_area(bboxes[0])
    for idx, box in enumerate(bboxes[1: ]):
        area = find_area(box)
        if area > max_area:
            max_area = area
            max_idx = idx + 1

    return max_idx


def get_face_from_box(bgr_img, box):
    ori_h, ori_w = bgr_img.shape[:2]
    x1 = max(int(box[0]), 0)
    y1 = max(int(box[1]), 0)
    x2 = min(int(box[2] + 1), ori_w)
    y2 = min(int(box[3] + 1), ori_h)
    face = bgr_img[y1:y2, x1:x2, :]
    return face


def crop_and_align_face(model, rgb_image, center_point, target_fs):
    bth_bboxes, _, bth_landmarks = detection_md.inference([rgb_image], landmark=True)
    bboxes = bth_bboxes[0]
    landmarks = bth_landmarks[0]
    aligned_face = None
    if len(bboxes) > 0:
        max_idx = find_max_area_box(bboxes)
        moved_landmark = move_landmark_to_box(bboxes[max_idx], landmarks[max_idx])
        face = get_face_from_box(rgb_image, bboxes[max_idx])
        bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        aligned_face = alignment(bgr_face, center_point, moved_landmark, 
                            target_fs[0], target_fs[1])
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    return aligned_face, len(bboxes)


def crop_face(model, rgb_image):
    bth_bboxes, _ = detection_md.inference([rgb_image], landmark=False)
    bboxes = bth_bboxes[0]
    face = None
    if len(bboxes) > 0:
        max_idx = find_max_area_box(bboxes)
        face = get_face_from_box(rgb_image, bboxes[max_idx])
    
    return face, len(bboxes)


def crop_face_dataset(input_dir, output_dir, detection_md, unknown_file, 
                many_boxes_file, label_file, align_params=None):
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

        if align_params is None:
            face, n_faces = crop_face(detection_md, rgb_img)
        else:
            face, n_faces = crop_and_align_face(detection_md, rgb_img, 
                                align_params['center_point'], align_params['target_fs'])

        if n_faces >  1:
            many_boxes_file.write(img_path + '\n')
            many_boxes += 1
            continue
        elif n_faces < 1:
            unknown_file.write(img_path + '\n')
            n_no_face += 1
            continue

        bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_face)
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
    args_parser.add_argument('--align', action='store_true')
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    
    args = args_parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()

    # tracker file 
    unknown_file = open(args.un_face_file, 'w')
    many_boxes_file = open(args.many_boxes_file, 'w')

    # face alignment params
    align_params = None
    if args.align:
        target_fs = (args.target_face_size, args.target_face_size)
        center_point = center_point_dict[str(target_fs)]
        align_params = {'center_point': center_point, 'target_fs': target_fs}

    crop_face_dataset(args.input_dir, args.output_dir, detection_md, unknown_file, 
                many_boxes_file, args.label_file, align_params)

    unknown_file.close()
    many_boxes_file.close()
