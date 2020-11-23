from utils import read_image, read_json
from PIL import Image
from align_face import alignment, center_point_dict
from data_loader import transforms_default
from imgaug import augmenters as iaa
import torch
import cv2
import argparse
import pandas as pd
import numpy as np
import models as model_md 
import face_alignment


def load_model_classify(checkpoint_path, model):
    cp = torch.load(checkpoint_path)
    print("Loading checkpoint: {} ... after training for {} epochs."\
                .format(checkpoint_path, cp['epoch']))
    model.load_state_dict(cp['state_dict'])
    return model
    

def detech_faces(image, detect_model):
    detect_model.eval()
    cropped_images_ts, batch_boxes = detect_model(image)
    return cropped_images_ts, batch_boxes


def find_embedding(image_tensor, embedding_model):
    embedding_model.eval()
    embeddings = embedding_model(image_tensor)
    return embeddings.detach()

def identify_person(embeddings, classify_model, name_df, threshold=0.0):
    classify_model.eval()
    output = classify_model(embeddings)
    preditions = torch.argmax(output, dim=1)
    preditions_np = preditions.detach().cpu().numpy()
    probs = torch.exp(output).detach().cpu().numpy()

    chosen_prob = [probs[idx][chosen_idx] for idx, chosen_idx in enumerate(preditions_np)]
    filtered_preditions = []
    for idx, prob in enumerate(chosen_prob):
        if prob >= threshold:
            filtered_preditions.append(preditions_np[idx])
        else:
            filtered_preditions.append(output.size(1))
            
    list_names = []
    for pred in filtered_preditions:
        name = list(name_df['name'][name_df['label'] == pred])
        if len(name) > 0:
            list_names.append(name[0])
        else:
            list_names.append('Unknown')

    return list_names

def draw_boxes_on_image(image, boxes, list_names):
    np_image = np.array(image)
    for box, name in zip(boxes, list_names):
        cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), 
                        (0, 255, 0), 2)
        cv2.putText(np_image, name, (box[2], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 
        0.75, (0, 255, 0), 2, cv2.LINE_AA)

    return np_image

def get_face_from_boxes(image, boxes, box_requirements=None):
    list_faces = []
    face_idx = []
    ori_h, ori_w = image.shape[:2]
    for idx, box in enumerate(boxes):
        x1 = max(int(box[0]), 0)
        y1 = max(int(box[1]), 0)
        x2 = min(int(box[2] + 1), ori_w)
        y2 = min(int(box[3] + 1), ori_h)
        w, h = x2 - x1, y2 - y1
        max_dim = max(w, h)
        min_dim = min(w, h)
        
        chosen = False
        if box_requirements is None:
            chosen = True
        else:
            if min_dim > box_requirements['min_dim'] and (max_dim/min_dim < box_requirements['box_ratio']):
                chosen = True
        if chosen:
            face = image[y1:y2, x1:x2, :]
            list_faces.append(face)
            face_idx.append(idx)

    return list_faces, face_idx

def align_face(face, fa_model):
    rgb_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    landmarks = fa_model.get_landmarks(rgb_image)
    points = None
    if landmarks is None:
        range_check = list(np.linspace(0.0, 3.0, num=11))
        for sigma in range_check:
            blur_aug = iaa.GaussianBlur(sigma)
            image_aug = blur_aug.augment_image(rgb_image)
            landmarks = fa_model.get_landmarks(image_aug)
            if landmarks is not None:
                points = landmarks[0]
                break
    else:
        points = landmarks[0]

    if points is not None:
        p1 = np.mean(points[36:42,:], axis=0)
        p2 = np.mean(points[42:48,:], axis=0)
        p3 = points[33,:]
        p4 = points[48,:]
        p5 = points[54,:]
        lankmarks_cond = np.mean([p1[1],p2[1]]) < p3[1] \
                            and p3[1] < np.mean([p4[1],p5[1]]) \
                            and np.min([p4[1], p5[1]]) > \
                                np.max([p1[1], p2[1]]) \
                            and np.min([p1[1], p2[1]]) < p3[1] \
                            and p3[1] < np.max([p4[1], p5[1]])

        if lankmarks_cond:
            dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
            return dst
    
    return None

def move_landmark_to_box(box, landmark):
    top_left = box[:2]
    moved_landmark = landmark - top_left
    return moved_landmark
    

def sequential_detection_and_alignment(rgb_image, detection_md, fa_model, emb_model, classify_model, 
                                        center_point, target_fs, box_requirements, threshold):
    boxes, _,  = detection_md.inference(rgb_image, landmark=False)
    np_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if len(boxes) > 0:
        list_face, face_idx = get_face_from_boxes(np_image, boxes, box_requirements)
        aligned_face_list = []
        new_face_idx = []
        for idx, face in enumerate(list_face):
            dst = align_face(face, fa_model)
            if dst is not None:
                aligned_face = alignment(face, center_point, dst, target_fs[0], 
                                target_fs[1])
                rgb_alg_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_face_list.append(rgb_alg_face)
                new_face_idx.append(idx)

        remain_idx = [face_idx[x] for x in new_face_idx]
        
        if len(remain_idx) > 0:
            tf_list = []
            for face in aligned_face_list:
                tf_face = transforms_default(face)
                tf_list.append(tf_face)

            aligned_faces_tf = torch.stack(tf_list, dim=0)
            chosen_boxes = [boxes[x] for x in remain_idx]
            embeddings = find_embedding(aligned_faces_tf.to(device), emb_model)
            names = identify_person(embeddings, classify_model, label2name_df, threshold)
            np_image_recog = draw_boxes_on_image(np_image, chosen_boxes, names)
            cv2.imwrite(args.output_path, np_image_recog)
            print('Face recognized image saved at {} ...'.format(args.output_path))
        else:
            print('Bounding boxes were not qualified or could not detect landmarks !')
    else:
        print('Face not found in this image !')
    

def parallel_detection_and_alignment(rgb_image, detection_md, emb_model, classify_model, center_point, 
                                        target_fs, threshold):
    boxes, _, landmarks, = detection_md.inference(rgb_image, landmark=True)
    np_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if len(boxes) > 0:
        list_face, face_idx = get_face_from_boxes(np_image, boxes)
        if len(face_idx) > 0:
            aligned_face_list = []
            chosen_boxes = [boxes[x] for x in face_idx]
            chosen_landmarks = [landmarks[x] for x in face_idx]
            
            for idx, face in enumerate(list_face):
                moved_landmark = move_landmark_to_box(chosen_boxes[idx], chosen_landmarks[idx])
                aligned_face = alignment(face, center_point, moved_landmark, target_fs[0], target_fs[1])
                rgb_alg_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_face_list.append(rgb_alg_face)

            tf_list = []
            for face in aligned_face_list:
                tf_face = transforms_default(face)
                tf_list.append(tf_face)
            
            aligned_faces_tf = torch.stack(tf_list, dim=0)
            
            embeddings = find_embedding(aligned_faces_tf.to(device), emb_model)
            names = identify_person(embeddings, classify_model, label2name_df, threshold)
            np_image_recog = draw_boxes_on_image(np_image, chosen_boxes, names)
            cv2.imwrite(args.output_path, np_image_recog)
            print('Face recognized image saved at {} ...'.format(args.output_path))
        else:
            print('Bounding boxes were not qualified or could not detect landmarks !')
    else:
        print('Face not found in this image !')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face \
                    recognition on a image')

    args_parser.add_argument('-fs', '--face_size', default=160, type = int)
    args_parser.add_argument('-mfs', '--min_face_size', default=50, type=int)
    args_parser.add_argument('-i', '--image_path', default='demo.png', type=str)
    args_parser.add_argument('-o', '--output_path', default='demo_recognition.png', 
                                type=str)
    args_parser.add_argument('-m', '--classify_model', default='model_best.pth', 
                                type=str)
    args_parser.add_argument('-l2n', '--label2name', default='label2name.csv', 
                                type=str)
    args_parser.add_argument('-w', '--pre_trained_emb', default='vggface2', 
                                type=str)
    args_parser.add_argument('-dv', '--device', default='GPU', type=str) 
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-enc', '--encoder', default='InceptionResnetV1', 
                                type=str)
    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/embedding/iresnet100_enc.json', type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, type=int)
    args_parser.add_argument('--inference_method', default='seq_fd_vs_aln', type=str)
    args_parser.add_argument('--min_dim_box', default=50, type=int)
    args_parser.add_argument('--box_ratio', default=2.0, type=float)
    args_parser.add_argument('--recog_threshold', default=0.0, type=float)
    
    args = args_parser.parse_args()

    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'
    
    # Prepare 3 models, database for label to name
    label2name_df = pd.read_csv(args.label2name)
    # face detection model
    det_args = read_json(args.detection_args)
    detection_md = getattr(model_md, args.detection)(**det_args)
    detection_md.eval()
   
    # face alignment model
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                flip_input=False, device=device)

    # face embedding model
    enc_args = read_json(args.encoder_args)
    emb_model = getattr(model_md, args.encoder)(**enc_args).to(device)

    # classify from embedding model
    classify_model = model_md.MLPModel(args.input_dim_emb, args.num_classes)
    load_model_classify(args.classify_model, classify_model)
    classify_model = classify_model.to(device)

    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]
    
    # Do face recognition process
    np_image = cv2.imread(args.image_path)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    if args.inference_method == 'seq_fd_vs_aln':
        box_requirements = {
            'min_dim': args.min_dim_box,
            'box_ratio': args.box_ratio
        }
        sequential_detection_and_alignment(rgb_image, detection_md, fa_model, emb_model, classify_model, 
                                            center_point, target_fs, box_requirements, args.recog_threshold)
    elif args.inference_method == 'par_fd_vs_aln':
        parallel_detection_and_alignment(rgb_image, detection_md, emb_model, classify_model, 
                                            center_point, target_fs, args.recog_threshold)
    else:
        print('Do not support {} method.'.format(args.args.inference_method))
