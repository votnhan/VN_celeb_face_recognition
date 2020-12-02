from utils import read_image, read_json, load_pickle
from PIL import Image
from align_face import alignment, center_point_dict
from data_loader import transforms_default, trans_emotion_inf
from imgaug import augmenters as iaa
from torch.nn import functional as F
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
    with torch.no_grad():
        embeddings = embedding_model(image_tensor)
    return embeddings.detach()


def find_emotion(image_tensor, emotion_model, topk=6):
    emotion_model.eval()
    with torch.no_grad():
        output, _ = emotion_model(image_tensor)
    output_np = output.detach().cpu().numpy()
    percent_np = F.softmax(output, dim=1).detach().cpu().numpy()
    sorted_output = np.argsort(output_np, axis=1)
    sorted_percent_np = np.sort(percent_np, axis=1)
    chosen_idx = sorted_output[:, -topk:]
    chosen_prob = sorted_percent_np[:, -topk:]
    return np.flip(chosen_idx, axis=1), np.flip(chosen_prob, axis=1)


def recognize_celeb(bth_alg_face_list, device, emb_model, classify_model, 
                        transforms, label2name_df, threshold):
    alg_face_list = []
    for x in bth_alg_face_list:
        alg_face_list += x
    
    tf_list = []
    for face in alg_face_list:
        tf_face = transforms(face)
        tf_list.append(tf_face)

    if len(tf_list) > 0:
        bth_names = []
        aligned_faces_tf = torch.stack(tf_list, dim=0)
        embeddings = find_embedding(aligned_faces_tf.to(device), emb_model)
        names = identify_person(embeddings, classify_model, label2name_df, 
                                    threshold)
        n_faces_4_image = [len(x) for x in bth_alg_face_list]
        
        counter = 0
        for n_face in n_faces_4_image:
            bth_names.append(names[counter: counter + n_face])
            counter += n_face
    else:
        bth_names = [[] for x in range(len(bth_alg_face_list))]
    
    return bth_names


def recognize_emotion(bth_alg_face_list, device, emt_model, transforms, 
                        map_label_func, topk=6):
    alg_face_list = []
    for x in bth_alg_face_list:
        alg_face_list += x

    emt_tf_list = []
    for face in alg_face_list:
        face_obj = Image.fromarray(face)
        tf_face = transforms(face_obj)
        emt_tf_list.append(tf_face)

    if len(emt_tf_list) > 0:
        bth_emotions, bth_probs = [], []
        aligned_faces_tf = torch.stack(emt_tf_list, dim=0)
        emotions_cls, probs = find_emotion(aligned_faces_tf.to(device), 
                                emt_model, topk)
        n_faces_4_image = [len(x) for x in bth_alg_face_list]
        counter = 0
        for n_face in n_faces_4_image:
            if n_face > 0:
                emotions = map_label_func(emotions_cls[counter: counter + n_face])
            else:
                emotions = []
            bth_emotions.append(emotions)
            bth_probs.append(probs[counter: counter + n_face])
            counter += n_face
    else:
        bth_emotions = [[] for x in range(len(bth_alg_face_list))]
        bth_probs = [[] for x in range(len(bth_alg_face_list))]
    
    return bth_emotions, bth_probs


def identify_person(embeddings, classify_model, name_df, threshold):
    classify_model.eval()
    with torch.no_grad():
        output = classify_model(embeddings)
    n_classes = output.size(1)
    if type(threshold) is float:
        threshold_dict = {}
        for i in range(n_classes):
            threshold_dict[str(i)] = threshold
    else:
        threshold_dict = threshold

    preditions = torch.argmax(output, dim=1)
    preditions_np = preditions.detach().cpu().numpy()
    probs = torch.exp(output).detach().cpu().numpy()

    chosen_prob = [probs[idx][chosen_idx] for idx, chosen_idx in enumerate(preditions_np)]
    filtered_preditions = []

    for idx, prob in enumerate(chosen_prob):
        main_thres = threshold_dict[str(preditions_np[idx])]
        if prob >= main_thres:
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


def draw_emotions(image, bboxes, emotion_tags, emotion_percent):
    for idx, box in enumerate(bboxes):
        emotion_pred = emotion_tags[idx]
        percent_pred = emotion_percent[idx]
        for i, (emotion, percent) in enumerate(zip(emotion_pred, percent_pred)):
            cv2.putText(image, '{} - {:.2f}%'.format(emotion, percent*100), 
                            (int(box[0]+5), int(box[1]) + (i+1)*16), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 
                            cv2.LINE_AA)

    return image


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
            box_ratio = box_requirements['box_ratio']
            if min_dim > box_requirements['min_dim'] and (max_dim/min_dim < box_ratio):
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
    

def sequential_detect_and_align(rgb_image, detection_md, fa_model, center_point, 
                                    target_fs, box_requirements=None, log=False):
    boxes, _,  = detection_md.inference(rgb_image, landmark=False)
    if len(boxes) > 0:
        list_face, face_idx = get_face_from_boxes(rgb_image, boxes, box_requirements)
        aligned_face_list = []
        new_face_idx = []
        for idx, face in enumerate(list_face):
            dst = align_face(face, fa_model)
            if dst is not None:
                bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                aligned_face = alignment(bgr_face, center_point, dst, target_fs[0], 
                                target_fs[1])
                aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_face_list.append(aligned_face)
                new_face_idx.append(idx)

        remain_idx = [face_idx[x] for x in new_face_idx]
        if len(remain_idx) > 0:
            chosen_boxes = [boxes[x] for x in remain_idx]
            return aligned_face_list, chosen_boxes
        else:
            if log:
                print('Bounding boxes were not qualified or could not detect landmarks !')
            return [], []
    else:
        if log:
            print('Face not found in this image !')
        return [], []
    

def parallel_detect_and_align(rgb_images, detection_md, center_point, 
                                target_fs, log=False):
    bth_boxes, _, bth_landmarks, = detection_md.inference(rgb_images, 
                                        landmark=True)
    zip_bb_ldm = zip(bth_boxes, bth_landmarks)
    bth_aligned_faces, bth_chosen_bb = [], []
    for idx, (boxes, landmarks) in enumerate(zip_bb_ldm):
        aligned_face_list = []
        chosen_boxes = []
        rgb_image = rgb_images[idx]
        if len(boxes) > 0:
            list_face, face_idx = get_face_from_boxes(rgb_image, boxes)
            if len(face_idx) > 0:
                chosen_boxes = [boxes[x] for x in face_idx]
                chosen_landmarks = [landmarks[x] for x in face_idx]
                for idx, face in enumerate(list_face):
                    moved_landmark = move_landmark_to_box(chosen_boxes[idx], 
                                        chosen_landmarks[idx])
                    bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    aligned_face = alignment(bgr_face, center_point, moved_landmark, 
                                        target_fs[0], target_fs[1])
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    aligned_face_list.append(aligned_face)
            else:
                if log:
                    print('Bounding boxes were not qualified or could not detect landmarks !')
        else:
            if log:
                print('Face not found in this image !')
        
        bth_aligned_faces.append(aligned_face_list) 
        bth_chosen_bb.append(chosen_boxes)

    return bth_aligned_faces, bth_chosen_bb

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
    args_parser.add_argument('-dv', '--device', default='cuda:0', type=str) 
    args_parser.add_argument('-id', '--input_dim_emb', default=512, type=int) 
    args_parser.add_argument('-nc', '--num_classes', default=1001, type=int)
    args_parser.add_argument('-enc', '--encoder', default='InceptionResnetV1', 
                                type=str)
    args_parser.add_argument('-det', '--detection', default='MTCNN', type=str)
    args_parser.add_argument('-eargs', '--encoder_args', 
                                default='cfg/embedding/iresnet100_enc.json', 
                                type=str)
    args_parser.add_argument('-dargs', '--detection_args', 
                                default='cfg/detection/mtcnn.json', type=str)
    args_parser.add_argument('-tg_fs', '--target_face_size', default=112, 
                                type=int)
    args_parser.add_argument('--inference_method', default='seq_fd_vs_aln', 
                                type=str)
    args_parser.add_argument('--min_dim_box', default=50, type=int)
    args_parser.add_argument('--box_ratio', default=2.0, type=float)
    args_parser.add_argument('--recog_threshold', default=0.0, type=float)
    args_parser.add_argument('--recog_emotion', action='store_true')
    args_parser.add_argument('-emt', '--emotion', default='resnet_2branch_50', 
                                type=str)
    args_parser.add_argument('-emtargs', '--emotion_args', 
                                default='cfg/emotion/resnet50_2_branch.json', 
                                type=str)
    args_parser.add_argument('-t2i', '--etag2idx_file', 
                        default='meta_data/emotion_recognition/etag2idx.pkl.keep', 
                        type=str)
    args_parser.add_argument('--topk_emotions', default=6, type=int)

    args = args_parser.parse_args()

    device = args.device
    
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

    # emotion model (if need)
    if args.recog_emotion:
        idx2etag = load_pickle(args.etag2idx_file)['idx2key']
        emt_args = read_json(args.emotion_args)
        emt_model = getattr(model_md, args.emotion)(**emt_args).to(device)

    target_fs = (args.target_face_size, args.target_face_size)
    center_point = center_point_dict[str(target_fs)]
    
    # Do face recognition process
    np_image = cv2.imread(args.image_path)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    rgb_images = [rgb_image]

    if args.inference_method == 'seq_fd_vs_aln':
        box_requirements = {
            'min_dim': args.min_dim_box,
            'box_ratio': args.box_ratio
        }
        bth_alg_faces, bth_chosen_boxes = sequential_detect_and_align(rgb_images, 
                                        detection_md, fa_model, center_point, target_fs, 
                                        box_requirements, True)
    elif args.inference_method == 'par_fd_vs_aln':
        bth_alg_faces, bth_chosen_boxes = parallel_detect_and_align(rgb_images, 
                                        detection_md, center_point, target_fs,
                                         True)
    else:
        print('Do not support {} method.'.format(args.args.inference_method))
    

    bth_names = recognize_celeb(bth_alg_faces, device, emb_model, 
                classify_model, transforms_default, label2name_df, 
                    args.recog_threshold)

    names = bth_names[0]
    chosen_boxes = bth_chosen_boxes[0]
    np_image_recog = draw_boxes_on_image(np_image, chosen_boxes, names)

    if args.recog_emotion:
        map_func = np.vectorize(lambda x: idx2etag[x])
        bth_emotions, bth_probs = recognize_emotion(bth_alg_faces, device, 
                                emt_model, trans_emotion_inf, map_func, 
                                args.topk_emotions)
        np_image_recog = draw_emotions(np_image_recog, chosen_boxes, 
                            bth_emotions[0], bth_probs[0])
    
    cv2.imwrite(args.output_path, np_image_recog)
    print('Face recognized image saved at {} ...'.format(args.output_path))
