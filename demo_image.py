from models import InceptionResnetV1, MTCNN, MLPModel
from utils import read_image
from PIL import Image
import torch
import cv2
import argparse
import pandas as pd
import numpy as np


def load_model_classify(checkpoint_path, model):
    cp = torch.load(checkpoint_path)
    print("Loading checkpoint: {} ... after training for {} epochs."\
                .format(checkpoint_path, cp['epoch']))
    model.load_state_dict(cp['state_dict'])
    return model
    

def detech_faces(image, detect_model):
    cropped_images_ts, batch_boxes = detect_model(image)
    return cropped_images_ts, batch_boxes


def find_embedding(image_tensor, embedding_model):
    embedding_model.eval()
    embeddings = embedding_model(image_tensor)
    return embeddings.detach()

def identify_person(embeddings, classify_model, name_df):
    output = classify_model(embeddings)
    preditions = torch.argmax(output, dim=0)
    preditions_np = preditions.detach().cpu().numpy()
    list_names = []
    for pred in preditions_np:
        name = list(name_df['name'][name_df['label'] == pred])
        if len(name) > 0:
            list_names.append(name[0])
        else:
            list_names.append('Unknown')

    return list_names

def draw_boxes_on_image(image, boxes, list_names, output_path):
    np_image = image
    if type(image) is Image:
        np_image = np.array(image)
    
    for box, name in zip(boxes, list_names):
        cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), 
                        (0, 255, 0), 2)
        cv2.putText(np_image, name, (box[2], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 
        0.75, (0, 255, 0), 2, cv2.LINE_AA)

    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, rgb_image)
    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Demostration face \
                    recognition on a image')

    args_parser.add_argument('-fs', '--face_size', default=160, type = int)
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
    args_parser.add_argument('-nc', '--num_classes', default=1000, type=int)


    args = args_parser.parse_args()

    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'
    
    # Prepare 3 models, database for label to name
    label2name_df = pd.read_csv(args.label2name)
    # face detection model
    mtcnn = MTCNN(args.face_size, keep_all=True, device=device)

    # face embedding model
    emb_model = InceptionResnetV1(args.pre_trained_emb, device=device)

    # classify from embedding model
    classify_model = MLPModel(args.input_dim_emb, args.num_classes)
    load_model_classify(args.classify_model, classify_model)
    classify_model = classify_model.to(device)
    
    # Do face recognition process
    pil_image = read_image(args.image_path)
    tensors_face, boxes = detech_faces(pil_image, mtcnn)
    embeddings = find_embedding(tensors_face, emb_model)
    names = identify_person(embeddings, classify_model, label2name_df)
    draw_boxes_on_image(pil_image, boxes, names, args.output_path)
    print('Face recognized image saved at {} ...'.format(args.output_path))

