import cv2
import os
import argparse
import glob
import time
import pandas as pd
from pathlib import Path
from utils import read_image
from models import MTCNN, InceptionResnetV1, MLPModel
from demo_image import detech_faces, find_embedding, identify_person, \
                        draw_boxes_on_image, load_model_classify


def recognize_faces_image(np_image, detect_model, embedding_model, 
                            classify_model, device, label2name_df):
    tensors_face, boxes = detech_faces(np_image, detect_model)
    if tensors_face is not None:
        embeddings = find_embedding(tensors_face.to(device), embedding_model)
        names = identify_person(embeddings, classify_model, label2name_df)
        rgb_image = draw_boxes_on_image(np_image, boxes, names)
        return rgb_image, names
    else:
        return np_image, None

def export_video_face_recognition(output_frame_dir, fps, output_path):
    container_path = Path(output_frame_dir)
    image_files = glob.glob(str(container_path / '*'))
    n_images = len(image_files)
    size = None
    first_img_path = container_path / 'frame_{}.png'.format(1)
    first_img = cv2.imread(str(first_img_path))
    height, width, channels = first_img.shape
    size = (width, height)
    out_writer = cv2.VideoWriter(output_path, 
                    cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(1, n_images + 1):
        img_path = container_path / 'frame_{}.png'.format(i)
        img = cv2.imread(str(img_path))
        out_writer.write(img)

    out_writer.release()
    print('Save exported video in {} ...'.format(output_path))
        

def main(args, detect_model, embedding_model, classify_model, device, 
            label2name_df):
    
    if not os.path.exists(args.output_frame):
        os.makedirs(args.output_frame)

    cap = cv2.VideoCapture(args.video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = []
    start_time = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        time_in_video = count / fps
        print('Processing for frame: {}, time: {:.2f} s'.format(count, 
                    time_in_video))
        recognized_img, names = recognize_faces_image(frame, detect_model, 
                                    embedding_model, classify_model, device,
                                    label2name_df)

        if args.save_frame_recognized != '':
            image_name = 'frame_{}.png'.format(count)
            image_path = os.path.join(args.output_frame, image_name)
            cv2.imwrite(image_path, recognized_img)
        
        if names is None:
            names = []
        
        tracker.append((time_in_video, str(names)))
    
    end_time = time.time()
    processed_time = end_time - start_time
    fps_process = int(cv2.CAP_PROP_FRAME_COUNT / processed_time)
    tracked_df = pd.DataFrame(data=tracker, columns=['Time', 'Names'])
    tracked_df.to_csv(args.output_tracker, index=False)
    cap.release()
    print('Saved tracker file in {} ...'.format(args.output_tracker))
    print('FPS for recognition face: {}'.format(fps_process))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face \
                    recognition on a video')

    args_parser.add_argument('-fs', '--face_size', default=160, type = int)
    args_parser.add_argument('-mfs', '--min_face_size', default=50, type=int)
    args_parser.add_argument('-i', '--video_path', default='video.mp4', type=str)
    args_parser.add_argument('-o', '--output_frame', default='output_frame', 
                                type=str)
    args_parser.add_argument('-ot', '--output_tracker', default='tracker.csv', 
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
    args_parser.add_argument('-ov', '--output_video', default='', type=str)
    args_parser.add_argument('-fps', '--fps_video', default=25.0, type=float)
    args_parser.add_argument('-sfr', '--save_frame_recognized', default='', 
                                    type=str)

    args = args_parser.parse_args()

    device = 'cpu'
    if args.device == 'GPU':
        device = 'cuda:0'

    # Prepare 3 models, database for label to name
    label2name_df = pd.read_csv(args.label2name)
    # face detection model
    mtcnn = MTCNN(args.face_size, keep_all=True, device=device, 
                    min_face_size=args.min_face_size)

    # face embedding model
    emb_model = InceptionResnetV1(args.pre_trained_emb, device=device)

    # classify from embedding model
    classify_model = MLPModel(args.input_dim_emb, args.num_classes)
    load_model_classify(args.classify_model, classify_model)
    classify_model = classify_model.to(device)
    main(args, mtcnn, emb_model, classify_model, device, label2name_df)
    if args.output_video != '':
        export_video_face_recognition(args.output_frame, args.fps_video, 
            args.output_video)
