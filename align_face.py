import face_alignment
import numpy as np
import os
import cv2
import argparse
from imgaug import augmenters as iaa
from skimage import transform as trans
from shutil import copyfile
from pathlib import Path
from skimage import io

center_point_dict = {
    '(96, 112)': np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], 
            dtype=np.float32),
    '(112, 112)': np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], 
            dtype=np.float32),
    '(150, 150)': np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], 
            dtype=np.float32),
    '(160, 160)': np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], 
            dtype=np.float32),
    '(224, 224)': np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], 
            dtype=np.float32)
}


def alignment(cv_img, src, dst, dst_w, dst_h):
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    affine_mt = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img, affine_mt, (dst_w, dst_h), 
                    borderValue = 0.0)
    return face_img

def face_image_from_landmarks(center_points, dst, img_rgb, output_dir, 
                                img_file, aligned_size):
    cv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    face_resized = alignment(cv_img, center_points, dst, 
                        aligned_size[0], aligned_size[1])

    output_path = str(output_dir / img_file)
    cv2.imwrite(output_path, face_resized)
    print('Finding face for {} is done ...'.format(img_file))


def align_face(input_dir, output_dir, aligned_size, fa_model, center_points, 
                unknown_file):
    n_no_face = 0
    total = 0
    img_files = os.listdir(input_dir)
    img_files.sort()
    n_images = len(img_files)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for idx, img_file in enumerate(img_files):
        img_path = str(input_dir / img_file)
        output_path = str(output_dir / img_file)
        print('---------{}/{}---------'.format(idx, n_images))
        if os.path.exists(output_path):
            continue
        print('Processing {}'.format(img_path))
        bgr_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        landmarks = fa_model.get_landmarks(rgb_image)
        have_face = False
        if landmarks is None:
            print('Step 1: unknown {}'.format(img_path))
            range_check = list(np.linspace(0.0, 3.0, num=11))
            for sigma in range_check:
                blur_aug = iaa.GaussianBlur(sigma)
                image_aug = blur_aug.augment_image(rgb_image)
                landmarks = fa_model.get_landmarks(image_aug)
                
                if landmarks is not None:
                    print('sigma {} help finding face'.format(sigma))
                    points = landmarks[0]
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
                        face_image_from_landmarks(center_points, dst, rgb_image, 
                                                    output_dir, img_file, 
                                                    aligned_size)
                        have_face = True
                        break
        else:
            points = landmarks[0]
            p1 = np.mean(points[36:42,:], axis=0)
            p2 = np.mean(points[42:48,:], axis=0)
            p3 = points[33,:]
            p4 = points[48,:]
            p5 = points[54,:]
            dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
            face_image_from_landmarks(center_points, dst, rgb_image, output_dir, 
                                        img_file, aligned_size)
            have_face = True

        if not have_face:
            n_no_face += 1
            print('{} has no face'.format(img_path))
            unknown_file.write(img_path + '\n')
            face_resized = cv2.resize(bgr_image, aligned_size, 
                                        interpolation=cv2.INTER_CUBIC)
            output_path = str(output_dir / img_file)
            cv2.imwrite(output_path, face_resized)

        total += 1
    print('No face: {}'.format(n_no_face))
    print('Total images: {}'.format(total))

 
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Face alignment to \
                            specific size by landmarks detection model')
    args_parser.add_argument('-id', '--input_dir', default='test', type=str)
    args_parser.add_argument('-od', '--output_dir', default='test_aligned', 
                                type=str)
    args_parser.add_argument('-as', '--aligned_size', nargs='+', type=int)
    args_parser.add_argument('-nf', '--un_face_file', default='unknown.txt', 
                                type=str)
    args_parser.add_argument('-dv', '--device', default='cuda:0', type=str)
    args = args_parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                flip_input=False, device=args.device)
    aligned_size = tuple(args.aligned_size)
    center_point = center_point_dict[str(aligned_size)]
    unknown_file = open(args.un_face_file, 'w')
    align_face(args.input_dir, args.output_dir, aligned_size, fa_model, 
                    center_point, unknown_file)
    unknown_file.close()
