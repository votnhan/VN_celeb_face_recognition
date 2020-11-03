import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from utils import read_json


def visualize_false_prediction(result_path, description_dict, img_container, 
                                    output_container):
    if not os.path.exists(output_container):
        os.makedirs(output_container)

    df_result = pd.read_csv(result_path)
    miss_match = df_result['Target'] != df_result['Prediction']
    rows_miss_match = df_result.loc[miss_match]
    for idx, (index, row) in enumerate(rows_miss_match.iterrows()):
        image_input = Image.open(row[0])
        image_target = find_class_anchor(description_dict, row[1], img_container)
        image_pred = find_class_anchor(description_dict, row[2], img_container)
        create_image_to_compare(image_input, image_target, image_pred, 
                                    row[1], row[2], row[3], row[0], idx, 
                                    output_container)
        print('Save visualization for sample at: {}, {}'.format(idx, row[0]))
        

def find_class_anchor(description_dict, class_idx, container):
    images4class = description_dict[str(class_idx)]
    anchor = images4class[0]
    anchor_path = os.path.join(container, anchor)
    image = Image.open(anchor_path)
    return image


def create_image_to_compare(image_input, image_target, image_pred, target_cls, 
                                pred_class, prob, input_path, idx, 
                                output_container):
    fig, axes = plt.subplots(1, 3)
    image_name = input_path.split('/')[-1]
    axes[0].imshow(image_input)
    axes[1].imshow(image_target)
    axes[2].imshow(image_pred)
    first_img_title = 'Input image: {}; '.format(image_name)
    second_img_title = 'Target class index: {}; '.format(target_cls)
    third_img_title = 'Prediction class: {}, probability: {:.2f};'.format\
                        (pred_class, prob)
    title = first_img_title + second_img_title + third_img_title
    output_path = os.path.join(output_container, 
                    'vlz_for_sample_{}.png'.format(idx))
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Visualization for false \
                        prediction.')
    argparser.add_argument('-rp', '--result_path', default='result.csv', 
                            type=str)
    argparser.add_argument('-dcrf', '--description_file', 
                            default='vn_celeb.json', type=str)
    
    argparser.add_argument('-icnt', '--image_container', default='train', 
                            type=str)
    
    argparser.add_argument('-ocnt', '--output_container', default='output_vlz', 
                            type=str)
    
    args = argparser.parse_args()
    desc_dict = read_json(args.description_file)
    visualize_false_prediction(args.result_path, desc_dict, 
                                args.image_container, args.output_container)

