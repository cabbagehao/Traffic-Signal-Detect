#coding:utf-8
import os
import sys
import glob
import cv2
import configparser
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image

def get_label_dict(label_path):
    label_map_dict = {}
    label_list = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.split():
                continue
            line = line.strip()
            number, name = line.split(' ', 1)
            label_map_dict[name] = int(number)
            label_list.append(name)
            assert len(label_list) == int(number), str(label_list) + ' ' + number

    # assert len(label_map_dict) == 77
    return label_map_dict, label_list

def to_image_coords(boxes, height, width):
    box_coords = np.zeros_like(boxes)
    box_coords[:, 1] = boxes[:, 0] * height
    box_coords[:, 0] = boxes[:, 1] * width
    box_coords[:, 3] = boxes[:, 2] * height
    box_coords[:, 2] = boxes[:, 3] * width
    
    return box_coords

def process_image(image, idx, classes, box_coords, label_list, max_score):
    font_size = 20
    font = ImageFont.truetype(cf.get('font_path', 'simsun'), font_size)

    clazz = int(classes[0][idx:idx+1][0])
    box = box_coords[idx:idx+1][0]
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    # bounding object
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
    # label text and box
    label_text = "{}: {:.1f}%".format(label_list[clazz-1], max_score*100)
    label_w, label_h= int(14.5*len(label_text)+4), font_size+4
    label_start = (pt2[0], pt1[1])
    
    # 如果x标签超出了最右边：
    #   如果y标签不会超出最上边，移动到上面显示
    #   如果y超出了最上面，移动到下面显示
    if label_start[0] + label_w > image.shape[1]:
        if label_start[1] - label_h > 0:
            shift = (- label_w, - label_h - 1)
        else:
            shift = (- label_w, pt2[1]-pt1[1] + 2)
        label_start = (label_start[0] + shift[0], label_start[1] + shift[1])

    label_end = (label_start[0] + label_w, label_start[1] + label_h)
    cv2.rectangle(image, label_start, label_end, (0, 255, 0), -1)
    
    # 转换为PIL写入中文后再转为cv2
    cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    drawObject = ImageDraw.Draw(pil_im) 
    drawObject.text((label_start[0] + 2, label_start[1] + 2), label_text,"red", font=font)
    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)    

    return image

def load_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def main():

    _, label_list = get_label_dict(label_path)
    detection_graph = load_graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            for image_path in tqdm(all_test_images):
                image_name = os.path.basename(image_path)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                (im_height, im_width, _) = image.shape
                image_np_expanded = np.expand_dims(image, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                box_coords = to_image_coords(boxes[0], im_height, im_width)
                
                max_score = 0
                idx = None
                for i, score in enumerate(scores[0]):
                    if score > min_score and score > max_score:
                        max_score = score
                        idx = i
                if idx is not None:
                    image = process_image(image, idx, classes, box_coords, label_list, max_score)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, image_name), image)



data_dir = '../data'
test_dir = os.path.join(data_dir, 'test_samples')
label_path = os.path.join(data_dir,'traffic.label')
output_dir = os.path.join(data_dir, '../output/test_result')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pb_path = './model_pb/frozen_inference_graph.pb'

min_score = 0
all_test_images=glob.glob(os.path.join(test_dir, '*/*.png'))
all_test_images.sort()

cf = configparser.ConfigParser()
cf.read('../config/traffic.config')

if __name__ == '__main__':
    main()