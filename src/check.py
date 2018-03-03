import os
import hashlib
import io
import random
import shutil
import configparser
import pylab as plt
import tensorflow as tf

from tqdm import tqdm
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def save_img_with_box(image, target_list, img_name, group):
    for target in target_list:
        x1, y1, x2, y2, label = target
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        box = (x1, y1), (x2, y2)
        font = ImageFont.truetype(cf.get('font_path', 'simsun'), 20)
        drawObject = ImageDraw.Draw(image)  
        drawObject.rectangle(box, outline = "red")  
        drawObject.text([x1+50, y1+50], label,"red", font=font)

    save_dir = os.path.join(result_dir, group)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    image.save(os.path.join(save_dir, img_name))
    # plt.imshow(image) 
    # plt.show()
    # exit(0) 
    
missing_group = []
missing_frame = []
anno_error = []
empty_frame = []
data_dir = '../data'
img_dir = os.path.join(data_dir, 'test_samples')
result_dir = os.path.join(data_dir, 'check_result')
anno_dir = os.path.join(data_dir, 'annotations_test_samples_fc2017')
cf = configparser.ConfigParser()
cf.read('../config/traffic.config')
    
def main():
    # check whether missing frame xml 
    for group_name in tqdm(os.listdir(img_dir)):
    #for group_name in ['TSD-Signal-00255']:
        img_group_dir = os.path.join(img_dir, group_name)
        xml_group_dir = os.path.join(anno_dir, group_name)
        #print(xml_group_dir)
        if not os.path.exists(xml_group_dir):
            missing_group.append(xml_group_dir)
            continue
        for img_name in os.listdir(img_group_dir):
        #for img_name in ['TSD-Signal-00255-00018.png']:
            img_file = os.path.join(img_group_dir, img_name)
            xml_file = os.path.join(xml_group_dir, img_name.split('.')[-2] + '.xml')
            #print(img_file)
            if not os.path.exists(xml_file):
                missing_frame.append(img_name.split('.')[-2] + '.xml')
                #print(xml_file)
                continue
            # get target_list from xml
            with tf.gfile.FastGFile(xml_file, 'rb') as fid:
                xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']                 
                #print(data)
                target_list = []
                if 'object' not in data:
                    empty_frame.append(xml_file)
                    continue
                else:
                    data = data['object']
                    for i in data:
                        name = i['name']
                        ymin = i['bndbox']['ymin']
                        xmax = i['bndbox']['xmax']
                        xmin = i['bndbox']['xmin']
                        ymax = i['bndbox']['ymax']
                        target_list.append([xmin, ymin, xmax, ymax, name] )
            # plot box to image
            with tf.gfile.GFile(img_file, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            # plt.imshow(image) 
            width = image.width
            height = image.height
            #x2 = int(xmax) - int(xmin)
            #y2 = int(ymax) - int(ymin)
            #if x2 > width or y2 > height:
            #    anno_error.append(img_file)
            if target_list is not []:
                save_img_with_box(image, target_list, img_name, group_name)
    print("missing group:",missing_group)
    print("missing frame:", missing_frame)
    print("empty frame:", empty_frame)
    

if __name__ == '__main__':
    
    main()
