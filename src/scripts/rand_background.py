"""
    将图片的背景以高概率随机更换为其他图片
    解决视频帧数据集图片非常相似问题. 以期改善测试结果.
"""
import os
import cv2
import copy
import shutil
import random
import numpy as np
from lxml import etree
from tqdm import tqdm

def change_img_light(img):
    # img = cv2.imread(img_path)
    dst = img.copy()
    a = random.random() + 0.5
    # b = np.random.normal(-5, 5)
    b = 0
    cv2.convertScaleAbs(img, dst, alpha=a, beta=b)
    return dst

def get_image_box_and_background(img_path):
    '''返回图片背景和标注框.  背景的原标注框部门使用图片随机区块填充 '''
    ROI_list = []
    image = cv2.imread(img_path)
    group = img_path.split('-')[-2]
    frame = img_path.split('-')[-1].replace('.png', '')

    xml_path = os.path.join(gt_dir, 'TSD-Signal-' + group + '-GT.xml')
    tree = etree.parse(xml_path)
    data = tree.getroot()
    target_num = int(data.xpath('Frame' + frame + 'TargetNumber')[0].text)
    for i in range(target_num):
        str_i = ('00000' + str(i))[-5:]
        target_node_name = 'Frame' + frame + 'Target' + str_i
        target_type = data.xpath(target_node_name + '/Type')[0].text
        x1, y1, w, h = data.xpath(target_node_name + '/Position')[0].text.strip().split()
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        # 处理超出边界的异常标注
        if x1+w > image.shape[1]: 
            w = image.shape[1] - x1
            print('box out of image x', img_path)
        if y1+h > image.shape[0]: 
            h = image.shape[0] - y1
            print('box out of image y', img_path)
        x2, y2 = x1+w, y1+h

        # 得到标注框的ROI
        ROI = copy.copy(image[y1:y2, x1:x2])
        # 随机选取一块区域填充原图的ROI区域
        rand_x = random.randint(0, image.shape[1]-w)
        rand_y = random.randint(0, image.shape[0]-h)

        try:
            image[y1:y2, x1:x2] = image[rand_y:rand_y+h, rand_x:rand_x+w]
        except ValueError:
            print("ValueError at roi img.", x1, y1, x2, y2, image.shape, rand_x, rand_y, img_path)
            exit(1)
        # cv2.imwrite('test.png', ROI)
        # cv2.imshow(image)
        ROI_list.append([ROI, target_type, x1, y1, x2, y2])

    
    # b,g,r = cv2.split(img)
    return image, ROI_list


    # image = cv2.open(img_path)

def main():
    # 得到所有所有图片的路径
    # 遍历数组 以高概率替换背景
    img_list = []
    for group in os.listdir(test_img_dir):
        group_path = os.path.join(test_img_dir, group)
        imgs = os.listdir(group_path)
        for i in range(len(imgs)):
            imgs[i] = os.path.join(group_path, imgs[i])
        # img_list.append(imgs)
        img_list.extend(imgs)

    rand_threshold = 0.8
    for img_path in tqdm(img_list):
        rand = random.random()
        group, img_name = img_path.split('/')[-2], img_path.split('/')[-1]
        group_path = os.path.join(new_img_dir, group)
        if not os.path.exists(group_path):
            os.makedirs(group_path)
        new_img_path = os.path.join(group_path, img_name)

        if rand < rand_threshold:
            # replace_img_path = random.sample(img_list, 1)[0]
            # image_background_old, ROI_old = get_image_box_and_background(img_path)
            # image_background_new, ROI_new = get_image_box_and_background(replace_img_path)
            # for roi in ROI_old:
            #     roi_img, _, x1, y1, x2, y2 = roi
            #     image_background_new[y1:y2, x1:x2] = roi_img
            # cv2.imwrite(new_img_path, image_background_new)
            
            image = cv2.imread(img_path)
            image = change_img_light(image)
            cv2.imwrite(new_img_path, image)
        else:
            shutil.copy(img_path, new_img_path)




random.seed(2017*2)
np.random.seed(123*2)
data_dir = '../../data'
gt_dir = os.path.join(data_dir, 'TSD-Signal-GT')
test_img_dir = os.path.join(data_dir, 'TSD-Signal')
# new_img_dir = os.path.join(data_dir, 'TSD-Signal_rand_bg')
new_img_dir = os.path.join(data_dir, 'TSD-Signal_rand_color')


if __name__ == '__main__':
    if os.path.exists(new_img_dir):
        shutil.rmtree(new_img_dir)
    os.makedirs(new_img_dir)

    main()
