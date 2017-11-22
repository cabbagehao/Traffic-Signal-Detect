#coding:utf-8
"""
使用说明:
    输入输出:
        1. 读取test_images里面的文件进行推理,输出到output/TSD-Signal-Result-Cargo目录
        2. test_images下的图片位置为TSD-Signal-**/*.png
        3. 输出xml的文件名就是上面图片的目录名TSD-Signal-**
        4. 所有路径指定在页面底部.
    功能:
        1. 得到推理结果xml,用于提交
        2. 得到推理结果图片,用于调试
        3. 得到F1等统计结果,用于调试

"""
import os
import sys
import cv2
import glob
import shutil
import configparser
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from f1_score import Score
from write_xml import build_xml
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

def draw_box_and_text(image, box, label, pred_score, font_size, font):
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    # bounding object
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)
    # label text and box
    label_text = "{}: {:.1f}%".format(label, pred_score*100)
    label_w, label_h= int(14.5*len(label_text)+4), font_size+4
    label_start = (pt2[0], pt1[1])
    # TODO: 多标签重叠处理
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
# 非极大值抑制 from: http://blog.csdn.net/gan_player/article/details/78204960
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #从大到小排列，取index
    order = scores.argsort()[::-1]  #逆序排序
    #keep为最后保留的边框
    keep = []
    while order.size > 0:
        #order[0]是当前分数最大的窗口，之前没有被过滤掉，肯定是要保留的
        i = order[0]
        keep.append(i)

        #计算窗口i与其他所以窗口的交叠部分的面积，矩阵计算
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #ind为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        #下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1
        order = order[inds + 1]
    return keep


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
    score = Score(GT_xmls_dir)   
    cf = configparser.ConfigParser()
    cf.read('../config/traffic.config')    
    font_size = 20
    font = ImageFont.truetype(cf.get('font_path', 'simsun'), font_size)

    _, label_list = get_label_dict(label_path)
    detection_graph = load_graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            ii = 0
            for group_dir in tqdm(os.listdir(test_dir)):
                img_dir = os.path.join(test_dir, group_dir)
                if not is_test_norm_data and 'TSD' not in img_dir:
                    continue
                frames_dict = {}
                for image_name in os.listdir(img_dir):
                    image_path = os.path.join(img_dir, image_name)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # TODO
                    (im_height, im_width, _) = image.shape
                    image_np_expanded = np.expand_dims(image, axis=0)
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    box_coords = to_image_coords(boxes[0], im_height, im_width)
                    
                    # 获得符合最低置信度的预测结果
                    pred_idx = []
                    for i, pred in enumerate(scores[0]):
                        if pred > pred_threshold:
                            pred_idx.append([i, pred])

                    targets = []
                    dets = []
                    for idx, pred in pred_idx:
                        clazz = int(classes[0][idx:idx+1][0])
                        label = label_list[clazz-1]   
                        box = box_coords[idx:idx+1][0]
                        targets.append([label, box, pred])
                        dets.append([box[0], box[1], box[2], box[3], pred])
                    dets = np.array(dets)
                    
                    # 非极大值抑制
                    y = py_cpu_nms(dets, 0.3)
                    a = len(targets) - len(y)
                    if a > 0: print('NMS Dorped: ', a)
                    targets = [targets[i] for i in y]
                    # 写入结果框到image
                    for label, box, pred in targets:
                        image = draw_box_and_text(image, box, label, pred, font_size, font)
                    
                    # 计算score
                    is_pass = score.update(image_name, img_dir, targets)
                    
                    # 生成结果图片 正确的和错误的分开保存
                    if is_gen_img:
                        if not is_pass: # TODO : 真值 box和type写入
                            cv2.imwrite(os.path.join(fail_case_dir, image_name), image)                        
                        else:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(img_result_dir, image_name), image)

                    # 填充xml数据
                    frame_id = image_name.split('-')[-1].replace('.png', '')
                    frames_dict[frame_id] = targets


                # 生成xml文件 eg: TSD-Signal-00120-Result.xml
                if is_gen_xml:
                    frames_dict['frame_number'] = len(os.listdir(img_dir))
                    result_xml_name = group_dir + '-Result.xml'
                    result_xml_path = os.path.join(xml_result_dir, result_xml_name)
                    build_xml(result_xml_path, frames_dict)
    return score

data_dir = '../data'
output_dir = '../output'
test_dir = os.path.join(data_dir, 'test_samples')
label_path = os.path.join(data_dir,'traffic.label')
img_result_dir = os.path.join(output_dir, 'test_result')
fail_case_dir = os.path.join(img_result_dir, 'FailedCase')
xml_result_dir = os.path.join(output_dir, 'TSD-Signal-Result-Cargo')

GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT') 
pb_path = './model_bak/model_pb/rcnn_resnet101_27941/frozen_inference_graph.pb'

pred_threshold = 0.5
# all_test_images=glob.glob(os.path.join(test_dir, '*/*.png'))
# all_test_images.sort()

# 是否生成带框的结果图片
is_gen_img = True
# 是否生成提交结果的xml文件 
is_gen_xml = True
# 是否测试Norm数据集
is_test_norm_data = False

if __name__ == '__main__':

    if os.path.exists(img_result_dir):
        shutil.rmtree(img_result_dir)
    os.makedirs(img_result_dir)    

    if os.path.exists(xml_result_dir):
        shutil.rmtree(xml_result_dir)
    os.makedirs(xml_result_dir)  
    os.makedirs(fail_case_dir)  
    

    score = main()
    f1, precision, recall = score.get_f1_score() 
    acc, total = score.get_combine_accuracy()
    box_acc = score.get_box_accuracy()

    # print(score.TP, score.FP, score.box_TP, score.box_FP)
    print("F1 score: %.2f, precision: %.2f, recall: %.2f" %(f1, precision, recall))
    print('Total: ', total, ", Accuracy: ", format(acc,'0.1%'))
    print("box_acc: ", format(box_acc,'0.1%'))
    print("box match but type not match count: ", score.type_missed_count)



