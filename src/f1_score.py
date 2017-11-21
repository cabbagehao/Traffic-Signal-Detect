import os
import numpy as np
from lxml import etree

class Score():
    def __init__(self, gt_dir):
        self.gt_dir = gt_dir
        self.xml_file_dict = {}
        self.prefix = 'TSD-Signal-'
        self.suffix = '-GT.xml'
        self.TP = 0     # 正检
        self.FP = 0     # 误检
        self.FN = 0     # 漏检
        self.ALL = 0    # 总数
        self.box_TP = 0 # box正检
        self.box_FP = 0 # box误检
        self.type_missed_count = 0

    def _get_xml_object(self, image_name):
        group, frame = image_name.replace(self.prefix, '').replace('.png', '').split('-')
        xml_name = self.prefix + group + self.suffix   
        if xml_name not in self.xml_file_dict.keys():
            xml_path = os.path.join(self.gt_dir, xml_name)
            tree = etree.parse(xml_path)
            data = tree.getroot()
            self.xml_file_dict[xml_name] = data

        return self.xml_file_dict[xml_name], frame

    def _pross_not_TSD_file(self, image_name, img_dir, targets):
        gt_label = img_dir.split('/')[-1]
        label_txt = os.path.join(img_dir, 'labels.txt')
        label_path = os.path.join('../data/TrafficNorm/', gt_label, 'labels.txt')
        self.ALL += 1
        for label, box, pred in targets:
            x, y =  0.5*(box[0] + box[2]),  0.5*(box[1] + box[3])
            with open(label_path) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if image_name in lines[i]:
                        line = lines[i].split()
                        img_name, x1, y1, w, h = line
                        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            self.box_TP += 1
                            if label == gt_label:
                                self.TP += 1
                            else:
                                self.FP += 1
                        else:
                            self.FP += 1
                    # 所有标注只有一个,如果匹配了直接return
                    return

        

    def update(self, image_name, img_dir, targets):
        # 新增数据集单独处理
        if 'TSD' not in image_name:
            self._pross_not_TSD_file(image_name, img_dir, targets)
            return

        xml, frame = self._get_xml_object(image_name)
        frame_name = 'Frame' + frame
        targets_num = int(xml.xpath(frame_name + 'TargetNumber')[0].text)
        self.ALL += targets_num
        last_FP = self.FP
        gt_boxes = []
        for i in range(targets_num):
            number = ('00000'+str(i))[-5:]
            target = frame_name + 'Target' + number 
            label = xml.xpath(target + '/Type')[0].text
            position = xml.xpath(target + '/Position')[0].text.split()
            position = [int(i) for i in position]
            gt_boxes.append([label, position])
        # 判断当前box中心是否在图片内所有GT-box任一个
        # 一个GT-box肯定只包含一个元素,因此判断成功后会删除该box
        for label, box, pred in targets:
            x, y =  0.5*(box[0] + box[2]),  0.5*(box[1] + box[3])

            box_correct = False
            for i in range(len(gt_boxes)):
                gt_label, gt_box = gt_boxes[i]
                x1, x2 = gt_box[0], gt_box[0]+gt_box[2]
                y1, y2 = gt_box[1], gt_box[1]+gt_box[3]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    box_correct = True
                    box_gt_label = gt_label
                    del(gt_boxes[i])
                    break

            if box_correct:
                box_gt_label = box_gt_label.strip('\"')
                self.box_TP += 1
                if label == box_gt_label:
                    self.TP += 1
                else:
                    self.FP += 1
                    print('pos match but type not match:',label, box_gt_label, image_name)
                    self.type_missed_count += 1
            else:
                # 长边尺寸小于 16 像素的交通标志、长边尺寸小于 9 像素的交通信号灯不作检测要求
                # 如果物体较小，GT里就没有。 就不算误检。
                # if (type == 'sign' and (width < 16 or height < 16)) or (type == 'light' and (width < 9 or height < 9)):
                    # continue
                self.box_FP += 1
                self.FP += 1

        self.FN += len(gt_boxes)
        # TODO 处理每个label的准确率
        return last_FP == self.FP


    def get_box_accuracy(self):
        return self.box_TP / self.ALL

    def get_combine_accuracy(self):
        return self.TP / self.ALL, self.ALL

    def get_f1_score(self):
        TP, FP, FN = self.TP, self.FP, self.FN
        if TP == 0:
            return 0
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 *  precision * recall / (precision + recall)

        return f1_score, precision, recall


if __name__ == '__main__':
    data_dir = '../../data'
    GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')        
    a = Score(GT_xmls_dir)