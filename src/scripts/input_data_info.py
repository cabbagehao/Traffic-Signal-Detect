import os
import tensorflow as tf
import matplotlib.pyplot as plt

from lxml import etree

def get_info():
    picture_count = 0
    # label计数
    label_dict ={}
    # 每个group包含的label数
    label_per_group = {}
    # 每个label所在的group个数
    group_per_label = {}
    # 每一帧包含的target个数
    obj_per_frame = [0 for i in range(10)]

    # 处理数据集1:
    for xml_path in os.listdir(GT_xmls_dir):
        group_name = xml_path.split('-')[2]
        label_per_group[group_name] = []
        xml_path = os.path.join(GT_xmls_dir, xml_path)
        tree = etree.parse(xml_path)
        # layer 1
        data = tree.getroot()
        # layer 2
        for node in data:
            if node.tag == 'FrameNumber':
                frame_count = int(node.text)
                picture_count += frame_count

            if 'TargetNumber' in node.tag:
                obj_num = int(node.text)
                obj_per_frame[obj_num] += 1

            # layer 3
            childs = node.getchildren()
            for child in childs:
                if child.tag == 'Type':
                    label = child.text
                    if label not in label_dict:
                        label_dict[label] = 0
                    label_dict[label] += 1

                    if label not in label_per_group[group_name]:
                        label_per_group[group_name].append(label)
                    
    # 处理数据集2
    if is_vis_norm_data:
        for label_dir in os.listdir(TrafficNorm):
            label = label_dir
            img_dir = os.path.join(TrafficNorm, label_dir)
            num = len(os.listdir(img_dir))

            if not label in label_dict:
                label_dict[label] = 0
            label_dict[label] += num
            picture_count += num

    # 计算每个label分布的组数
    for group in label_per_group:
        labels = label_per_group[group]
        for label in labels:
            if label not in group_per_label:
                group_per_label[label] = []
            if group not in group_per_label[label]:
                group_per_label[label].append(group)

    return picture_count,  obj_per_frame, label_dict, label_per_group, group_per_label

def visualize_count(label_dict):
    font= '/usr/share/fonts/winFonts/simsun.ttc'
    x = []
    y = []
    for key in label_dict:
        x.append(key)
        y.append(label_dict[key])

    rects = plt.bar(range(len(y)), y, tick_label=x, align="center")

    for rect in rects:
        height = rect.get_height()
        if height > 100:
            plt.text(rect.get_x() + rect.get_width()/2., 1.03*height, "%s" % float(height))
    # plt.xticks(size='small', rotation=70)

   
    # plt.text(2, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    # plt.bar(left = (0,1),height = (1,0.5),width = 0.35, )
    # xticks1 = [i for i in range(len(x))]
    # plt.xticks(x,xticks1,size='small',rotation=30)
    plt.show()

def visualize_group(group_dict, label=None):
    font= '/usr/share/fonts/winFonts/simsun.ttc'
    x = []
    y = []
    for key in group_dict:
        x.append(key)
        y.append(len(group_dict[key]))

    rects = plt.bar(range(len(y)), y, tick_label=x, align="center")
    plt.title(label)
    plt.show()


def main():
    picture_count,  obj_per_frame, label_dict, label_per_group, group_per_label = get_info()

    print("picture count: ", picture_count,'\n')
    print("label count: ", label_dict, '\n')
    print('obj_per_frame: ', obj_per_frame)

    # print(label_per_group)
    # print(group_per_label)
    # visualize_count(label_dict)
    visualize_group(label_per_group, label='labels_per_group')
    visualize_group(group_per_label, label='groups_per_label')

data_dir = '../../data'
GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')
TrafficNorm = os.path.join(data_dir, 'TrafficNorm')
is_vis_norm_data = False

if __name__ == '__main__':
    main()
