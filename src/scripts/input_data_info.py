'''
    1. 统计训练集中每个label的个数
    2. 统计每一组里label的个数
    3. 统计每个label分布在哪些组

    1. 检查数据集完整性: 是否覆盖所有label类型
    2. 检查数据集正确性: 是否存在label文件中没有的类型
    3. 是否每个图片都有GT文件,是否每个GT文件都有图片 TODO
'''
import os
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

from lxml import etree

def is_all_label_covered(label_dict, label_map_dict):
    missed_label = []
    # print(label_dict.keys(), label_map_dict.keys())
    for label in label_map_dict.keys():
        if label not in label_dict.keys():
            missed_label.append(label)
    return missed_label

def get_label_dict(label_path):
    label_map_dict = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.split():
                continue
            line = line.strip()
            number, name = line.split(' ', 1)
            name = '\"' + name + '\"'
            label_map_dict[name] = int(number)

    # assert len(label_map_dict) == 77
    return label_map_dict

def visualize_count(label_dict):
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
    plt.xticks(size='small', rotation=70)
   
    # plt.text(2, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    # plt.bar(left = (0,1),height = (1,0.5),width = 0.35, )
    # xticks1 = [i for i in range(len(x))]
    # plt.xticks(x,xticks1,size='small',rotation=30)
    plt.show()

def visualize_group(group_dict, label=None):
    if group_dict is None:
        print('group_dict is None.')
        return
    x = []
    y = []
    for key in group_dict:
        x.append(key)
        y.append(len(group_dict[key]))

    rects = plt.bar(range(len(y)), y, tick_label=x, align="center")
    plt.title(label)
    plt.show()

def loop_norm_dataset(picture_count, label_dict, label_map_dict, label_not_match):
    for label_dir in os.listdir(norm_data_path):
        label = label_dir
        img_dir = os.path.join(norm_data_path, label_dir)
        num = len(os.listdir(img_dir))
        # 数据集的异常label type
        if label not in label_map_dict.keys():
            if label not in label_not_match:
                label_not_match[label] = []
            label_not_match[label].append(label_dir)

        if not label in label_dict:
            label_dict[label] = 0
        label_dict[label] += num
        picture_count += num    

    return label_dict, picture_count

def loop_traffic_dataset(picture_count, label_dict, label_per_group, obj_per_frame, label_map_dict, label_not_match):
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
                    # 数据集的异常label type
                    if label not in label_map_dict.keys():
                        if label not in label_not_match:
                            label_not_match[label] = []
                        label_not_match[label].append(group_name)    

                    if label not in label_dict:
                        label_dict[label] = 0
                    label_dict[label] += 1

                    if label not in label_per_group[group_name]:
                        label_per_group[group_name].append(label)
    return picture_count

def get_info(label_map_dict):
    picture_count = 0
    # 每个label所在的图片路径
    label_dict ={}
    # 每个group包含的label数
    label_per_group = {}
    # 每个label所在的group个数
    group_per_label = {}
    # 每一帧包含的target个数
    obj_per_frame = [0 for i in range(10)]
    # 数据集中错误的label type
    label_not_match = {}

    # 处理数据集1:
    if is_deal_traffic_data:
        picture_count = loop_traffic_dataset(picture_count, label_dict, label_per_group, 
                                                obj_per_frame, label_map_dict, label_not_match)
    
    # 处理武大数据集
    if is_deal_norm_data:
        picture_count = loop_norm_dataset(picture_count, label_dict, label_map_dict, label_not_match)

    # 计算每个label分布的组数
    for group in label_per_group:
        labels = label_per_group[group]
        for label in labels:
            if label not in group_per_label:
                group_per_label[label] = []
            if group not in group_per_label[label]:
                group_per_label[label].append(group)

    return picture_count,  obj_per_frame, label_dict, label_per_group, group_per_label, label_not_match


def main():
    label_map_dict = get_label_dict(label_path)
    picture_count,  obj_per_frame, label_dict, label_per_group, group_per_label, label_not_match = get_info(label_map_dict)
    missed_label = is_all_label_covered(label_dict, label_map_dict)
    # 异常信息
    if missed_label:
        print('数据集缺失的label: ', missed_label)
    if label_not_match:
        print('数据集错误的label: ', label_not_match)

    # 正常信息
    print("数据集图片数量: ", picture_count)
    print("数据集label数量: ",len(label_dict))
    # sorted_cnt = [(k, label_dict[k]) for k in sorted(label_dict, key=label_dict.get, reverse=True)]
    # top3 = sorted_cnt[:3]
    # last3 = sorted_cnt[-3:]
    # print("数据集label样例数 Top3:", top3)
    # print("数据集label样例数 Last3:", last3)

    df = pd.DataFrame.from_dict(data=label_dict, orient='index')
    df.columns = ['cnt']
    s = df['cnt'].sort_values(ascending=False)
    print(s.head(3))
    print(s.tail(3))
    print(s.describe(percentiles=[0.25, 0.5, 0.75, 0.8]))
    

    for i in range(len(obj_per_frame)-1, -1, -1):
        if obj_per_frame[i] != 0:
            print("每张图片最多有", i, "个target.")
            break    
    if obj_per_frame[0] != 0:
        print("有", obj_per_frame[0], '张图片里没有target.')
    # print(label_dict)
    # print(group_per_label)
    visualize_count(label_dict)
    # visualize_group(label_per_group, label='labels_per_group')
    # visualize_group(group_per_label, label='groups_per_label')

data_dir = '../../data'
label_path = os.path.join(data_dir, 'traffic.label')
# 交通信号数据集路径
GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')
# 是否统计交通灯数据集
is_deal_traffic_data = True

# 武大数据集路径
norm_data_path = os.path.join(data_dir, 'TrafficNorm')
# 是否统计武大数据集
is_deal_norm_data = False

font= '/usr/share/fonts/winFonts/simsun.ttc'

if __name__ == '__main__':
    main()
