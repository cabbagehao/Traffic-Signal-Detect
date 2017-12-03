'''
    对label数量进行排序,低于75%以下的upsample,高于75%的进行down Sample
'''
import os
import shutil
import random
import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
from lxml import etree as ET
from tqdm import tqdm

def run_over_sampe(up_sample_imgs, up_sample_group): 
    '''
        过采样出的图片放到同一个组, 序号在原图片上加10000
    '''
    # 组装xml
    root = ET.Element("opencv_storage")
    node_frame_number = ET.SubElement(root, "FrameNumber")
    node_frame_number.text = str(len(up_sample_imgs))

    # 拷贝所有img到up sample目录
    up_sample_dir = os.path.join(new_traffic_data_path, 'TSD-Signal-' + up_sample_group)
    up_sample_gt = os.path.join(new_GT_xmls_dir, 'TSD-Signal-' + up_sample_group + '-GT.xml')
    if not os.path.exists(up_sample_dir): 
        # shutil.rmtree(up_sample_dir)
        os.makedirs(up_sample_dir)

    print('over sample...')
    for i in tqdm(range(len(up_sample_imgs))):
        img = up_sample_imgs[i]
        img_name = img.split('/')[-1].split('-')
        group = img_name[2]
        frame = img_name[-1].split('.')[0]
        new_frame = 10000 + i # + group*100 + frame
        # 复制图片
        new_img_name = 'TSD-Signal-' + up_sample_group + '-' + str(new_frame) + '.png'
        img_path = os.path.join(up_sample_dir, new_img_name)
        try:
            shutil.copy(img, img_path)
        except FileNotFoundError:
            print('oversample: img_path not exists.', img_path)
            continue

        # build new xml node
        xml_path = os.path.join(new_GT_xmls_dir, 'TSD-Signal-' + group + '-GT.xml')
        tree = etree.parse(xml_path)
        data = tree.getroot()
        for node in data:
            if 'Frame' + frame in node.tag:
                node.tag = 'Frame' + str(new_frame) + node.tag[10:]
                root.append(node)

    # 存在xml文件则修改, 不存在则新建
    if os.path.exists(up_sample_gt):
        root_xml = etree.parse(up_sample_gt).getroot()
        root_xml.extend(root)
        root = root_xml

    with open(up_sample_gt, "w") as f:
        xml = ET.tostring(root, encoding = 'utf-8', xml_declaration = True, pretty_print = True)
        xml = xml.decode('utf-8')
        f.writelines(xml)

def run_down_sample(down_sample_imgs):
    # 删除img文件
    print("Running down sample")
    for img in tqdm(set(down_sample_imgs)):
        os.remove(img)
        # print('Down sample rm img: ', img)
        img_name = img.split('/')[-1].split('-')
        group = img_name[2]
        frame = img_name[-1].split('.')[0]

        # 修改xml文件的FrameNumber并删除该图片对应的node
        xml_path = os.path.join(new_GT_xmls_dir, 'TSD-Signal-' + group + '-GT.xml')
        del_frame_in_xml(xml_path, frame)

def down_sample(label_dict, label, delta):
    sample_list = []
    img_path_list = label_dict[label]
    # 去重,去掉10000组里的数据
    for path in img_path_list:
        if path not in sample_list :
            sample_list.append(path)
    # 随机shuffle
    random.shuffle(sample_list)
    # 取末尾delta个, 低于30张照片不执行下采样.
    if len(sample_list) <= delta-30:
        print('len(sample_list) <= delta-10, can\'t down sample.')
        return []
    sample_list = sample_list[:delta]
    return sample_list

def up_sample(label_dict, label, delta):
    sample_list = []
    img_path_list = label_dict[label]
    assert len(img_path_list) != 0, 'img_path_list is empty.'
    random.shuffle(img_path_list)
    for img_path in img_path_list:
        if img_path.split('-')[-2] != '10000':
            sample_list.append(img_path)
    # 如果Samplelist的图片数量够,则截取
    # 如果samplelist的图片数量不够,直接复制
    if len(sample_list) == 0 or len(img_path_list) == 0:
        print('sample_list or img_path_list is empty', label, len(sample_list), len(img_path_list))
        sample_list = img_path_list

    if len(sample_list) > delta:
        sample_list = sample_list[:delta]

    if len(sample_list) < delta:
        for i in range(delta - len(sample_list)):
            idx = random.randint(0, len(sample_list)-1)
            sample_list.append(sample_list[idx])
    assert len(sample_list) == delta
    return sample_list    

def get_sample_path(label_dict, cnt_threshold):
    up_sample_path = []
    down_sample_path = []
    for label in label_dict:
        delta = len(set(label_dict[label])) - cnt_threshold
        if delta < 0:
            sample_path = up_sample(label_dict, label, abs(delta))
            up_sample_path.extend(sample_path)
        else:
            sample_path = down_sample(label_dict, label, abs(delta))
            down_sample_path.extend(sample_path)
    
    return up_sample_path, down_sample_path


def get_cnt_threshold(proportion):
    label_cnt = {}
    for key in label_dict.keys():
        label_cnt[key] = len(label_dict[key])
    df = pd.DataFrame.from_dict(data=label_cnt, orient='index')
    s = df[0].sort_values(ascending=False)
    cnt_threshold = int(s.describe(percentiles=[proportion])['80%'])    
    return cnt_threshold

def del_frame_in_xml(xml_path, frame):
    """ 删除xml里该frame对应的node"""
    tree = etree.parse(xml_path)
    data = tree.getroot()

    frame_number_node = data.xpath('FrameNumber')[0]
    frame_number_node.text = str(int(frame_number_node.text) - 1)
    for node in data:
        if 'Frame' + frame in node.tag:
            data.remove(node)
    with open(xml_path, "w") as f:
        xml = ET.tostring( tree, encoding = 'utf-8', xml_declaration = True, pretty_print = True)
        xml = xml.decode('utf-8')
        f.writelines(xml) 

def get_label_imgpath(data_dir, gt_dir):
    # 获得每一个target label所在图片的路径
    # 每一个type的所有target都有一个图片路径,所以可能包含重复图片.
    label_dict = {}
    for xml_path in os.listdir(gt_dir):
        group_name = xml_path.split('-')[2]
        xml_path = os.path.join(gt_dir, xml_path)
        parser = ET.XMLParser(encoding="utf-8")
        tree = etree.parse(xml_path, parser=parser)
        # layer 1
        data = tree.getroot()
        # layer 2
        for node in data:
            if node.tag == 'FrameNumber':
                frame_number_node = node
                continue

            # node.tag: Frame00000Target00000
            frame_name = node.tag.split('Target')[0].split('Frame')[1]
            group_path = os.path.join(data_dir, 'TSD-Signal-' + group_name )
            img_path = os.path.join(group_path, 'TSD-Signal-' + group_name + '-' + frame_name + '.png')
            # 删除target为0无效图片
            if 'TargetNumber' in node.tag:
                obj_num = int(node.text)
                if obj_num == 0:
                    data.remove(node)
                    if os.path.exists(img_path): 
                        os.remove(img_path)
                        print('rm no target img: ', img_path)
                        # 修改xml文件的FrameNumber并删除该图片对应的node
                        xml_path = os.path.join(new_GT_xmls_dir, 'TSD-Signal-' + group_name + '-GT.xml')
                        del_frame_in_xml(xml_path, frame_name)                               
                continue
            # layer 3
            childs = node.getchildren()
            for child in childs:
                if child.tag == 'Type':
                    label = child.text   

                    if label not in label_dict:
                        label_dict[label] = []
                    label_dict[label].append(img_path)

    return label_dict

def copy_origin_data(old_data_dir, old_gt_dir, new_data_dir, new_gt_dir,):
    print('Copying data...')
    # empty new dir
    if os.path.exists(new_data_dir): shutil.rmtree(new_data_dir)
    if os.path.exists(new_gt_dir):  shutil.rmtree(new_gt_dir)
    os.makedirs(new_data_dir)
    os.makedirs(new_gt_dir)

    # copy data
    groups = os.listdir(old_data_dir)
    xmls = os.listdir(old_gt_dir)
    for i in tqdm(range(len(groups))):
        src_data_dir = os.path.join(old_data_dir, groups[i])
        dst_data_dir = os.path.join(new_data_dir, groups[i])
        shutil.copytree(src_data_dir, dst_data_dir)

        src_xml_path = os.path.join(old_gt_dir, xmls[i])
        # dst_xml_dir = os.path.join(new_gt_dir, xmls[i])
        shutil.copy(src_xml_path, new_gt_dir)


random.seed(2017)
data_dir = '../../data'
label_path = os.path.join(data_dir, 'traffic.label')
# 交通信号数据集路径和GT文件路径
traffic_data_path = os.path.join(data_dir, 'TSD-Signal_origin')
GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT_origin')

# 新路径
new_traffic_data_path = os.path.join(data_dir, 'TSD-Signal')
new_GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')

# is_need_copy_data = False
is_need_copy_data = True
# 过采样图片目录
up_sample_group = '10000'

if __name__ == '__main__':
    if is_need_copy_data:
        copy_origin_data(traffic_data_path, GT_xmls_dir, new_traffic_data_path, new_GT_xmls_dir)

    # proportion = 0.8
    # cnt_threshold = get_cnt_threshold(proportion)
    # cnt_threshold = 80

    # 执行80的下采样
    label_dict = get_label_imgpath(new_traffic_data_path, new_GT_xmls_dir)
    up_sample_imgs, down_sample_imgs = get_sample_path(label_dict, 80)
    run_down_sample(down_sample_imgs)

    # 执行60的上采样
    label_dict = get_label_imgpath(new_traffic_data_path, new_GT_xmls_dir)
    up_sample_imgs, down_sample_imgs = get_sample_path(label_dict, 60)
    run_over_sampe(up_sample_imgs, up_sample_group)
    

    # print('Upsample cnt: ', len(up_sample_imgs))
    # print('Downsample cnt: ', len(down_sample_imgs))