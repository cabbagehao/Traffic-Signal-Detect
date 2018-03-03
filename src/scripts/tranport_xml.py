# coding:utf-8
# 转换voc格式的xml到需要的xml格式。

import os
import shutil
from lxml import etree

img_dir = 'test_samples'
xml_dir = 'origin_xml'
output_dir = 'new_xml'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

def main():
    group_list = os.listdir(img_dir)
    for group_name in group_list:
        img_group_dir = os.path.join(img_dir, group_name)
        xml_group_dir = os.path.join(xml_dir, group_name)
        new_xml_name = os.path.join(output_dir,  group_name + '-GT.xml')
        # 创建xml的frame_number节点
        new_xml_root = etree.Element('opencv_storage')
        frame_number_node = etree.Element('FrameNumber')
        frame_number_node.text = str(len(os.listdir(img_group_dir)))
        new_xml_root.append(frame_number_node)

        for img_name in os.listdir(img_group_dir):
            # img_file = os.path.join(img_group_dir, img_name)
            xml_file = os.path.join(xml_group_dir, img_name.split('.')[-2] + '.xml')
            frame = img_name.split('.png')[0].split('-')[-1]
            if not os.path.exists(xml_file):
                print('xml_file missing.', xml_file)
                # xml miss是因为图片里没有target。 targetNumber置零            
                target_number_tag = 'Frame' + frame + 'TargetNumber'
                target_number_node = etree.Element(target_number_tag)
                target_number_node.text = '0'
                new_xml_root.append(target_number_node)
                continue

            # 创建图片的TargetNumber节点
            target_number_tag = 'Frame' + frame + 'TargetNumber'
            target_number_node = etree.Element(target_number_tag)
            new_xml_root.append(target_number_node)

            tree = etree.parse(xml_file)
            data = tree.getroot()
            target_nodes = data.xpath('object')
            target_number_node.text = str(len(target_nodes))
            for i in range(len(target_nodes)):
                target = target_nodes[i]
                label = target.xpath('name')[0].text
                xmin = target.xpath('bndbox/xmin')[0].text
                ymin = target.xpath('bndbox/ymin')[0].text
                xmax = target.xpath('bndbox/xmax')[0].text
                ymax = target.xpath('bndbox/ymax')[0].text
                w, h = str(int(xmax)-int(xmin)), str(int(ymax)-int(ymin))

                # 创建target节点
                target_tag = 'Frame' + frame + 'Target' + '0000' + str(i)
                target_node = etree.Element(target_tag)
                etree.SubElement(target_node, 'Type').text = '\"' + label + '\"'
                etree.SubElement(target_node, 'Position').text = xmin + ' ' + ymin + ' ' + w + ' ' + h
                new_xml_root.append(target_node)

        with open(new_xml_name, "w") as f:
            xml = etree.tostring(new_xml_root, encoding = 'utf-8', xml_declaration = True, pretty_print = True)
            f.writelines(xml)

main()