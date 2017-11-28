# coding=utf-8
import os
import xml.etree.ElementTree as ET
from lxml import etree as ET

# Create a xml file by xml.etree.ElementTree
def build_xml(result_xml_path, frames_dict):
    frame_number = frames_dict.get('frame_number', 0)
    del frames_dict['frame_number']
    target_number = 3
    root = ET.Element("opencv_storage")
    node_frame_number = ET.SubElement(root, "FrameNumber")
    node_frame_number.text = str(frame_number)

    frames_dict = sorted(frames_dict.items(), key=lambda d: d[0])
    for frame_id, targets in frames_dict:
        # Create Node FrameNumber
        target_num = len(targets)
        #target_num_str =  ('00000' + str(target_num))[-5:]
        frame_name = 'Frame' + frame_id
        node_target_number = frame_name + 'TargetNumber' # + target_num_str
        node_target = ET.SubElement(root, node_target_number)
        node_target.text = str(target_num)

        # Create all target Node
        # target eg: [label, box, max_pred]
        for target_id in range(target_num):
            target = targets[target_id]
            target_type = target[0]
            tar_position = target[1]

            target_id_str = ('00000' + str(target_id))[-5:]
            target_name = frame_name + 'Target' + target_id_str
            node_target = ET.SubElement(root, target_name)

            node_type = ET.SubElement(node_target, 'Type')
            target_type = '\"' + target_type + '\"'
            node_type.text = target_type

            node_position = ET.SubElement(node_target, 'Position')
            tar_position
            x1, y1 = tar_position[0], tar_position[1]
            x2, y2 = tar_position[2], tar_position[3]
            width = int(round(x2 - x1))
            height = int(round(y2 - y1))
            position_str = str(int(round(x1))) + ' ' + str(int(round(y1))) + ' ' + str(width) + ' ' + str(height)
            node_position.text = position_str

    pretty_xml = ET.tostring(
        root, encoding = 'gbk', xml_declaration = True, pretty_print = True)
    file = open(result_xml_path, "w")
    pretty_xml = pretty_xml.decode('gbk')
    pretty_xml = pretty_xml.replace('\'', '\"')
    file.writelines(pretty_xml)
    file.close()
    conv_cmd = 'iconv -f UTF-8 -t gbk -c ' + result_xml_path + ' > tmp; mv -f tmp ' + result_xml_path
    os.system(conv_cmd)
# Pretty a xml file
# def pretty_xmlfile(result_xml_path):
#     parser = ET.XMLParser(
#         remove_blank_text=False, resolve_entities=True, strip_cdata=True)
#     xmlfile = ET.parse(result_xml_path, parser)
#     pretty_xml = ET.tostring(
#         xmlfile, encoding = 'gbk', xml_declaration = True, pretty_print = True)
#     file = open(result_xml_path, "w")
#     pretty_xml = pretty_xml.decode('gbk')
#     file.writelines(pretty_xml)
#     file.close()

if __name__ == '__main__':
    path = 'test.xml'
    frame_dict_ = { '00038': [['警1-T型交叉左', 
                             [  953.53875732,   271.48202515,  1021.10412598,   331.95425415], 
                             0.99213398]], 
                    'frame_number': 6}
    build_xml(path, frame_dict_)
    # pretty_xmlfile(path)
