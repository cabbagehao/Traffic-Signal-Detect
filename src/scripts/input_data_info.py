import os
import tensorflow as tf
from lxml import etree


def get_info():
    picture_count = 0
    label_dict ={}
    obj_per_frame = [0 for i in range(10)]

    for xml_path in os.listdir(GT_xmls_dir):
  
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
                    if not label in label_dict:
                        label_dict[label] = 0
                    label_dict[label] += 1

    return picture_count, label_dict, obj_per_frame

def main()                :
    picture_count, label_dict, obj_per_frame = get_info()

    print("picture count: ", picture_count,'\n')
    print("label count: ", label_dict, '\n')

    print('obj_per_frame: ', obj_per_frame)

data_dir = '../../data'
GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')

if __name__ == '__main__':
    main()