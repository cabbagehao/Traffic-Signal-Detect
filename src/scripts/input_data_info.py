import os
import tensorflow as tf
import matplotlib.pyplot as plt

from lxml import etree

def get_info():
    picture_count = 0
    label_dict ={}
    obj_per_frame = [0 for i in range(10)]

    # 处理数据集1:
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

    # 处理数据集2
    for label_dir in os.listdir(TrafficNorm):
        label = label_dir
        img_dir = os.path.join(TrafficNorm, label_dir)
        num = len(os.listdir(img_dir))

        if not label in label_dict:
            label_dict[label] = 0
        label_dict[label] += num
        picture_count += num

    return picture_count, label_dict, obj_per_frame

def visualize(label_dict):
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


def main()                :
    picture_count, label_dict, obj_per_frame = get_info()

    print("picture count: ", picture_count,'\n')
    print("label count: ", label_dict, '\n')
    print('obj_per_frame: ', obj_per_frame)

    visualize(label_dict)

data_dir = '../../data'
GT_xmls_dir = os.path.join(data_dir, 'TSD-Signal-GT')
TrafficNorm = os.path.join(data_dir, 'TrafficNorm')

if __name__ == '__main__':
    main()