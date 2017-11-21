#coding:utf-8
import tensorflow as tf
from PIL import Image

def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 'image/object/class/text': tf.FixedLenFeature([], tf.string),
                                           'image/encoded' :  tf.FixedLenFeature([], tf.string),  
                                           'image/filename' : tf.FixedLenFeature([], tf.string),               
                                       })#将image数据和label取出来

    image = tf.image.decode_png(features['image/encoded']) #图像解码  
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.reshape(image, [1024, 1280,3])
    # label = tf.cast(features['image/object/class/text'], tf.string) #在流中抛出label张量
    img_name = tf.cast(features['image/filename'], tf.string)
    label = 'a'
    return image, label, img_name


record_file = '../../data/records/val2.record'
img_save_file = '../../output/img_from_record/'

image, label, img_name = read_and_decode(record_file)

with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(200):
        image_arr, image_name = sess.run([image, img_name])
        img=Image.fromarray(image_arr, 'RGB')
        image_name = image_name.decode("utf-8")
        img.save(img_save_file + image_name)
        print(i, ': ', image_name)
    coord.request_stop()
    coord.join(threads)
