import sys

import cv2
import numpy
import pytesseract
import tensorflow as tf

yzm_downloader_path = r'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\'

sys.path.append(yzm_downloader_path)
from yzm import get_yzm

all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']
destination_path = 'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\train\\'
train_save_path = destination_path + 'new\\'


def get_train_yzm(repeat_num):
    """用来爬取验证码图片，分割，用ocr/knn识别并标记"""
    knn = cv2.ml.KNearest_create()
    with numpy.load(destination_path + 'basic_knn.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(
            r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\models\aao_cnn_model.meta')
        new_saver.restore(sess,
                          tf.train.latest_checkpoint(r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\models'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        k = graph.get_tensor_by_name("keep_prop:0")
        y = graph.get_tensor_by_name("result:0")
        wrong = [0, 0, 0]

        for i in range(repeat_num):
            for image_tem in get_yzm():
                char1 = pytesseract.image_to_string(image_tem, config='--psm 10')
                _, result, *_ = knn.findNearest(image_tem.reshape(1, -1).astype(numpy.float32), k=5)
                char2 = all_char[int(result[0])]
                if char1 not in all_char:
                    char1 = '++'

                char3 = all_char[sess.run(y, feed_dict={
                    x: numpy.reshape(image_tem, newshape=[-1, 300]).astype(numpy.float32), k: 1.0})[0]]
                if char1 != char2 and char2 == char3:
                    wrong[0] += 1
                elif char1 != char2 and char1 == char3:
                    wrong[1] += 1
                elif char1 != char3 and char2 == char1:
                    wrong[2] += 1
                elif char1 != char2 and char1 != char3:
                    cv2.imshow('0', image_tem)
                    char_true = cv2.waitKey(0)
                    if char1 != char_true:
                        wrong[0] += 1
                    if char2 != char_true:
                        wrong[1] += 1
                    if char3 != char_true:
                        wrong[2] += 1

            print(i, '/', repeat_num)

        print("the accruary of OCR is ", 1 - wrong[0] / repeat_num)
        print("the accruary of KNN is ", 1 - wrong[1] / repeat_num)
        print("the accruary of CNN is ", 1 - wrong[2] / repeat_num)


get_train_yzm(500)
