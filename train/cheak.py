import os
from time import sleep

import cv2
import numpy
import pytesseract
import requests
import tensorflow as tf

all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']
destination_path = 'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\train\\'
train_save_path = destination_path + 'new\\'


def interference_point(x_s, y_s, img):
    """点降噪"""
    height, width = img.shape[:2]

    for y in range(y_s, width - 1):
        for x in range(x_s, height - 1):
            cur_pixel = img[x, y]  # 当前像素点的值
            # 具备9领域条件的
            sum = int(img[x - 1, y - 1]) \
                  + int(img[x - 1, y]) \
                  + int(img[x - 1, y + 1]) \
                  + int(img[x, y - 1]) \
                  + int(cur_pixel) \
                  + int(img[x, y + 1]) \
                  + int(img[x + 1, y - 1]) \
                  + int(img[x + 1, y]) \
                  + int(img[x + 1, y + 1])
            if sum >= 6 * 245:
                img[x, y] = 255


def get_yzm():
    """请求验证码"""
    try:
        yzm = requests.get("https://zhjw.neu.edu.cn/ACTIONVALIDATERANDOMPICTURE.APPPROCESS")
    except:
        sleep(10)
        return None

    if yzm.status_code != 200:
        sleep(10)
        yzm = requests.get("https://zhjw.neu.edu.cn/ACTIONVALIDATERANDOMPICTURE.APPPROCESS")
    if yzm.status_code != 200:
        return None

    with open('vc.jpg', 'wb') as f:
        f.write(yzm.content)

    im = cv2.imread('vc.jpg')
    # 灰值化
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 二值化
    im = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)[1]

    # 去除四边的噪声
    h, w = im.shape[:2]
    width_q = 5
    height_q = 2
    for y in range(0, w):
        for x in range(0, h):
            if y < width_q or y > w - width_q or x < height_q or x > h - height_q:
                im[x, y] = 255
    interference_point(width_q, height_q, im)

    # 分割验证码
    yzm_images = list()
    x_r = int(w / 4)
    x_l = 0
    for i in range(3):
        yzm_images.append(im[:, x_l:x_r])
        x_l = x_r
        x_r += int(w / 4)

    os.remove('vc.jpg')
    sleep(0.5)
    return yzm_images


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
