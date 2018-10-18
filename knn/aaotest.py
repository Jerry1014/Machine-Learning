# -*- coding: utf-8 -*-
import requests
import os
import random
from time import sleep
import cv2
import pytesseract
import numpy


# 用knn进行判断时，选择的k
the_k_of_knn = 5


def interference_point(x_s, y_s):
    """点降噪"""
    img = cv2.imread('vc.jpg')
    # todo 判断图片的长宽度下限
    height, width = img.shape[:2]

    for y in range(y_s, width - 1):
        for x in range(x_s, height - 1):
            cur_pixel = img[x, y].max()  # 当前像素点的值
            # 具备9领域条件的
            sum = int(img[x - 1, y - 1].max()) \
              + int(img[x - 1, y].max()) \
              + int(img[x - 1, y + 1].max()) \
              + int(img[x, y - 1].max()) \
              + int(cur_pixel) \
              + int(img[x, y + 1].max()) \
              + int(img[x + 1, y - 1].max()) \
              + int(img[x + 1, y].max()) \
              + int(img[x + 1, y + 1].max())
            if sum >= 6 * 245:
                img[x, y] = 255
    cv2.imwrite('vc.jpg', img)


def get_yzm():
    """请求验证码"""
    try:
        yzm = requests.get("https://zhjw.neu.edu.cn/ACTIONVALIDATERANDOMPICTURE.APPPROCESS")
    except:
        sleep(10)
    if yzm.status_code != 200:
        sleep(10)

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
    interference_point(width_q, height_q)

    # 分割验证码
    yzm_images = list()
    x_r = int(w / 4)
    x_l = 0
    for i in range(3):
        yzm_images.append(im[:, x_l:x_r])
        x_l = x_r
        x_r += int(w / 4)

    os.remove('vc.jpg')
    sleep(1)
    return yzm_images


def get_train_yzm():
    """用来爬取验证码图片，分割，用ocr识别并标记"""
    # 创建保存目录
    for i in list(range(10))+['+', '++']:
        if not os.path.exists('D:\\tem\\' + str(i)):
            os.mkdir('D:\\tem\\' + str(i))

    for i in range(repeat_num):
        for image_tem in get_yzm():
            char = pytesseract.image_to_string(image_tem, config='--psm 10')
            if char not in all_char:
                char = '++'

            file_name = random.randint(0, 1000)
            file_path = 'D:\\tem\\' + char + '\\' + str(file_name) + '.jpg'
            while os.path.exists(file_path):
                file_name = random.randint(0, 1000)
                file_path = 'D:\\tem\\' + char + '\\' + str(file_name) + '.jpg'

            cv2.imwrite(file_path, image_tem)

        print(str(i + 1) + '/' + str(repeat_num))


def train_knn():
    """将准备好的图片训练集做一次整理，并将结果写到data.npz"""
    train = numpy.empty((0, 900))
    train_labels = numpy.empty((0, 1))
    train.dtype == numpy.float32
    train_labels.dtype == numpy.int

    for i in all_char:
        all_file = os.listdir('D:\\tem\\' + str(i))
        train_labels = numpy.append(train_labels, numpy.repeat(numpy.array([all_char.index(i)]), len(all_file))).astype(numpy.int)
        for j in all_file:
            img = cv2.imread('D:\\tem\\' + str(i) + '\\' + j)
            train = numpy.append(train, img.reshape((1, -1)).astype(numpy.int), 0).astype(numpy.float32)

    # 保存已经处理好的训练集
    numpy.savez('data.npz', train=train, train_labels=train_labels)


def test_knn():
    """knn训练完成之后的成功率测试"""
    correct_num = 0
    for i in range(test_repeat_num):
        for image_tem in get_yzm():
            char = pytesseract.image_to_string(image_tem, config='--psm 10')
            tem = numpy.repeat(image_tem, 3).reshape((1,-1)).astype(numpy.float32)
            ret, result, *_ = knn.findNearest(tem, k=the_k_of_knn)
            # 将knn模型识别的结果与ocr的作比较
            if all_char[int(result[0])] == char:
                correct_num += 1
            else:
                cv2.imshow('result', image_tem)
                char = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if char > 48:
                    char -= 49
                elif char == 43:
                    char = 7
                elif char == 42:
                    char = 8
                else:
                    char = -1

                if result[0] == char:
                    correct_num += 1
        print("已完成",i,"/",test_repeat_num)

    print("正确率为",correct_num/(test_repeat_num*3)*100.0,'%')
    print("如果正确率过低，可尝试提高训练集的数量和质量，或者提高the_k_of_knn的值")


if __name__ == '__main__':
    # 每次运行get_train_yzm爬取的验证码数量（张）
    repeat_num = 100

    # 定义全部的符号，教务处只有以下符号，因目录问题，用++代表*
    all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']

    # 利用设置好的训练集，训练knn模型
    knn = cv2.ml.KNearest_create()
    with numpy.load('data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    # 每次运行test_knn，爬取的验证码数量（张）
    test_repeat_num = 50

    test_knn()
