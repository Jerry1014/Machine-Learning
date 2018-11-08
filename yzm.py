import os
import random
import shutil
from time import sleep

import cv2
import numpy
import pytesseract
import requests


def interference_point(x_s, y_s, img):
    """
    点降噪
    :param x_s: 图片的高
    :param y_s: 图片的宽
    :param img: 单通道/三通道（使用最大值，按单通道处理）的图片
    """
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
    """
    请求验证码，并灰度二值去噪处理
    :return :切割后的验证码图片，num */+ num，一共三张（‘=’被抛弃）
    """
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
    # 创建保存目录
    for i in all_char:
        if not os.path.exists(train_save_path + str(i)):
            os.mkdir(train_save_path + str(i))

    for i in range(repeat_num):
        for image_tem in get_yzm():
            char1 = pytesseract.image_to_string(image_tem, config='--psm 10')
            _, result, *_ = knn.findNearest(image_tem.reshape(1, -1).astype(numpy.float32), k=5)
            char2 = all_char[int(result[0])]
            if char1 not in all_char:
                char1 = '++'

            file_name = random.randint(0, 10000)
            if char1 != char2:
                file_path = train_save_path + str(file_name) + '.jpg'
            else:
                file_path = train_save_path + char1 + '\\' + str(file_name) + '.jpg'

            while os.path.exists(file_path):
                file_name = random.randint(0, 1000)
                file_path = train_save_path + char1 + '\\' + str(file_name) + '.jpg'

            cv2.imwrite(file_path, image_tem)

        print(str(i + 1) + '/' + str(repeat_num))


def pre_train():
    """将准备好的图片训练集做一次整理，并将结果写到train.npz"""
    try:
        with numpy.load(destination_path + 'train.npz') as data:
            train = data['train']
            train_labels = data['train_labels']
    except:
        train = numpy.empty((0, 300))
        train_labels = numpy.empty((0, 9))

    train.dtype == numpy.float32
    train_labels.dtype == numpy.int

    for i in all_char:
        try:
            all_file = os.listdir(train_save_path + str(i))
        except FileNotFoundError:
            break
        label = numpy.zeros([1, 9])
        label[0, all_char.index(i)] = 1
        train_labels = numpy.append(train_labels, numpy.repeat(label.astype(int), len(all_file), axis=0), axis=0) \
            .astype(numpy.int)
        for j in all_file:
            img = cv2.imread(train_save_path + str(i) + '\\' + j, flags=cv2.IMREAD_GRAYSCALE)
            train = numpy.append(train, img.reshape((1, -1)).astype(numpy.int), 0).astype(numpy.float32)

    # 打乱训练集
    permu = list(numpy.random.permutation(len(train)))
    train_shuffled = train[permu]
    train_labels_shuffled = train_labels[permu]
    # 保存已经处理好的训练集
    numpy.savez(destination_path + 'train.npz', train=train_shuffled, train_labels=train_labels_shuffled)


if __name__ == '__main__':
    destination_path = 'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\train\\'
    train_save_path = destination_path + 'new\\'

    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)

    all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']

    get_train_yzm(10)

    input('请手动处理无法识别的图片')
    pre_train()
    shutil.rmtree(train_save_path[:-1])
