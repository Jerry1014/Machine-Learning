# -*- coding: utf-8 -*-

import cv2
import numpy
import pytesseract


def test_knn(test_repeat_num):
    """knn训练完成之后的成功率测试"""
    correct_num = 0
    for i in range(test_repeat_num):
        for image_tem in get_yzm():
            char = pytesseract.image_to_string(image_tem, config='--psm 10')
            ret, result, *_ = knn.findNearest(image_tem.reshape(1, -1).astype(numpy.float32), k=the_k_of_knn)
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
        print("已完成", i, "/", test_repeat_num)

    print("正确率为", correct_num / (test_repeat_num * 3) * 100.0, '%')
    print("如果正确率过低，可尝试提高训练集的数量和质量，或者提高the_k_of_knn的值")


if __name__ == '__main__':
    # 路径参数
    train_file = 'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\train\\train.npz'
    yzm_downloader_path = r'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\'

    import sys

    sys.path.append(yzm_downloader_path)
    from yzm import get_yzm

    # 定义全部的符号，教务处只有以下符号，因目录问题，用++代表*
    all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']

    # 利用设置好的训练集，训练knn模型
    knn = cv2.ml.KNearest_create()
    with numpy.load(train_file) as data:
        train = data['train']
        train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    # 用knn进行判断时，选择的k
    the_k_of_knn = 5

    test_knn(50)
