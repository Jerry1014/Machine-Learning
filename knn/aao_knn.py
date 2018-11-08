# -*- coding: utf-8 -*-

import cv2
import numpy
import pytesseract


def test_knn(test_repeat_num):
    """knn训练完成之后的成功率测试
    :param test_repeat_num: 要爬取用于测试的测试数，每一测试为num */+ num 三项
    """
    all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '*', '']
    correct_num = 0
    for i in range(test_repeat_num):
        try:
            for image_tem in get_yzm():
                char_ocr = pytesseract.image_to_string(image_tem, config='--psm 10')
                char_knn = myKnn.get_result(image_tem)
                # 将knn模型识别的结果与ocr的作比较
                if char_ocr == char_knn:
                    correct_num += 1
                else:
                    cv2.imshow('result', image_tem)
                    char_real = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if char_real > 48:
                        char_real -= 49
                    elif char_real == 43:
                        char_real = 7
                    elif char_real == 42:
                        char_real = 8
                    else:
                        char_real = 9

                    if char_knn == char_real:
                        correct_num += 1
            print("已完成", i, "/", test_repeat_num)
        except ModuleNotFoundError:
            print("获取验证码的模块未找到")

    print("正确率为", correct_num / (test_repeat_num * 3) * 100.0, '%')
    print("如果正确率过低，可尝试提高训练集的数量和质量，或者提高the_k_of_knn的值")


class AaoKnn(object):
    """构建教务处网站验证码的knn的识别类"""

    def __init__(self, knn_train_file=None, k_of_knn=5):
        """
        :param knn_train_file: 用于knn训练的训练集路径+文件
        :param k_of_knn: 分类时所依据的k
        """
        self._k_of_knn = 5
        self._knn_train_file = None
        self._the_max_k_num = 10
        self._all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '*']
        self._knn = cv2.ml.KNearest_create()
        self.set_knn_train_file(knn_train_file)
        self.set_k_of_knn(k_of_knn)

    def set_knn_train_file(self, knn_train_file):
        """设置knn训练的训练集路径+文件，并训练"""
        try:
            with numpy.load(knn_train_file) as data:
                self._knn.train(data['train'], cv2.ml.ROW_SAMPLE, data['train_labels'])
            self._knn_train_file = knn_train_file
            print("已训练")
        except FileNotFoundError:
            print("训练文件不存在")

    def set_k_of_knn(self, k_of_knn):
        """设置k值"""
        if k_of_knn < 0:
            print("k_of_knn不能小于0")
        elif k_of_knn > self._the_max_k_num:
            print("k_of_knn过大，如真需要使用非常大的k值，请修改class AaoKnn中的_the_max_k_num值")
        else:
            self._k_of_knn = k_of_knn

    def get_result(self, test_image):
        """
        根据训练集进行分类
        :param test_image: image need to de sorted,20(h)*15(w),single channel
        :return: char,sort result
        """
        if self._knn_train_file:
            try:
                ret, result, *_ = self._knn.findNearest(test_image.reshape(1, -1).astype(numpy.float32),
                                                        k=self._k_of_knn)
                return self._all_char[int(result)]
            except Exception as e:
                print("发生错误，请检查输入的图像是否正确")
                print(e)
        else:
            print("尚未训练，请先输入训练用的文件")


if __name__ == '__main__':
    # 用于knn训练的训练集路径+文件
    train_file = 'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\train\\basic_knn.npz'
    # 验证码下载模块的路径
    yzm_downloader_path = r'C:\\Users\\Jerry\\PycharmProjects\\Machine-Learning\\'

    # 导入验证码下载模块
    import sys

    sys.path.append(yzm_downloader_path)
    from yzm import get_yzm

    myKnn = AaoKnn(train_file)

    # 正确性测试
    test_knn(50)
