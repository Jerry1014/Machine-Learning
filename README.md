# Machine-Learning
利用教务处网站的验证码做训练集，进行简单的机器学习实验

## KNN
- aao_knn
    - 训练knn模型(Class AaoKnn)，且能在训练后爬取新的验证码进行测试(test_knn(repeat_num))

## Keras
- aao_cnn
    - 利用keras实现cnn，对教务处的验证码进行识别，保存
    - 相对tensorflow版本，优化网络，减少训练时间
    - epoch为1时，在7219的训练集中达到1.0的正确率

## TensorFlow
- aao_cnn
    - 利用tensorflow实现cnn，对教务处的验证码进行识别，保存
- example
    - 利用cnn识别数据集MNIDST的示例
    - 参考博客 https://blog.csdn.net/Sparta_117/article/details/66965760

## train
- basic_knn.npz
    - 已包装好的knn训练集
    - one-hot = false
    - 几百张图片，乱序
- basic_cnn.npz
    - 已包装好的cnn训练集
    - ont-hot = ture
    - 7000张，乱序
- label ['1', '2', '3', '4', '5', '6', '7', '+', '*']
- 用numpy.load()导入，图片训练集在['train']，标签在['train_labels']
***
- model
    - 已包装好的cnn模型
- cheak
    - 对三种识别方法的正确性测试，在最近的一次测试中，一共爬取了500张验证码，OCR的正确率为0.978，而KNN和CNN均为1.0
- train.npz
    - get_yzm爬取的数据集，one-hot
    - 已包装好的cnn训练集
    - 7000张，乱序

## yzm
- 用以批量下载验证码作为训练/测试集
- 被aao_knn, cheak调用
- 可在原来的数据集基础上，继续爬取添加新的数据
- 爬取的数据会根据ocr，knn自行标识分类，存在异议时，保存在分类目录外，由人工分类
- 最后以乱序保存到数据集内，并会删除爬取的所有图片
