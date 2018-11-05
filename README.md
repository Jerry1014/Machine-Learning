# Machine-Learning
利用教务处网站的验证码做训练集，进行简单的机器学习实验

## KNN
- aao_knn
    - 利用opencv库，实现从教务处网站自动下载，处理，利用OCR标识验证码，并训练knn模型，且能在训练后爬取新的验证码进行测试

## tensorflow
- aao_cnn
    - 利用tensorflow实现cnn，对教务处的验证码进行识别
- example
    - 利用cnn识别数据集MNIDST的示例
    - 参考博客 https://blog.csdn.net/Sparta_117/article/details/66965760

## train
- basic_knn.npz
    - 已包装好的knn训练集
- basic_cnn.npz
    - 已包装好的cnn训练集
- model
    - 已包装好的cnn模型
- cheak
    - 对三种识别方法的正确性测试，在最近的一次测试中，一共爬取了500张验证码，OCR的正确率为0.978，而KNN和CNN均为1.0
- get_yzm
    - 用以批量下载验证码作为训练/测试集
- train.npz
    - get_yzm爬取的数据集，one-hot