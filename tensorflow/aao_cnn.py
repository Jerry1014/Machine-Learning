import tensorflow as tf
import numpy
import cv2
import os


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=length))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_trained_model(train_images, train_labels):
    """
    输入训练数据，得到训练后的模型

    train_image:20*15（高*宽）的灰度二值化的图像，单通道
    train_labels:图像对应的标签
    """
    x = tf.placeholder(tf.float32, shape=[None, 300])
    x_image = tf.reshape(x, shape=[-1, 20, 15, 1])

    y = tf.placeholder(tf.float32, shape=[None, 9])

    # 卷积层池化层1
    c_w1 = new_weights([5, 5, 1, 16])
    c_b1 = new_biases([16])
    layer_conv1 = tf.nn.relu(conv2d(x_image, c_w1) + c_b1)
    layer_pool1 = max_pool_2x2(layer_conv1)

    # 卷积层池化层2
    c_w2 = new_weights([5, 5, 16, 32])
    c_b2 = new_biases([32])
    layer_conv2 = tf.nn.relu(conv2d(layer_pool1, c_w2) + c_b2)
    layer_pool2 = max_pool_2x2(layer_conv2)

    # 全连接层1
    fc_w1 = new_weights([5 * 4 * 32, 512])
    fc_b1 = new_biases([512])
    layer_fc1 = tf.nn.relu(tf.matmul(tf.reshape(layer_pool2, [-1, 5 * 4 * 32]), fc_w1) + fc_b1)

    keep_prob = tf.placeholder(tf.float32)
    layer_fc1_drop = tf.nn.dropout(layer_fc1, keep_prob)

    # 输出层
    fc_w2 = new_weights([512, 9])
    fc_b2 = new_biases([9])
    y_pre = tf.nn.relu(tf.matmul(layer_fc1_drop, fc_w2) + fc_b2)

    correct_prediction = tf.equal(tf.argmax(y_pre, dimension=1), tf.argmax(y, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 损失函数和优化器
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pre))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            sess.run(optimizer, feed_dict={x: train_images, y: train_labels, keep_prob: 0.5})
            print('The accuracy is{:.2g}'.format(sess.run(accuracy, feed_dict={x: train_images, y: train_labels, keep_prob: 1.0})))
        saver.save(sess, 'models/aao_cnn_model')


def model_test():
    pass


if __name__ == '__main__':
    train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'
    all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']

    with numpy.load(train_save_path + 'train.npz') as data:
        get_trained_model(data['train'], data['train_labels'])
