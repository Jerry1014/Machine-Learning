import random

import numpy
import tensorflow as tf

# 超参数
train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'
all_char = ['1', '2', '3', '4', '5', '6', '7', '+', '++']
train_batch_size = 25
repeat_time = 100

with numpy.load(train_save_path + 'train.npz') as data:
    train_images = data['train']
    train_labels = data['train_labels']


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=length))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 300], name='x')
with tf.name_scope('input'):
    y = tf.placeholder(tf.float32, shape=[None, 9])
    # x = tf.placeholder(tf.float32, shape=[None, 784])
    # y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, shape=[-1, 20, 15, 1])
    # x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
    # tf.summary.image('input', x_image, 1)

with tf.name_scope('conv1'):
    # 卷积层池化层1
    c_w1 = new_weights([5, 5, 1, 32])
    c_b1 = new_biases([32])
    layer_conv1 = tf.nn.relu(conv2d(x_image, c_w1) + c_b1)
    layer_pool1 = max_pool_2x2(layer_conv1)

with tf.name_scope('conv2'):
    # 卷积层池化层2
    c_w2 = new_weights([5, 5, 32, 64])
    c_b2 = new_biases([64])
    layer_conv2 = tf.nn.relu(conv2d(layer_pool1, c_w2) + c_b2)
    layer_pool2 = max_pool_2x2(layer_conv2)

with tf.name_scope('full-conncetion1'):
    # 全连接层1
    fc_w1 = new_weights([5 * 4 * 64, 1024])
    # fc_w1 = new_weights([7 * 7 * 64, 1024])
    fc_b1 = new_biases([1024])
    layer_pool1_flat = tf.reshape(layer_pool2, [-1, 5 * 4 * 64])
    # layer_pool1_flat = tf.reshape(layer_pool2, [-1, 7 * 7 * 64])
    layer_fc1 = tf.nn.relu(tf.matmul(layer_pool1_flat, fc_w1)) + fc_b1
    # layer_fc1 = tf.nn.relu(tf.matmul(layer_pool1_flat, fc_w1) + fc_b1)

keep_prob = tf.placeholder(tf.float32, name='keep_prop')
layer_fc1_drop = tf.nn.dropout(layer_fc1, keep_prob, name='dropout')

with tf.name_scope('full-conncetion2'):
    # 全连接层2
    fc_w2 = new_weights([1024, 9])
    fc_b2 = new_biases([9])
    # fc_w2 = new_weights([1024, 10])
    # fc_b2 = new_biases([10])
    y_pre = tf.matmul(layer_fc1_drop, fc_w2) + fc_b2

result = tf.argmax(y_pre, dimension=1, name='result')

with tf.name_scope('accurary'):
    correct_prediction = tf.equal(result, tf.argmax(y, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('train'):
    # 损失函数和优化器
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pre))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(train_save_path + 'logs', graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=1)

    sess.run(tf.global_variables_initializer())
    for i in range(repeat_time):
        # train_images, train_labels = data.train.next_batch(train_batch_size)
        start = random.randint(0, len(train_images) - train_batch_size)
        _, summary_train = sess.run([optimizer, merged],
                                    feed_dict={x: train_images[start:start + train_batch_size],
                                               y: train_labels[start:start + train_batch_size],
                                               keep_prob: 0.5})
        train_writer.add_summary(summary_train, i)

        print(i, '/', repeat_time,
              'The accuracy is {:.2g}'.format(sess.run(accuracy, feed_dict={x: train_images,
                                                                            y: train_labels,
                                                                            keep_prob: 1.0})))
        # print(i, '/', repeat_time,
        #       'The accuracy is {:.2g}'.format(sess.run(accuracy, feed_dict={x: data.test.images[:100],
        #                                                                     y: data.test.labels[:100],
        #                                                                     keep_prob: 1.0})))
    train_writer.close()
    saver.save(sess, train_save_path + 'aao_cnn_model')
