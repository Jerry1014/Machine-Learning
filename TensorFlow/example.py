# 参考博客 https://blog.csdn.net/Sparta_117/article/details/66965760
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=length))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(inputx):
    return tf.nn.max_pool(inputx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# import data
data = input_data.read_data_sets("MNIST_data", one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Conv 1
layer_cony1 = {"weights": new_weights([5, 5, 1, 32]),
               "biases": new_biases([32])}
h_conv1 = tf.nn.relu(conv2d(x_image, layer_cony1["weights"]) + layer_cony1["biases"])
h_pool1 = max_pool_2x2(h_conv1)

# Conv 2
layer_cony2 = {"weights": new_weights([5, 5, 32, 64]),
               "biases": new_biases([64])}
h_conv2 = tf.nn.relu(conv2d(h_pool1, layer_cony2["weights"]) + layer_cony2["biases"])
h_pool2 = max_pool_2x2(h_conv2)

# Full-connected layer 1
fc1_layer = {"weights": new_biases([7 * 7 * 64, 1024]),
             "biases": new_biases([1024])}
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_layer["weights"]) + fc1_layer["biases"])

# Droupout Layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Full-connected layer 2
fc2_layer = {"weights": new_biases([1024, 10]),
             "biases": new_biases([10])}

# Predicted class
y_pred = tf.matmul(h_fc1_drop, fc2_layer["weights"]) + fc2_layer["biases"]
y_pred_cls = tf.argmax(y_pred, dimension=1)

# cost function to be optimized
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_batch_size = 25
    test_batch_size = 100

    start = 1
    while start != 0:
        for i in range(2000):
            writer = tf.summary.FileWriter('D:\\tem\\example\\logs')
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            feed_dict_train_op = {x: x_batch, y_true: y_true_batch, keep_prob: 0.5}
            _, summary = sess.run([optimizer, merged], feed_dict=feed_dict_train_op)
            writer.add_summary(summary, i)

            print(i, "test accuracy %g" % accuracy.eval(feed_dict={
                x: data.test.images[:100], y_true: data.test.labels[:100], keep_prob: 1.0}))

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: data.test.images[:1000], y_true: data.test.labels[:1000], keep_prob: 1.0}))
        start = int(input("Continue?\n"))
