import random

import numpy
import tensorflow
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# 超参数
lr = 0.0001
train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'  # 训练后的日志模型保存目录
train_batch_size = 20
test_batch_size = 200
repeat_time = 2000
n_inputs = 20  # data input (img shape: 20*15)
n_steps = 15  # time steps
n_hidden_units = 512
n_middle_units = 128
n_classes = 9  # classes

# 导入训练集
with numpy.load(train_save_path + 'train.npz') as data:
    train_images = data['train']
    train_labels = data['train_labels']
    test_images = data['train'][5000:]
    test_labels = data['train_labels'][5000:]

# 对 weights biases 初始值的定义
weights = {
    'in': tensorflow.Variable(tensorflow.random_normal([n_inputs, n_hidden_units])),
    'mid': tensorflow.Variable(tensorflow.random_normal([n_hidden_units, n_middle_units])),
    'out': tensorflow.Variable(tensorflow.random_normal([n_middle_units, n_classes]))

}
biases = {
    'in': tensorflow.Variable(tensorflow.constant(0.1, shape=[n_hidden_units, ])),
    'mid': tensorflow.Variable(tensorflow.constant(0.1, shape=[n_middle_units, ])),
    'out': tensorflow.Variable(tensorflow.constant(0.1, shape=[n_classes, ]))
}
tensorflow.summary.histogram('RNN/in/weights', weights['in'])
tensorflow.summary.histogram('RNN/in/biases', biases['in'])
tensorflow.summary.histogram('RNN/out/weights', weights['out'])
tensorflow.summary.histogram('RNN/out/biases', biases['out'])

x = tensorflow.placeholder(tensorflow.float32, [None, 300])
x_reshape = tensorflow.reshape(x, [-1, n_inputs])
y = tensorflow.placeholder(tensorflow.float32, [None, n_classes])


def rnn(inputs):
    """进行rnn计算"""

    with tensorflow.name_scope('RNN'):
        x_in = tensorflow.matmul(inputs, weights['in']) + biases['in']
        x_in = tensorflow.reshape(x_in, [-1, n_steps, n_hidden_units])

        lstm_cell = tensorflow.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # 初始化全零 state
        init_state = lstm_cell.zero_state(train_batch_size, dtype=tensorflow.float32)

        outputs, final_state = tensorflow.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    return final_state[1]


fin_state = rnn(x_reshape)

mid_layer = tensorflow.nn.relu(tensorflow.matmul(fin_state, weights['mid']) + biases['mid'])
keep_prob = tensorflow.placeholder(tensorflow.float32, name='keep_prop')
drop_layer = tensorflow.nn.dropout(mid_layer, keep_prob, name='dropout')

pre = tensorflow.matmul(drop_layer, weights['out']) + biases['out']

result = tensorflow.argmax(pre, axis=1, name='result')

cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y, logits=pre))
tensorflow.summary.scalar('cost', cost)
train_op = tensorflow.train.AdamOptimizer(learning_rate=lr).minimize(cost)

correct_pre = tensorflow.equal(result, tensorflow.argmax(y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_pre, tensorflow.float32))
tensorflow.summary.scalar('accuracy', accuracy)

init = tensorflow.global_variables_initializer()
merged = tensorflow.summary.merge_all()

with tensorflow.Session() as sess:
    train_writer_train = tensorflow.summary.FileWriter(train_save_path + 'logs\\' + TIMESTAMP + '\\train',
                                                       graph=sess.graph)
    train_writer_test = tensorflow.summary.FileWriter(train_save_path + 'logs\\' + TIMESTAMP + '\\test',
                                                      graph=sess.graph)
    sess.run(tensorflow.global_variables_initializer())

    # 提前退出标记
    early_stopping_count = 10
    last_test_accuracy = 0.0

    for i in range(repeat_time):
        start = random.randint(0, len(train_images) - train_batch_size)
        _, summary_train = sess.run([train_op, merged], feed_dict={x: train_images[start:start + train_batch_size],
                                                                   y: train_labels[start:start + train_batch_size],
                                                                   keep_prob: 0.5})

        train_writer_train.add_summary(summary_train, i)

        test_start = random.randint(0, len(test_images) - test_batch_size)
        test_accuracy, test_cost, summary_train = sess.run([accuracy, cost, merged],
                                                           feed_dict={
                                                               x: test_images[test_start:test_start + test_batch_size],
                                                               y: test_labels[test_start:test_start + test_batch_size],
                                                               keep_prob: 1.0})
        train_writer_test.add_summary(summary_train, i)

        print('At {}, the accuracy is {:.2f}, the cost is {:.2f}'.format(i, test_accuracy, test_cost))
        # 提前退出判断，防止过拟合
        if test_accuracy > last_test_accuracy:
            early_stopping_count = 10
        else:
            early_stopping_count -= 1
            if early_stopping_count <= 0:
                print('At {}, there is an early stopping.'.format(i))
                break
        last_test_accuracy = test_accuracy

    train_writer_test.close()
    train_writer_train.close()
