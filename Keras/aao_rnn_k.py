import numpy
import keras
from keras.layers import SimpleRNN, Dense, LSTMCell

my_batch_size = 10
train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'  # 训练后的日志模型保存目录
with numpy.load(train_save_path + 'train.npz') as data:
    dropped_length = len(data['train_labels']) % my_batch_size
    train_images = data['train'][:-dropped_length]
    train_labels = data['train_labels'][:-dropped_length]

model = keras.models.Sequential()

# model.add(Reshape(target_shape=(20, 15,), input_shape=(300,)))
model.add(SimpleRNN(
    batch_input_shape=(my_batch_size, 20, 15),
    units=512,
    use_bias=True,
))
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dense(9, activation='softmax', use_bias=True))

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(numpy.reshape(train_images, newshape=(-1, 20, 15)), numpy.reshape(train_labels, newshape=(-1, 9)),
          batch_size=my_batch_size,
          epochs=1,
          shuffle=False)
