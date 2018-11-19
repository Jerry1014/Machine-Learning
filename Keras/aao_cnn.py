import keras
import numpy
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout

batch_size = 15
epochs = 1
num_classes = 9
train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'  # 训练后的日志模型保存目录
with numpy.load(train_save_path + 'train.npz') as data:
    train_images = data['train']
    train_labels = data['train_labels']
    test_images = data['train'][5000:]
    test_labels = data['train_labels'][5000:]

model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(20, 15, 1), use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu', use_bias=True))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax', use_bias=True))

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(numpy.reshape(train_images, newshape=[-1, 20, 15, 1]), train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.03)
