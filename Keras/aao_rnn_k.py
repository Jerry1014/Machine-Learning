import numpy
import keras
from keras.layers import SimpleRNN, Activation, Dense

train_save_path = r'C:\Users\Jerry\PycharmProjects\Machine-Learning\train\\'  # 训练后的日志模型保存目录
with numpy.load(train_save_path + 'train.npz') as data:
    train_images = data['train']
    train_labels = data['train_labels']

model = keras.models.Sequential()

model.add(SimpleRNN(
    batch_input_shape=(None,)
))