import tensorflow as tf
import numpy as np

print(tf.__version__)

shape = (140, 220, 4)
d1    = np.random.rand(shape[0], shape[1], shape[2])
d2    = np.random.rand(shape[0], shape[1], shape[2])
print(d1.shape)
print(d2.shape)

# (2, 140, 220, 4)
batch = np.array([d1, d2])
print(batch.shape)

# эхний convolutional давхарга
# (2, 140, 220, 4) -> (2, 34, 54, 32)
conv_layer1 = tf.keras.layers.Conv2D(32, (8,8), strides=4, activation='relu')
conv_out1   = conv_layer1(batch)
print("эхний convolutional давхаргын гаралтын дүрс")
print(conv_out1.shape)

# maxpool давхарга
# (2, 34, 54, 32) -> (2, 17, 27, 32)
maxpool_layer1 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
maxpool_out1   = maxpool_layer1(conv_out1)
print("эхний maxpool давхаргын гаралтын дүрс")
print(maxpool_out1.shape)

# хоёр дахь convolutional давхарга
# (2, 17, 27, 32) -> (2, 14, 24, 64)
conv_layer2 = tf.keras.layers.Conv2D(64, (4,4), strides=1, activation='relu')
conv_out2   = conv_layer2(maxpool_out1)
print("хоёр дахь convolutional давхаргын гаралтын дүрс")
print(conv_out2.shape)

# хоёр дахь maxpool давхарга
# (2, 14, 24, 64) -> (2, 7, 12, 64)
maxpool_layer2 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
maxpool_out2   = maxpool_layer2(conv_out2)
print("хоёр дахь maxpool давхаргын гаралт")
print(maxpool_out2.shape)


# гурав дахь convolutional давхарга
# (2, 7, 12, 64) -> (2, 1, 6, 1024)
conv_layer3 = tf.keras.layers.Conv2D(1024, (7,7), strides=1, activation='relu')
conv_out3   = conv_layer3(maxpool_out2)
print("гурав дахь convolutional гаралтын дүрс")
print(conv_out3.shape)

# сүүлийн давхаргаруу бэлтгэх үе шат, бүх неоронуудыг нэг векторлуу жагсаах
# (2, 1, 6, 1024) -> (2, 6144)
flatten_layer = tf.keras.layers.Flatten()
flatten_out   = flatten_layer(conv_out3)
print("flatten давхаргын гаралтын дүрс")
print(flatten_out.shape)

# үйлдлийн төрлийн тоог 3-н ширхэг үзье
n_actions = 3

# гаралтын давхарга
# (2, 6144) -> (2, 3)
output_layer = tf.keras.layers.Dense(n_actions, activation='softmax')
last_output  = output_layer(flatten_out)
print("сүүлийн гаралтын давхаргын дүрс")
print(last_output.shape)
