import tensorflow as tf
import numpy as np

print(tf.__version__)

shape = (140, 220, 4)
d1    = np.random.rand(shape[0], shape[1], shape[2])
d2    = np.random.rand(shape[0], shape[1], shape[2])
print(d1.shape)
print(d2.shape)

batch = np.array([d1, d2])
print(batch.shape)

conv_layer = tf.keras.layers.Conv2D(32, (8,8), strides=4, activation='relu')

conv_out   = conv_layer(batch)
print(conv_out.shape)

