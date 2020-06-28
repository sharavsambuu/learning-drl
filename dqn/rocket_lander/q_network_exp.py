import random
from time import sleep
from collections import deque
import numpy as np
import gym
import cv2
import rocket_lander_gym
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf


debug_render      = True
global_steps      = 0
num_episodes      = 2
train_start_count = 100       # хичнээн sample цуглуулсны дараа сургаж болох вэ
batch_size        = 64
desired_shape     = (140, 220) # фрэймыг багасгаж ашиглах хэмжээ
temporal_length   = 4
temporal_frames   = deque(maxlen=temporal_length+1)
memory_length     = 200
replay_memory     = deque(maxlen=memory_length)

def preprocess_frame(frame, shape=(84, 84)):
	frame = frame.astype(np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
	frame_reshaped = frame.reshape((*shape, 1))
	return frame, frame_reshaped


class DeepQNetwork(tf.keras.Model):
	def __init__(self, n_actions):
		super(DeepQNetwork, self).__init__()
		self.conv_layer1    = tf.keras.layers.Conv2D(32,   (8,8), strides=4, activation='relu')
		self.maxpool_layer1 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
		self.conv_layer2    = tf.keras.layers.Conv2D(64,   (4,4), strides=1, activation='relu')
		self.maxpool_layer2 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
		self.conv_layer3    = tf.keras.layers.Conv2D(1024, (7,7), strides=1, activation='relu')
		self.flatten_layer  = tf.keras.layers.Flatten()
		self.output_layer   = tf.keras.layers.Dense(n_actions, activation='softmax')
	def call(self, inputs):
		conv_out1    = self.conv_layer1(inputs)
		maxpool_out1 = self.maxpool_layer1(conv_out1)
		conv_out2    = self.conv_layer2(maxpool_out1)
		maxpool_out2 = self.maxpool_layer2(conv_out2)
		conv_out3    = self.conv_layer3(maxpool_out2)
		flatten_out  = self.flatten_layer(conv_out3)
		return self.output_layer(flatten_out)


env = gym.make('RocketLander-v0')
env.reset()
n_actions = env.action_space.n


q_network        = DeepQNetwork(n_actions)
target_q_network = DeepQNetwork(n_actions)



for episode in range(num_episodes):
	env.reset()
	print(episode, "is running...")
	done     = False
	state    = env.render(mode='rgb_array')
	state, _ = preprocess_frame(state, shape=desired_shape)

	img               = plt.imshow(state, cmap='gray', vmin=0, vmax=255)
	plt.colorbar(img, orientation='horizontal')
	plt.show(block=False)

	while not done:
		state    = env.render(mode='rgb_array')
		state, _ = preprocess_frame(state, shape=desired_shape)

		action             = env.action_space.sample()
		_, reward, done, _ = env.step(action)

		new_state                     = env.render(mode='rgb_array')
		new_state, new_state_reshaped = preprocess_frame(new_state, shape=desired_shape)

		img.set_data(new_state)
		plt.draw()
		plt.pause(1e-5)

		# sample цуглуулах
		temporal_frames.append(new_state_reshaped)
		if len(temporal_frames)==5:
			prev_state = list(temporal_frames)[:temporal_length]
			next_state = list(temporal_frames)[1:]

			# Дараалсан фрэймүүдийг нэгтгээд нэг тензор болгох. 
			# Неорон модельрүү чихэхэд амар 
			# Тензорын дүрс нь (өндөр, өргөн, фрэймийн тоо)
			prev_state = np.stack(prev_state, axis=-1)
			prev_state = np.reshape(
					prev_state, 
					(
						prev_state.shape[ 0], 
						prev_state.shape[ 1], 
						prev_state.shape[-1]
					)
				)
			next_state = np.stack(next_state, axis=-1)
			next_state = np.reshape(
					next_state,
					(
						next_state.shape[ 0],
						next_state.shape[ 1],
						next_state.shape[-1]
					)
				)
			replay_memory.append((prev_state, action, reward, next_state, done))
			pass

		if (len(replay_memory)>train_start_count):
			sampled_batch = random.sample(replay_memory, batch_size)
			q_out = q_network(np.array([sampled_batch[0][0]], dtype=np.float32))
			print(q_out.shape)
			#print(sampled_batch[0][0].shape)
			pass

		if done==True:
			plt.clf()
			print(episode, "is done.")

plt.close("all")
env.close()
