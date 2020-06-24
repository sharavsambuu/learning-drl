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


debug_render  = True

num_episodes  = 2
global_steps  = 0

desired_shape   = (140, 180)
temporal_frames = deque(maxlen=4)

memory_length = 2000
replay_memory  = deque(maxlen=memory_length)

def process_frame(frame, shape=(84, 84)):
	frame = frame.astype(np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
	frame_reshaped = frame.reshape((*shape, 1))
	return frame, frame_reshaped


env = gym.make('RocketLander-v0')
env.reset()


for episode in range(num_episodes):
	print(episode, "is running...")
	done = False
	env.reset()
	# https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server/48237220#48237220
	state         = env.render(mode='rgb_array')
	gray_state, _ = process_frame(state, shape=desired_shape)
	img           = plt.imshow(gray_state, cmap='gray', vmin=0, vmax=255)
	plt.colorbar(img, orientation='horizontal')
	plt.show(block=False)

	while not done:
		action        = env.action_space.sample()
		new_state, reward, done, _ = env.step(action)
		state         = env.render(mode='rgb_array')
		gray_state, _ = process_frame(state, shape=desired_shape)
		# https://www.scivision.dev/fast-update-matplotlib-plots/
		img.set_data(gray_state)
		plt.draw()
		plt.pause(1e-5)

		if done==True:
			plt.clf()
			print(episode, "is done.")

plt.close("all")
env.close()
