from time import sleep
import numpy as np
import gym
import cv2
import rocket_lander_gym
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

env      = gym.make('RocketLander-v0')
env.reset()

sleep(1)
print("state shape :", env.observation_space.shape[0])
print("action size :", env.action_space.shape[0])
# https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server/48237220#48237220
state = env.render(mode='rgb_array')
print(state.shape)

# rendering original numpy array (700, 504, 3)
img = plt.imshow(state)
plt.colorbar(img, orientation='horizontal')
plt.show(block=False)
plt.pause(2)
plt.close()


def process_frame(frame, shape=(84, 84)):
	frame = frame.astype(np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
	frame_reshaped = frame.reshape((*shape, 1))
	return frame, frame_reshaped

gray_state, _ = process_frame(state, shape=(state.shape[1], state.shape[0]))
print(gray_state.shape)
img = plt.imshow(gray_state, cmap='gray', vmin=0, vmax=255)
plt.colorbar(img, orientation='horizontal')
plt.show(block=False)
#plt.pause(3)
#plt.close()

for _ in range(1000):
	action        = env.action_space.sample()
	new_state, reward, done, _ = env.step(action)
	state         = env.render(mode='rgb_array')
	gray_state, _ = process_frame(state, shape=(state.shape[1], state.shape[0]))
	img.set_data(gray_state)
	
	plt.colorbar(img, orientation='horizontal')
	plt.show(block=False)
	#plt.pause(3)
plt.close("all")



env.close()
