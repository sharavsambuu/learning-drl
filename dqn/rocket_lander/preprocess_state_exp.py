from time import sleep
import numpy as np
import gym
import cv2
import rocket_lander_gym
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

num_steps     = 300
desired_shape = (140, 180)

env = gym.make('RocketLander-v0')
env.reset()

print("state shape :", env.observation_space.shape[0])
print("action size :", env.action_space.shape[0])
# https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server/48237220#48237220
state = env.render(mode='rgb_array')
print(state.shape)

def process_frame(frame, shape=(84, 84)):
	frame = frame.astype(np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
	frame_reshaped = frame.reshape((*shape, 1))
	return frame, frame_reshaped


gray_state, _ = process_frame(state, shape=desired_shape)
img = plt.imshow(gray_state, cmap='gray', vmin=0, vmax=255)
plt.colorbar(img, orientation='horizontal')
plt.show(block=False)


for _ in range(num_steps):
	action        = env.action_space.sample()
	new_state, reward, done, _ = env.step(action)
	state         = env.render(mode='rgb_array')
	gray_state, _ = process_frame(state, shape=desired_shape)

	# https://www.scivision.dev/fast-update-matplotlib-plots/
	img.set_data(gray_state)
	plt.draw()
	plt.pause(1e-5)

	#if done==True:
	#	break

plt.close("all")
env.close()
