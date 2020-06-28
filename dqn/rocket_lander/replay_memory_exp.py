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


debug_render    = True

num_episodes    = 2
global_steps    = 0

desired_shape   = (140, 220)
temporal_length = 4
temporal_frames = deque(maxlen=temporal_length+1)

memory_length   = 200
replay_memory   = deque(maxlen=memory_length)

def preprocess_frame(frame, shape=(84, 84)):
  frame = frame.astype(np.uint8)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
  frame_reshaped = frame.reshape((*shape, 1))
  return frame, frame_reshaped


env = gym.make('RocketLander-v0')
env.reset()


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
      print(prev_state.shape, next_state.shape)
      
      replay_memory.append((prev_state, action, reward, next_state, done))
      pass

    print("replay memory length", len(replay_memory))

    if done==True:
      plt.clf()
      print(episode, "is done.")

plt.close("all")
env.close()
