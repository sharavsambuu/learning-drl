import os
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
num_episodes      = 20000
train_start_count = 500        # хичнээн sample цуглуулсны дараа сургаж болох вэ
train_per_step    = 100        # хэдэн алхам тутамд сургах вэ
save_per_step     = 1000       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
sync_per_step     = 600        # хэдэн алхам тутам target_q неорон сүлжээг шинэчлэх вэ
train_count       = 1          # хэдэн удаа сургах вэ
batch_size        = 64
desired_shape     = (84, 84)    # фрэймыг багасгаж ашиглах хэмжээ
gamma             = 0.85       # discount factor

# exploration vs exploitation
epsilon           = 1.0        
epsilon_decay     = 0.999
epsilon_min       = 0.13

# replay memory
temporal_length   = 4          # хичнээн фрэймүүд цуглуулж нэг state болгох вэ
temporal_frames   = deque(maxlen=temporal_length+1)
memory_length     = 1000
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
    self.conv_layer1    = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
    self.conv_layer2    = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
    self.conv_layer3    = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
    self.flatten_layer  = tf.keras.layers.Flatten()
    self.dense_layer    = tf.keras.layers.Dense(512, activation='relu')
    self.output_layer   = tf.keras.layers.Dense(n_actions, activation='linear')
  def call(self, inputs):
    conv_out1    = self.conv_layer1(inputs)
    conv_out2    = self.conv_layer2(conv_out1)
    conv_out3    = self.conv_layer3(conv_out2)
    flatten_out  = self.flatten_layer(conv_out3)
    dense_out    = self.dense_layer(flatten_out)
    return self.output_layer(dense_out)


env = gym.make('Breakout-v0')
env.reset()
n_actions        = env.action_space.n

optimizer        = tf.keras.optimizers.Adam()

q_network        = DeepQNetwork(n_actions)

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/dqn'):
  q_network = tf.keras.models.load_model("model_weights/dqn")
  print("өмнөх сургасан dqn моделийг ачааллаа")


for episode in range(num_episodes):
  env.reset()
  print(episode, "р ажиллагаа эхэллээ")
  done     = False
  state    = env.render(mode='rgb_array')
  state, _ = preprocess_frame(state, shape=desired_shape)

  if debug_render:
    img = plt.imshow(state, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)

  episode_rewards = [] 

  while not done:
    global_steps = global_steps+1

    if (len(temporal_frames)==temporal_length+1):
      state = list(temporal_frames)[1:]
      state = np.stack(state, axis=-1)
      state = np.reshape(
        state, 
        (
          state.shape[ 0], 
          state.shape[ 1], 
          state.shape[-1]
        )
      )
      q_value = q_network(np.array([state], dtype=np.float32))
      action  = np.argmax(q_value[0])
    else:
      action =  env.action_space.sample()

    _, reward, done, _ = env.step(action)

    episode_rewards.append(reward)

    new_state                     = env.render(mode='rgb_array')
    new_state, new_state_reshaped = preprocess_frame(new_state, shape=desired_shape)

    if debug_render:
      img.set_data(new_state)
      plt.draw()
      plt.pause(1e-5)

    if done==True:
      if debug_render:
        plt.clf()
      print(episode, "р ажиллагаа дууслаа")
      print("нийт reward   :", sum(episode_rewards))
      print("дундаж reward :", sum(episode_rewards)/len(episode_rewards))

if debug_render:
  plt.close("all")
env.close()
