import os
import random
from time import sleep
from collections import deque
import numpy as np
import gym
import cv2
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf


debug_render      = False
num_episodes      = 5
train_start_count = 1000       # хичнээн sample цуглуулсны дараа сургаж болох вэ
save_per_step     = 2500       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
train_count       = 1          # хэдэн удаа сургах вэ
desired_shape     = (260, 120) # фрэймыг багасгаж ашиглах хэмжээ
gamma             = 0.99       # discount factor

# temporal state
temporal_length   = 4          # хичнээн фрэймүүд цуглуулж нэг state болгох вэ
temporal_frames   = deque(maxlen=temporal_length)



def preprocess_frame(frame, shape=(84, 84)):
  frame = frame.astype(np.uint8)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
  frame_reshaped = frame.reshape((*shape, 1))
  return frame, frame_reshaped


class PolicyGradient(tf.keras.Model):
  def __init__(self, n_actions):
    super(PolicyGradient, self).__init__()
    self.conv_layer1    = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
    self.conv_layer2    = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
    self.conv_layer3    = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
    self.flatten_layer  = tf.keras.layers.Flatten()
    self.dense_layer    = tf.keras.layers.Dense(512, activation='relu')
    self.output_layer   = tf.keras.layers.Dense(n_actions, activation='softmax')
  def call(self, inputs):
    conv_out1    = self.conv_layer1(inputs)
    conv_out2    = self.conv_layer2(conv_out1)
    conv_out3    = self.conv_layer3(conv_out2)
    flatten_out  = self.flatten_layer(conv_out3)
    dense_out    = self.dense_layer(flatten_out)
    return self.output_layer(dense_out)


env = gym.make('LunarLander-v2')
env.reset()

n_actions        = env.action_space.n

optimizer        = tf.keras.optimizers.Adam()

policy_gradient  = PolicyGradient(n_actions)


if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/PolicyGradient'):
  q_network = tf.keras.models.load_model("model_weights/PolicyGradient")
  print("өмнөх сургасан PolicyGradient моделийг ачааллаа")


global_steps = 0
for episode in range(num_episodes):
  env.reset()
  print(episode, "р ажиллагаа эхэллээ")
  done     = False
  state    = env.render(mode='rgb_array')
  state, _ = preprocess_frame(state, shape=desired_shape)

  if debug_render:
    img = plt.imshow(state, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)


  states, rewards, actions  = [], [], []
  
  while not done:
    global_steps = global_steps+1

    # stochastic action sampling
    if (len(temporal_frames)==temporal_length):
      state  = list(temporal_frames)[0:]
      state  = np.stack(state, axis=-1)
      state  = np.reshape(state, (state.shape[ 0], state.shape[ 1], state.shape[-1]))
      logits = policy_gradient(np.array([state], dtype=np.float32))[0].numpy()
      logits /= logits.sum()
      action = np.random.choice(n_actions, 1, p=logits)[0]
    else:
      action =  env.action_space.sample()

    _, reward, done, _ = env.step(action)

    # sample цуглуулах
    if len(temporal_frames)==temporal_length:
      states.append(state)
      actions.append(action)
      rewards.append(reward)  

    new_state                     = env.render(mode='rgb_array')
    new_state, new_state_reshaped = preprocess_frame(new_state, shape=desired_shape)
    temporal_frames.append(new_state_reshaped)    

    if debug_render:
      img.set_data(new_state)
      plt.draw()
      plt.pause(1e-5)

    if global_steps%save_per_step==0 and training_happened==True:
      q_network.save("model_weights/PolicyGradient")
      print("моделийг model_weights/PolicyGradient фолдерт хадгаллаа")

    if done==True:
      if debug_render:
        plt.clf()

      episode_length     = len(states)

      # discount factor-г reward жагсаалтруу оруулж ирэх
      discounted_rewards = np.zeros_like(rewards)
      running_add = 0
      for t in reversed(range(0, episode_length)):
        running_add = running_add*gamma + rewards[t]
        discounted_rewards[t] = running_add
      discounted_rewards -= np.mean(discounted_rewards)
      discounted_rewards /= np.std(discounted_rewards)
      
      inputs     = np.zeros((episode_length, desired_shape[0], desired_shape[1], temporal_length), dtype=np.float32)
      advantages = np.zeros((episode_length, n_actions), dtype=np.float32)
      for i in range(episode_length):
        inputs[i]                 = states[i]
        advantages[i][actions[i]] = discounted_rewards[i]

      # Policy неорон сүлжээг сургах
      with tf.GradientTape() as tape:
        predictions = policy_gradient(inputs)
        loss        = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(advantages, predictions)
      gradients = tape.gradient(loss, policy_gradient.trainable_variables)
      optimizer.apply_gradients(zip(gradients, policy_gradient.trainable_variables))
      print("%s урттай эпизодыг неорон сүлжээнд сургалаа."%(episode_length))
            
      training_happened = True
      states, rewards, actions  = [], [], []
      print(episode, "р ажиллагаа дууслаа")


if debug_render:
  plt.close("all")
env.close()