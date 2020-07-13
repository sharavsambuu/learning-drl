import os, sys
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

tf.get_logger().setLevel('ERROR')

debug_render  = True
num_episodes  = 20
save_per_step = 1000  # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
gamma         = 0.95  # discount factor


class PolicyNetwork(tf.keras.Model):
  def __init__(self, n_actions):
    super(PolicyNetwork, self).__init__()
    self.dense_layer   = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')
    self.dropout_layer = tf.keras.layers.Dropout(0.1)
    self.output_layer  = tf.keras.layers.Dense(n_actions, activation='softmax', kernel_initializer='glorot_uniform')
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs):
    dense_out   = self.dense_layer(inputs)
    dropout_out = self.dropout_layer(dense_out)
    return self.output_layer(dropout_out)


env       = gym.make('CartPole-v0')
env.reset()
n_actions = env.action_space.n


policy  = PolicyNetwork(n_actions)


if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/ReinforcePolicy'):
  policy = tf.keras.models.load_model("model_weights/ReinforcePolicy")
  print("өмнөх сургасан моделийг ачааллаа")
else:
  sys.exit(0)


global_steps    = 0
rewards_history = []

for episode in range(num_episodes):
  done  = False
  score = 0
  states, actions, rewards = [], [], []
  state = env.reset()
  
  while not done:
    logits        = policy(np.array([state], dtype=np.float32), training=False) # πθ(a|s)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    action        = np.random.choice(n_actions, p=probabilities)
    new_state, reward, done, _ = env.step(action)
    score = score+reward
    state = new_state

    if debug_render:
      env.render()

    if done==True:
      print("score : ", score)
      break


if debug_render:
  plt.close("all")
env.close()
