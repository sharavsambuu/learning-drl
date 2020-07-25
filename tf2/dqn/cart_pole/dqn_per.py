import os
import random
import math
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

tf.get_logger().setLevel('ERROR')

debug_render      = False
num_episodes      = 100000
train_start_count = 1000       # хичнээн sample цуглуулсны дараа сургаж болох вэ
train_per_step    = 1          # хэдэн алхам тутамд сургах вэ
save_per_step     = 2500       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
sync_per_step     = 100        # хэдэн алхам тутам target_q неорон сүлжээг шинэчлэх вэ
train_count       = 1          # хэдэн удаа сургах вэ
batch_size        = 64
gamma             = 0.99       # discount factor

# exploration vs exploitation
epsilon           = 1.0
epsilon_decay     = 0.001
epsilon_max       = 1
epsilon_min       = 0.01

# replay memory
memory_length     = 100000


class SumTree:
  write = 0
  def __init__(self, capacity):
    self.capacity = capacity
    self.tree     = np.zeros(2*capacity - 1)
    self.data     = np.zeros(capacity, dtype=object)
  def _propagate(self, idx, change):
    parent             = (idx - 1) // 2
    self.tree[parent] += change
    if parent != 0:
      self._propagate(parent, change)
  def _retrieve(self, idx, s):
    left  = 2 * idx + 1
    right = left + 1
    if left >= len(self.tree):
      return idx
    if s <= self.tree[left]:
      return self._retrieve(left, s)
    else:
      return self._retrieve(right, s-self.tree[left])
  def total(self):
    return self.tree[0]
  def add(self, p, data):
    idx                   = self.write + self.capacity - 1
    self.data[self.write] = data
    self.update(idx, p)
    self.write           += 1
    if self.write >= self.capacity:
      self.write = 0
  def update(self, idx, p):
    change         = p - self.tree[idx]
    self.tree[idx] = p
    self._propagate(idx, change)
  def get(self, s):
    idx     = self._retrieve(0, s)
    dataIdx = idx - self.capacity + 1
    return (idx, self.tree[idx], self.data[dataIdx])


class PERMemory:
  e = 0.01
  a = 0.6
  def __init__(self, capacity):
    self.tree = SumTree(capacity)
  def _get_priority(self, error):
    return (error+self.e)**self.a
  def add(self, error, sample):
    p = self._get_priority(error)
    self.tree.add(p, sample) 
  def sample(self, n):
    batch   = []
    segment = self.tree.total()/n
    for i in range(n):
      a = segment*i
      b = segment*(i+1)
      s = random.uniform(a, b)
      (idx, p, data) = self.tree.get(s)
      batch.append((idx, data))
    return batch
  def update(self, idx, error):
    p = self._get_priority(error)
    self.tree.update(idx, p)


class DeepQNetwork(tf.keras.Model):
  def __init__(self, n_actions):
    super(DeepQNetwork, self).__init__()
    self.dense_layer  = tf.keras.layers.Dense(128, activation='relu')
    self.mid_layer    = tf.keras.layers.Dense(128, activation='relu') 
    self.output_layer = tf.keras.layers.Dense(n_actions)
  def call(self, inputs):
    dense_out   = self.dense_layer(inputs)
    mid_out     = self.mid_layer(dense_out) 
    return self.output_layer(mid_out)


env = gym.make('CartPole-v0')
env.reset()
n_actions        = env.action_space.n

optimizer        = tf.keras.optimizers.Adam(learning_rate=0.001)

q_network        = DeepQNetwork(n_actions)
target_q_network = DeepQNetwork(n_actions)

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/dqn_per_q'):
  q_network        = tf.keras.models.load_model("model_weights/dqn_per_q")
  target_q_network = tf.keras.models.load_model("model_weights/dqn_per_q_target")
  print("өмнөх сургасан dqn моделийг ачааллаа")


per_memory = PERMemory(memory_length)


global_steps = 0
for episode in range(num_episodes):
  #print(episode, "р ажиллагаа эхэллээ")
  done        = False
  state       = env.reset()
  state_shape = state.shape

  if debug_render:
    env.render()

  episode_rewards = []

  while not done:
    global_steps = global_steps+1

    # exploration vs exploitation
    if np.random.rand() <= epsilon:
      action  = env.action_space.sample()
    else:
      q_value = q_network(np.array([state], dtype=np.float32))
      action  = np.argmax(q_value[0])

    new_state, reward, done, _ = env.step(action)

    episode_rewards.append(reward)

    # TD error-г тооцоолох, энэ алдааны утгаар sample-д priority утга өгнө
    # алдааны утга нь их байх тусмаа сургах batch дээр гарч ирэх магадлал нь ихэснэ
    #if epsilon == 1:
    #  done = True
    q_out        = q_network(np.array([state], dtype=np.float32)).numpy()
    old_value    = q_out[0][action]
    target_q_out = target_q_network(np.array([new_state], dtype=np.float32)).numpy()
    if done:
      q_out[0][action] = reward
    else:
      q_out[0][action] = reward + gamma*np.amax(target_q_out[0])
    td_error = abs(old_value-q_out[0][action])
    per_memory.add(td_error, (state, action, reward, new_state, done))

    # explore хийх epsilon утга шинэчлэх
    if epsilon>epsilon_min:
      epsilon = epsilon_min + (epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
      #print(epsilon)

    # хангалттай sample цугларсан тул Q неорон сүлжээг сургах
    if (global_steps%train_per_step==0):
      #print("Q сүлжээг сургаж байна түр хүлээгээрэй")
      for train_step in range(train_count):
        # цугларсан жишээнүүдээсээ эхлээд batch sample-дэж үүсгэх
        sampled_batch  = per_memory.sample(batch_size)
        print(sampled_batch[0])
        state_shape    = sampled_batch[0][1][0].shape

        q_input        = np.zeros((batch_size, state_shape[0]), dtype=np.float32)
        target_q_input = np.zeros((batch_size, state_shape[0]), dtype=np.float32)
        actions        = []
        rewards        = []
        dones          = []

        td_errors      = np.zeros(batch_size)

        for i in range(batch_size):
          q_input       [i] = sampled_batch[i][1][0] # curr_state
          target_q_input[i] = sampled_batch[i][1][3] # next_state
          actions.append(sampled_batch[i][1][1])     # action
          rewards.append(sampled_batch[i][1][2])     # reward
          dones  .append(sampled_batch[i][1][4])     # is done

        q_out        = q_network(q_input).numpy()
        target_q_out = target_q_network(target_q_input).numpy()

        # bellman q утгыг дөхүүлэхийн тулд сургах batch шинэчлэн тохируулах
        for i in range(batch_size):
          old_value = q_out[i][actions[i]]
          if dones[i]:
            q_out[i][actions[i]] = rewards[i]
          else:
            q_out[i][actions[i]] = rewards[i] + gamma*np.amax(target_q_out[i])
          # шинэ batch-аас TD error-г тооцох
          td_errors[i] = abs(old_value - q_out[i][actions[i]])

        # PER санах ой дээрхи td_error-уудыг шинэчлэх
        # дараа дахин sample-дэхэд хэрэгтэй
        for i in range(batch_size):
          idx = sampled_batch[i][0]
          per_memory.update(idx, td_errors[i])

        # Q неорон сүлжээг сургах
        with tf.GradientTape() as tape:
          prediction_q_out = q_network(q_input)
          loss             = tf.keras.losses.Huber()(q_out, prediction_q_out)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

      training_happened = True
      #print("Q сүлжээг сургаж дууслаа")

    # target q неорон сүлжээг шинэчлэх цаг боллоо
    if global_steps%sync_per_step==0 and training_happened==True:
      target_q_network.set_weights(q_network.get_weights())
      #print("target неорон сүлжээнийг жингүүдийг шинэчиллээ")

    if global_steps%save_per_step==0 and training_happened==True:
      q_network.save("model_weights/dqn_per_q")
      target_q_network.save("model_weights/dqn_per_q_target")
      #print("моделийг model_weights/ фолдерт хадгаллаа")

    if done==True:
      #print(episode, "р ажиллагаа дууслаа")
      #print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
      #print("дундаж reward :", sum(episode_rewards)/len(episode_rewards))
      pass

env.close()
