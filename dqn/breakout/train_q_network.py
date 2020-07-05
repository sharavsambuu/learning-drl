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


debug_render      = False
num_episodes      = 2000
train_start_count = 1000       # хичнээн sample цуглуулсны дараа сургаж болох вэ
train_per_step    = 500        # хэдэн алхам тутамд сургах вэ
save_per_step     = 2500       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
sync_per_step     = 1000       # хэдэн алхам тутам target_q неорон сүлжээг шинэчлэх вэ
train_count       = 2          # хэдэн удаа сургах вэ
batch_size        = 32
desired_shape     = (84, 84) # фрэймыг багасгаж ашиглах хэмжээ
gamma             = 0.99       # discount factor

# exploration vs exploitation
epsilon           = 1.0        
epsilon_decay     = 0.999
epsilon_min       = 0.13

# replay memory
temporal_length   = 4          # хичнээн фрэймүүд цуглуулж нэг state болгох вэ
temporal_frames   = deque(maxlen=temporal_length+1)
memory_length     = 3000 


def preprocess_frame(frame, shape=(84, 84)):
  frame = frame.astype(np.uint8)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
  frame_reshaped = frame.reshape((*shape, 1))
  return frame, frame_reshaped


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


class DuelingDQN(tf.keras.Model):
  def __init__(self, n_actions):
    super(DuelingDQN, self).__init__()
    self.conv_layer1                = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
    self.conv_layer2                = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
    self.conv_layer3                = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
    self.splitting_layer            = tf.keras.layers.Lambda(lambda w: tf.split(w, 2, 3))
    self.value_stream_flattened     = tf.keras.layers.Flatten()
    self.value_stream_dense         = tf.keras.layers.Dense(1)
    self.advantage_stream_flattened = tf.keras.layers.Flatten()
    self.advantage_stream_dense     = tf.keras.layers.Dense(n_actions)
  def call(self, inputs):
    conv_out1        = self.conv_layer1(inputs)
    conv_out2        = self.conv_layer2(conv_out1)
    conv_out3        = self.conv_layer3(conv_out2)
    
    value_stream, advantage_stream = self.splitting_layer(conv_out3)
    
    value_stream     = self.value_stream_flattened(value_stream)
    value            = self.value_stream_dense(value_stream)
    advantage_stream = self.advantage_stream_flattened(advantage_stream)
    advantage        = self.advantage_stream_dense(advantage_stream)

    # value болон advantage хоёр stream-г нэгтгээд гаралтын Q утгууд болгох
    q_values         = tf.keras.layers.Add()([
        value, 
        tf.keras.layers.Subtract()([
          advantage,
          tf.keras.layers.Lambda(
            lambda w: tf.reduce_mean(w, axis=1, keepdims=True)
          )(advantage)
        ])
      ])

    return q_values


env = gym.make('RocketLander-v0')
env.reset()
n_actions        = env.action_space.n

optimizer        = tf.keras.optimizers.Adam()

q_network        = DuelingDQN(n_actions)
target_q_network = DuelingDQN(n_actions)

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/DuelingDQN'):
  q_network = tf.keras.models.load_model("model_weights/DuelingDQN")
  print("өмнөх сургасан dqn моделийг ачааллаа")


per_memory = PERMemory(memory_length)


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

  episode_rewards = [] 

  while not done:
    global_steps = global_steps+1

    # exploration vs exploitation
    if (len(temporal_frames)==temporal_length+1):
      if np.random.rand() <= epsilon:
        action  = env.action_space.sample()
      else:
        state   = list(temporal_frames)[1:]
        state   = np.stack(state, axis=-1)
        state   = np.reshape(state, (state.shape[ 0], state.shape[ 1], state.shape[-1]))
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

    # sample цуглуулах
    temporal_frames.append(new_state_reshaped)
    if len(temporal_frames)==temporal_length+1:
      curr_state = list(temporal_frames)[:temporal_length]
      next_state = list(temporal_frames)[1:]
      # дараалсан фрэймүүдийг нэгтгээд нэг тензор болгох
      # неорон модельрүү чихэхэд амар, тензорын дүрс нь (өндөр, өргөн, фрэймийн тоо)
      curr_state = np.stack(curr_state, axis=-1)
      curr_state = np.reshape(curr_state, (curr_state.shape[ 0], curr_state.shape[ 1], curr_state.shape[-1]))
      next_state = np.stack(next_state, axis=-1)
      next_state = np.reshape(next_state, (next_state.shape[ 0], next_state.shape[ 1], next_state.shape[-1]))

      # TD error-г тооцоолох, энэ алдааны утгаар sample-д priority утга өгнө
      # алдааны утга нь их байх тусмаа сургах batch дээр гарч ирэх магадлал нь ихэснэ
      if epsilon == 1:
        done = True
      q_out        = q_network(np.array([curr_state], dtype=np.float32)).numpy()
      old_value    = q_out[0][action]
      target_q_out = target_q_network(np.array([next_state], dtype=np.float32)).numpy()
      if done:
        q_out[0][action] = reward
      else:
        q_out[0][action] = reward + gamma*np.amax(target_q_out[0])
      td_error = abs(old_value-q_out[0][action])
      #print("TD error", td_error)
      per_memory.add(td_error, (curr_state, action, reward, next_state, done))
      
      # explore хийх epsilon утга шинэчлэх
      if epsilon>epsilon_min:
        epsilon = epsilon*epsilon_decay

    # хангалттай sample цугларсан тул Q неорон сүлжээг сургах
    if (global_steps%train_per_step==0):
      print("Q сүлжээг сургаж байна түр хүлээгээрэй")
      for train_step in range(train_count):
        # цугларсан жишээнүүдээсээ эхлээд batch sample-дэж үүсгэх
        sampled_batch  = per_memory.sample(batch_size)
        state_shape    = sampled_batch[0][1][0].shape

        q_input        = np.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)
        target_q_input = np.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)
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
      print("Q сүлжээг сургаж дууслаа")

    # target q неорон сүлжээг шинэчлэх цаг боллоо
    if global_steps%sync_per_step==0 and training_happened==True:
      target_q_network.set_weights(q_network.get_weights())
      print("target неорон сүлжээнийг жингүүдийг шинэчиллээ")

    if global_steps%save_per_step==0 and training_happened==True:
      q_network.save("model_weights/DuelingDQN")
      print("моделийг model_weights/DuelingDQN фолдерт хадгаллаа")

    if done==True:
      if debug_render:
        plt.clf()
      print(episode, "р ажиллагаа дууслаа")
      print("нийт reward   :", sum(episode_rewards))
      print("дундаж reward :", sum(episode_rewards)/len(episode_rewards))


if debug_render:
  plt.close("all")
env.close()
