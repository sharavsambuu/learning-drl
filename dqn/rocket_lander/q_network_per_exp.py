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
global_steps      = 0
num_episodes      = 2000
train_start_count = 500        # хичнээн sample цуглуулсны дараа сургаж болох вэ
train_per_step    = 100        # хэдэн алхам тутамд сургах вэ
save_per_step     = 1000       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
sync_per_step     = 600        # хэдэн алхам тутам target_q неорон сүлжээг шинэчлэх вэ
train_count       = 1          # хэдэн удаа сургах вэ
batch_size        = 64
desired_shape     = (320, 420) # фрэймыг багасгаж ашиглах хэмжээ
gamma             = 0.85       # discount factor

# exploration vs exploitation
epsilon           = 1.0        
epsilon_decay     = 0.999
epsilon_min       = 0.13

# replay memory
temporal_length   = 4          # хичнээн фрэймүүд цуглуулж нэг state болгох вэ
temporal_frames   = deque(maxlen=temporal_length+1)
memory_length     = 4000 
replay_memory     = deque(maxlen=memory_length)


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
    self.tree     = numpy.zeros( 2*capacity - 1 )
    self.data     = numpy.zeros( capacity, dtype=object )
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


class Memory:
  e = 0.01
  a = 0.6
  def __init__(self, capacity):
    self.tree = SumTree(capacity)
  def _getPriority(self, error):
    return (error + self.e) ** self.a
  def add(self, error, sample):
    p = self._getPriority(error)
    self.tree.add(p, sample) 
  def sample(self, n):
    batch = []
    segment = self.tree.total() / n
    for i in range(n):
      a = segment * i
      b = segment * (i + 1)
      s = random.uniform(a, b)
      (idx, p, data) = self.tree.get(s)
      batch.append( (idx, data) )
    return batch
  def update(self, idx, error):
    p = self._getPriority(error)
    self.tree.update(idx, p)


class DeepQNetwork(tf.keras.Model):
  def __init__(self, n_actions):
    super(DeepQNetwork, self).__init__()
    self.conv_layer1    = tf.keras.layers.Conv2D(32,   (8,8), strides=4, activation='relu')
    self.maxpool_layer1 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
    self.conv_layer2    = tf.keras.layers.Conv2D(64,   (4,4), strides=1, activation='relu')
    self.maxpool_layer2 = tf.keras.layers.MaxPooling2D((2,2), strides=2)
    self.conv_layer3    = tf.keras.layers.Conv2D(1024, (7,7), strides=1, activation='relu')
    self.flatten_layer  = tf.keras.layers.Flatten()
    self.output_layer   = tf.keras.layers.Dense(n_actions)
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
n_actions        = env.action_space.n

optimizer        = tf.keras.optimizers.Adam()

q_network        = DeepQNetwork(n_actions)
target_q_network = DeepQNetwork(n_actions)

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
    #print(global_steps)

    # exploration vs exploitation
    if (len(temporal_frames)==temporal_length+1):
      if np.random.rand() <= epsilon:
        action  = env.action_space.sample()
      else:
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
        #print(q_value)
        action  = np.argmax(q_value[0])
        #print("Q неорон сүлжээ", action, "үйлдлийг сонголоо")
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
      curr_state = np.reshape(
          curr_state, 
          (
            curr_state.shape[ 0], 
            curr_state.shape[ 1], 
            curr_state.shape[-1]
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
      replay_memory.append((curr_state, action, reward, next_state, done))

      # explore хийх epsilon утга шинэчлэх
      if epsilon>epsilon_min:
        epsilon = epsilon*epsilon_decay

      pass

    # хангалттай sample цугларсан тул Q неорон сүлжээнүүдээ сургах
    if (len(replay_memory)>train_start_count) and (global_steps%train_per_step==0):
      print("Q сүлжээг сургаж байна түр хүлээгээрэй")
      for train_step in range(train_count):
        #print(train_step, "р batch")
        # цугларсан жишээнүүдээсээ эхлээд batch sample-дэж үүсгэх
        sampled_batch = random.sample(replay_memory, batch_size)
        state_shape   = sampled_batch[0][0].shape

        q_input        = np.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)
        target_q_input = np.zeros((batch_size, state_shape[0], state_shape[1], state_shape[2]), dtype=np.float32)
        actions        = []
        rewards        = []
        dones          = []

        for i in range(batch_size):
          q_input       [i] = sampled_batch[i][0] # curr_state
          target_q_input[i] = sampled_batch[i][3] # next_state
          actions.append(sampled_batch[i][1])     # action
          rewards.append(sampled_batch[i][2])     # reward
          dones  .append(sampled_batch[i][4])     # is done
      
        q_out        = q_network(q_input).numpy()
        target_q_out = target_q_network(target_q_input).numpy()

        # bellman q утгыг дөхүүлэхийн тулд сургах batch шинэчлэн тохируулах
        for i in range(batch_size):
          if dones[i]:
            q_out[i][actions[i]] = rewards[i]
          else:
            q_out[i][actions[i]] = rewards[i] + gamma*np.amax(target_q_out[i])

        # Q неорон сүлжээг сургах
        with tf.GradientTape() as tape:
          prediction_q_out = q_network(q_input)
          loss             = tf.keras.losses.MeanSquaredError()(q_out, prediction_q_out)/2
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
      training_happened = True
      print("Q сүлжээг сургаж дууслаа")

    # target q неорон сүлжээг шинэчлэх цаг боллоо
    #if global_steps%sync_per_step==0 and training_happened==True:
    #  target_q_network.set_weights(q_network.get_weights())
    #  print("шинэ сурсан мэдлэгээрээ target q неорон сүлжээг шинэчиллээ")
    if global_steps%save_per_step==0 and training_happened==True:
      q_network.save("model_weights/dqn")
      print("моделийг model_weights/dqn фолдерт хадгаллаа")

    if done==True:
      if debug_render:
        plt.clf()
      print(episode, "р ажиллагаа дууслаа")
      print("нийт reward   :", sum(episode_rewards))
      print("дундаж reward :", sum(episode_rewards)/len(episode_rewards))
      if training_happened==True:
        target_q_network.set_weights(q_network.get_weights())
        print("шинэ сурсан мэдлэгээрээ target q неорон сүлжээг шинэчиллээ")

if debug_render:
  plt.close("all")
env.close()
