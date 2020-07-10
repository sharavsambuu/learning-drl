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


debug_render      = True
num_episodes      = 5000
train_start_count = 1000       # хичнээн sample цуглуулсны дараа сургаж болох вэ
save_per_step     = 2500       # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
training_happened = False
train_count       = 1          # хэдэн удаа сургах вэ
desired_shape     = (84, 84) # фрэймыг багасгаж ашиглах хэмжээ
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


class PolicyNetwork(tf.keras.Model):
  def __init__(self, n_actions):
    super(PolicyNetwork, self).__init__()
    self.conv_layer1    = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
    self.conv_layer2    = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
    self.conv_layer3    = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
    self.flatten_layer  = tf.keras.layers.Flatten()
    self.dense_layer    = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')
    self.output_layer   = tf.keras.layers.Dense(n_actions, activation='softmax', kernel_initializer='glorot_uniform')
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs):
    conv_out1    = self.conv_layer1(inputs)
    conv_out2    = self.conv_layer2(conv_out1)
    conv_out3    = self.conv_layer3(conv_out2)
    flatten_out  = self.flatten_layer(conv_out3)
    dense_out    = self.dense_layer(flatten_out)
    return self.output_layer(dense_out)

 
optimizer   = tf.keras.optimizers.Adam()

env       = gym.make('Pong-v0')
env.reset()
n_actions = env.action_space.n


policy  = PolicyNetwork(n_actions)


@tf.function(experimental_relax_shapes=True)
def loss_fn(action_logits, actions, targets):
  # [actions] -> [(idx, action_index)...]
  actions                     = tf.convert_to_tensor(
    list(zip(np.arange(len(actions)), actions))
    )
  action_probabilities        = tf.nn.softmax(action_logits)
  # Үйлдлийн индексд харгалзах магадлалын оноог авах
  picked_action_probabilities = tf.gather_nd(action_probabilities, actions) 
  # logπθ(a|s)
  log_probabilites            = tf.cast(tf.math.log(picked_action_probabilities), dtype=tf.float64)
  # logπθ(a|s)*G_t, үйлдлийн магадлалыг discount авсан reward-аар үржих
  loss                        = tf.multiply(log_probabilites, tf.convert_to_tensor(targets))
  # gradient ascent
  return -tf.reduce_sum(loss)


@tf.function(experimental_relax_shapes=True)
def train_policy_network(inputs, actions, advantages):
  with tf.GradientTape() as tape:
    predictions = policy(inputs, training=True)
    loss        = loss_fn(predictions, actions, advantages)
  if debug_render:
    tf.print("loss : ", loss)
  gradients = tape.gradient(loss, policy.trainable_variables)
  optimizer.apply_gradients(zip(gradients, policy.trainable_variables))
  

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/ReinforceDiscounted'):
  policy = tf.keras.models.load_model("model_weights/ReinforceDiscounted")
  print("өмнөх сургасан ReinforceDiscounted моделийг ачааллаа")


global_steps = 0
for episode in range(num_episodes):
  env.reset()
  done     = False
  state    = env.render(mode='rgb_array')
  state, _ = preprocess_frame(state, shape=desired_shape)

  if debug_render:
    img = plt.imshow(state, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)


  score = 0
  states, rewards, actions  = [], [], []
  
  while not done:
    global_steps = global_steps+1

    # stochastic action sampling
    if (len(temporal_frames)==temporal_length):
      inp_state     = list(temporal_frames)[0:]
      inp_state     = np.stack(inp_state, axis=-1)
      inp_state     = np.reshape(inp_state, (inp_state.shape[ 0], inp_state.shape[ 1], inp_state.shape[-1]))
      logits        = policy(np.array([inp_state], dtype=np.float32), training=False)
      probabilities = tf.nn.softmax(logits).numpy()[0]
      action        = np.random.choice(n_actions, p=probabilities)
    else:
      action = env.action_space.sample()

    _, reward, done, _ = env.step(action)

    score = score+reward

    # sample цуглуулах
    if len(temporal_frames)==temporal_length:
      states.append(inp_state)
      actions.append(action)
      rewards.append(reward)

    new_state                     = env.render(mode='rgb_array')
    new_state, new_state_reshaped = preprocess_frame(new_state, shape=desired_shape)
    temporal_frames.append(np.reshape(new_state_reshaped, (desired_shape[0], desired_shape[1], 1)))

    if debug_render:
      #img.set_data(new_state)
      img.set_data(np.reshape(list(temporal_frames)[0], (desired_shape[1], desired_shape[0])))
      plt.draw()
      plt.pause(1e-6)
      pass

    if global_steps%save_per_step==0 and training_happened==True:
      policy.save("model_weights/ReinforceDiscounted")
      print("моделийг model_weights/ReinforceDiscounted фолдерт хадгаллаа")

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

      train_policy_network(inputs, actions, discounted_rewards)
      
      training_happened = True
      states, rewards, actions  = [], [], []
      print("%s : %s урттай ажиллагаа %s оноотой дууслаа"%(episode, episode_length, score))


if debug_render:
  plt.close("all")
env.close()
