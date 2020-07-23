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
desired_shape     = (84, 84)   # фрэймыг багасгаж ашиглах хэмжээ
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

class ValueNetwork(tf.keras.Model):
  def __init__(self,):
    super(ValueNetwork, self).__init__()
    self.conv_layer1    = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
    self.conv_layer2    = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
    self.conv_layer3    = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
    self.flatten_layer  = tf.keras.layers.Flatten()
    self.dense_layer    = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')
    self.output_layer   = tf.keras.layers.Dense(1)
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs):
    conv_out1    = self.conv_layer1(inputs)
    conv_out2    = self.conv_layer2(conv_out1)
    conv_out3    = self.conv_layer3(conv_out2)
    flatten_out  = self.flatten_layer(conv_out3)
    dense_out    = self.dense_layer(flatten_out)
    return self.output_layer(dense_out)



env       = gym.make('Pong-v0')
env.reset()
n_actions = env.action_space.n


policy_optimizer = tf.keras.optimizers.Adam()
value_optimizer  = tf.keras.optimizers.Adam()

policy  = PolicyNetwork(n_actions)
value   = ValueNetwork()

value_loss = tf.keras.losses.Huber()

@tf.function(experimental_relax_shapes=True)
def policy_loss_fn(action_logits, actions, targets):
  # [actions,...] -> [(idx, action_index),...]
  actions                     = tf.convert_to_tensor(
    list(zip(np.arange(len(actions)), actions))
    )
  # πθ(a|s)
  action_probabilities        = tf.nn.softmax(action_logits)
  # Үйлдлийн индексд харгалзах магадлалын оноог авах
  picked_action_probabilities = tf.gather_nd(action_probabilities, actions) 
  # logπθ(a|s)
  log_probabilites            = tf.cast(tf.math.log(picked_action_probabilities), dtype=tf.float64)
  # logπθ(a|s)*G_t, үйлдлийн магадлалыг discount авсан reward-аар үржих
  loss                        = tf.multiply(log_probabilites, tf.convert_to_tensor(targets))
  # максимумчилахын тулд оптимайзерын минимумчилагчийн эсрэг хасах loss
  return -tf.reduce_sum(loss)


@tf.function(experimental_relax_shapes=True)
def train_policy_network(inputs, actions, advantages):
  with tf.GradientTape() as tape:
    # πθ(a|s)
    predictions = policy(inputs, training=True)
    loss        = policy_loss_fn(predictions, actions, advantages)
  if debug_render:
    tf.print("loss : ", loss)
  gradients = tape.gradient(loss, policy.trainable_variables)
  policy_optimizer.apply_gradients(zip(gradients, policy.trainable_variables))

@tf.function(experimental_relax_shapes=True)
def train_value_network(inputs, targets):
  with tf.GradientTape() as tape:
    predictions = value(inputs)
    loss        = value_loss(targets, predictions)
  gradients = tape.gradient(loss, value.trainable_variables)
  value_optimizer.apply_gradients(zip(gradients, value.trainable_variables))

  

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/ActorCriticPolicy'):
  policy = tf.keras.models.load_model("model_weights/ActorCriticPolicy")
  value  = tf.keras.models.load_model("model_weights/ActorCriticValue")
  print("өмнөх сургасан моделийг ачааллаа")


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
  states, actions, rewards, next_states  = [], [], [], []
  
  while not done:
    global_steps        = global_steps+1
    current_state       = None
    first_state_sampled = False
    # stochastic action sampling
    if (len(temporal_frames)==temporal_length):
      current_state = list(temporal_frames)[0:]
      current_state = np.stack(current_state, axis=-1)
      current_state = np.reshape(current_state, (current_state.shape[ 0], current_state.shape[ 1], current_state.shape[-1]))
      logits        = policy(np.array([current_state], dtype=np.float32), training=False) # πθ(a|s)
      probabilities = tf.nn.softmax(logits).numpy()[0]
      action        = np.random.choice(n_actions, p=probabilities)

      first_state_sampled = True
    else:
      action = env.action_space.sample()

    _, reward, done, _ = env.step(action)

    score = score+reward

    new_state_rendered    = env.render(mode='rgb_array')
    _, new_state_reshaped = preprocess_frame(new_state_rendered, shape=desired_shape)
    temporal_frames.append(np.reshape(new_state_reshaped, (desired_shape[0], desired_shape[1], 1)))

    new_state             = list(temporal_frames)[0:]
    new_state             = np.stack(new_state, axis=-1)
    new_state             = np.reshape(new_state, (new_state.shape[ 0], new_state.shape[ 1], new_state.shape[-1]))

    # sample цуглуулах
    if (len(temporal_frames)==temporal_length and first_state_sampled==True):
      states     .append(current_state)
      actions    .append(action       )
      rewards    .append(reward       )
      next_states.append(new_state    )

    if debug_render:
      img.set_data(np.reshape(list(temporal_frames)[-1], (desired_shape[1], desired_shape[0])))
      plt.draw()
      plt.pause(1e-6)
      pass

    if global_steps%save_per_step==0 and training_happened==True:
      policy.save("model_weights/ActorCriticPolicy")
      value .save("model_weights/ActorCriticValue")
      print("ActorCritic алгоритмын Policy болон Value моделиудыг model_weights/ фолдерт хадгаллаа")

    if done==True:
      if debug_render:
        plt.clf()

      episode_length        = len(states)
      
      input_states          = tf.convert_to_tensor(states, dtype=tf.float32)
      input_next_states     = tf.convert_to_tensor(next_states, dtype=tf.float32)
      
      estimated_values      = value(input_states).numpy()
      estimated_next_values = value(input_next_states).numpy()
      
      td_targets            = np.zeros_like(rewards) # Temporal Difference targets
      td_errors             = np.zeros_like(rewards) # Temporal Difference errors
      
      # Advantage Actor Critic
      
      for t in range(0, episode_length):
        td_targets[t] = rewards   [t] + gamma*estimated_next_values[t]
        td_errors [t] = td_targets[t] - estimated_values[t]
      
      train_value_network(input_states, td_targets)
      print("value неорон сүлжээг сургалаа")

      # td_error-ийг advantage дөхөлтөнд хэрэглэх
      train_policy_network(input_states, actions, td_errors)
      print("policy неорон сүлжээг сургалаа")
      
      training_happened = True
      states, actions, rewards, next_states  = [], [], [], []
      print("%s : %s урттай ажиллагаа %s оноотой дууслаа"%(episode, episode_length, score))


if debug_render:
  plt.close("all")
env.close()
