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

tf.get_logger().setLevel('ERROR')

debug_render  = False
num_episodes  = 5000
save_per_step = 50   # хэдэн алхам тутамд сургасан моделийг хадгалах вэ
train_count   = 1000 # хэдэн удаа сургах вэ
gamma         = 0.99 # discount factor


class PolicyNetwork(tf.keras.Model):
  def __init__(self, n_actions):
    super(PolicyNetwork, self).__init__()
    self.dense_layer  = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')
    self.output_layer = tf.keras.layers.Dense(n_actions, activation='softmax', kernel_initializer='glorot_uniform')
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs):
    dense_out = self.dense_layer(inputs)
    return self.output_layer(dense_out)

class ValueNetwork(tf.keras.Model):
  def __init__(self,):
    super(ValueNetwork, self).__init__()
    self.dense_layer    = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')
    self.output_layer   = tf.keras.layers.Dense(1)
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs):
    dense_out = self.dense_layer(inputs)
    return self.output_layer(dense_out)



env       = gym.make('CartPole-v0')
env.reset()
n_actions = env.action_space.n


policy_optimizer = tf.keras.optimizers.RMSprop(lr=0.0007)
value_optimizer  = tf.keras.optimizers.RMSprop(lr=0.0007)

policy  = PolicyNetwork(n_actions)
value   = ValueNetwork()


@tf.function(experimental_relax_shapes=True)
def policy_loss_fn(action_logits, actions, targets):
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
    loss        = tf.keras.losses.mean_squared_error(targets, predictions)
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
  done  = False
  score = 0
  states, actions, rewards, next_states  = [], [], [], []
  state = env.reset()
  
  while not done:
    global_steps  = global_steps+1
    
    logits        = policy(np.array([state], dtype=np.float32), training=False) # πθ(a|s)
    probabilities = tf.nn.softmax(logits).numpy()[0]
    action        = np.random.choice(n_actions, p=probabilities)

    new_state, reward, done, _ = env.step(action)

    score = score+reward

    states     .append(state)
    actions    .append(action)
    rewards    .append(reward)
    next_states.append(new_state)

    state = new_state

    if debug_render:
      env.render()

    if global_steps%save_per_step==0:
      policy.save("model_weights/ActorCriticPolicy")
      value .save("model_weights/ActorCriticValue")
      print("ActorCritic алгоритмын Policy болон Value моделиудыг model_weights/ фолдерт хадгаллаа")

    if done==True:
      episode_length        = len(states)
      
      input_states          = tf.convert_to_tensor(states     , dtype=tf.float32)
      input_next_states     = tf.convert_to_tensor(next_states, dtype=tf.float32)
      
      estimated_values      = value(input_states     ).numpy()
      estimated_next_values = value(input_next_states).numpy()
      
      td_targets            = np.zeros_like(rewards) # Temporal Difference targets
      td_errors             = np.zeros_like(rewards) # Temporal Difference errors
      
      # Advantage Actor Critic
      
      for t in range(0, episode_length):
        td_targets[t] = rewards   [t] + gamma*estimated_next_values[t]
        td_errors [t] = td_targets[t] - estimated_values[t]
      
      train_value_network(input_states, td_targets)

      # td_error-ийг advantage дөхөлтөнд хэрэглэх
      train_policy_network(input_states, actions, td_errors)
      
      training_happened = True
      states, actions, rewards, next_states  = [], [], [], []
      print("%s : %s урттай ажиллагаа %s оноотой дууслаа"%(episode, episode_length, score))


if debug_render:
  plt.close("all")
env.close()
