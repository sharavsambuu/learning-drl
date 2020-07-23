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
num_episodes  = 2000
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


#policy_optimizer = tf.keras.optimizers.RMSprop(lr=0.0007)
policy_optimizer = tf.keras.optimizers.Adam(0.001)

policy  = PolicyNetwork(n_actions)


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
  

if not os.path.exists("model_weights"):
  os.makedirs("model_weights")
if os.path.exists('model_weights/ReinforcePolicy'):
  policy = tf.keras.models.load_model("model_weights/ReinforcePolicy")
  print("өмнөх сургасан моделийг ачааллаа")


global_steps    = 0
rewards_history = []

for episode in range(num_episodes):
  done  = False
  score = 0
  states, actions, rewards = [], [], []
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

    state = new_state

    if debug_render:
      env.render()

    if global_steps%save_per_step==0:
      policy.save("model_weights/ReinforcePolicy")
      print("Моделийг хадгаллаа")

    if done==True:
      episode_length     = len(states)
      input_states       = tf.convert_to_tensor(states, dtype=tf.float32)
      discounted_rewards = np.zeros_like(rewards)
      running_add        = 0
      #for i in reversed(range(len(rewards))):
      #  running_add = running_add * gamma + rewards[i]
      #  discounted_rewards[i] = running_add
      for t in range(0, episode_length):
        G_t = 0
        for idx, j in enumerate(range(t, episode_length)):
          G_t = G_t + (gamma**idx)*rewards[j]
        discounted_rewards[t] = G_t
      # normalize rewards
      discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)
      #discounted_rewards = discounted_rewards / np.std(discounted_rewards - np.mean(discounted_rewards))
      
      train_policy_network(input_states, actions, discounted_rewards)
      
      training_happened         = True
      states, rewards, actions  = [], [], []
      print("%s : %s урттай ажиллагаа %s оноотой дууслаа"%(episode, episode_length, score))
      rewards_history.append(score)


plt.plot(rewards_history)
x = np.array(range(len(rewards_history)))
smooth_func = np.poly1d(np.polyfit(x, rewards_history, 3))
plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
plt.savefig('reinforce_baseline.png')
#plt.show()


if debug_render:
  plt.close("all")
env.close()
