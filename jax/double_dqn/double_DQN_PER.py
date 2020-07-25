import os
import random
import math
import gym
from collections import deque

import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render  = True
debug         = False
num_episodes  = 500
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99 # discount factor


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


class DeepQNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_layer       = flax.nn.Dense(activation_layer_2, n_actions)
        return output_layer


env   = gym.make('CartPole-v0')
state = env.reset()

n_actions        = env.action_space.n

dqn_module       = DeepQNetwork.partial(n_actions=n_actions)
_, params        = dqn_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
q_network        = flax.nn.Model(dqn_module, params)
target_q_network = flax.nn.Model(dqn_module, params)

optimizer        = flax.optim.Adam(learning_rate).create(q_network)

per_memory       = PERMemory(memory_length)

@jax.jit
def policy(model, x):
    predicted_q_values = model(x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


@jax.vmap
def calculate_td_error(q_value_vec, next_q_value_vec, target_q_value_vec, action, reward):
    action_select = jnp.argmax(next_q_value_vec)
    td_target     = reward + gamma*target_q_value_vec[action_select]
    td_error      = td_target - q_value_vec[action]
    return jnp.abs(td_error)

@jax.jit
def td_error(model, target_model, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    predicted_q_values      = model(batch[0])
    predicted_next_q_values = model(batch[3])
    target_q_values         = target_model(batch[3])
    return calculate_td_error(predicted_q_values, predicted_next_q_values, target_q_values, batch[1], batch[2])


@jax.vmap
def q_learning_loss(q_value_vec, next_q_value_vec, target_q_value_vec, action, reward, done):
    action_select = jnp.argmax(next_q_value_vec)
    td_target     = reward + gamma*target_q_value_vec[action_select]*(1.-done)
    td_error      = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit
def train_step(optimizer, target_model, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        # reference : https://mc.ai/introduction-to-double-deep-q-learning-ddqn/
        predicted_q_values      = model(batch[0])
        predicted_next_q_values = model(batch[3])
        target_q_values         = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    predicted_next_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_error(optimizer.target, target_model, batch)


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(optimizer.target, state)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    #print("epsilon :", epsilon)
                    pass

            new_state, reward, done, _ = env.step(int(action))

            # sample нэмэхдээ temporal difference error-ийг тооцож нэмэх
            temporal_difference = float(td_error(optimizer.target, target_q_network, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([new_state])
                ))[0])
            per_memory.add(temporal_difference, (state, action, reward, new_state, int(done)))

            # Prioritized Experience Replay санах ойгоос batch үүсгээд DQN сүлжээг сургах
            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for i in range(batch_size):
                states.append     (batch[i][1][0])
                actions.append    (batch[i][1][1])
                rewards.append    (batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones.append      (batch[i][1][4])

            optimizer, loss, new_td_errors = train_step(
                                        optimizer,
                                        target_q_network,
                                        (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур 
                                            # төхөөрөмийн санах ойруу хуулах
                                            jnp.asarray(states),
                                            jnp.asarray(actions),
                                            jnp.asarray(rewards),
                                            jnp.asarray(next_states),
                                            jnp.asarray(dones)
                                        )
                                    )
            # batch-аас бий болсон temporal difference error-ийн дагуу санах ойг шинэчлэх
            new_td_errors = np.array(new_td_errors)
            for i in range(batch_size):
                idx = batch[i][0]
                per_memory.update(idx, new_td_errors[i])

            episode_rewards.append(reward)
            state = new_state

            # Тодорхой алхам тутамд target неорон сүлжээний жингүүдийг сайжирсан хувилбараар солих
            if global_steps%sync_steps==0:
                target_q_network = target_q_network.replace(params=optimizer.target.params)
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
