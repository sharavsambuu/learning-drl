import os
import random
import math
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np


debug_render  = True
debug         = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99
batch_size    = 64
sync_steps    = 100
memory_length = 4000

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01


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


class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class DuelingCriticNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 64)
        activation_layer_2 = flax.nn.relu(dense_layer_2)

        value_dense       = flax.nn.Dense(activation_layer_2, 64)
        value             = flax.nn.relu(value_dense)
        value             = flax.nn.Dense(value, 1)

        advantage_dense   = flax.nn.Dense(activation_layer_2, 64)
        advantage         = flax.nn.relu(advantage_dense)
        advantage         = flax.nn.Dense(advantage, n_actions)

        advantage_average = jnp.mean(advantage, keepdims=True)

        q_values_layer    = jnp.subtract(jnp.add(advantage, value), advantage_average)
        return q_values_layer


env       = gym.make('CartPole-v1')
state     = env.reset()
n_actions = env.action_space.n

actor_module        = ActorNetwork.partial(n_actions=n_actions)
_, actor_params     = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
actor_model         = flax.nn.Model(actor_module, actor_params)

critic_module       = DuelingCriticNetwork.partial(n_actions=n_actions)
_, critic_params    = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_model        = flax.nn.Model(critic_module, critic_params)
target_critic_model = flax.nn.Model(critic_module, critic_params)

actor_optimizer     = flax.optim.Adam(learning_rate).create(actor_model)
critic_optimizer    = flax.optim.Adam(learning_rate).create(critic_model)

per_memory          = PERMemory(memory_length)

@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def backpropagate_actor(optimizer, critic_model, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    # props[4] - action
    #value      = critic_model(jnp.asarray([props[0]]))[0][props[4]]
    #value      = jnp.nanmax(critic_model(jnp.asarray([props[0]]))[0])
    #next_value = jnp.nanmax(critic_model(jnp.asarray([props[1]]))[0])
    value      = critic_model(jnp.asarray([props[0]]))[0][props[4]]
    next_value = critic_model(jnp.asarray([props[1]]))[0][props[4]]
    advantage  = props[2]+(gamma*next_value)*(1-props[3]) - value + 1e-10
    def loss_fn(model, advantage):
        action_probabilities = model(jnp.asarray([props[0]]))[0]
        probability          = action_probabilities[props[4]]
        log_probability      = -jnp.log(probability)
        return log_probability*jax.lax.stop_gradient(advantage)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target, advantage)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


@jax.vmap
def calculate_td_error(q_value_vec, target_q_value_vec, action, reward):
    td_target = reward + gamma*jnp.amax(target_q_value_vec)
    td_error  = td_target - q_value_vec[action]
    return jnp.abs(td_error)

@jax.jit
def td_errors(model, target_model, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    predicted_q_values = model(batch[0])
    target_q_values    = target_model(batch[3])
    return calculate_td_error(predicted_q_values, target_q_values, batch[1], batch[2])

@jax.vmap
def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    td_target = reward + gamma*jnp.amax(target_q_value_vec)*(1.-done)
    td_error  = jax.lax.stop_gradient(td_target) - q_value_vec[action]
    return jnp.square(td_error)

@jax.jit
def backpropagate_critic(optimizer, target_model, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(model):
        predicted_q_values = model(batch[0])
        target_q_values    = target_model(batch[3])
        return jnp.mean(
                q_learning_loss(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                    )
                )
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss, td_errors(optimizer.target, target_model, batch)


global_step = 0

try:
    for episode in range(num_episodes):
        state           = env.reset()
        episode_rewards = []
        while True:
            global_step = global_step+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action_probabilities  = actor_inference(actor_optimizer.target, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action_probabilities /= (np.sum(action_probabilities) + 1e-10)
                action                = np.random.choice(n_actions, p=action_probabilities)

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_step)

            next_state, reward, done, _ = env.step(int(action))

            episode_rewards.append(reward)

            actor_optimizer, _  = backpropagate_actor(
                    actor_optimizer,
                    critic_optimizer.target,
                    (state, next_state, reward, int(done), action)
                    )

            temporal_difference = float(td_errors(critic_optimizer.target, target_critic_model, (
                    jnp.asarray([state]),
                    jnp.asarray([action]),
                    jnp.asarray([reward]),
                    jnp.asarray([next_state])
                ))[0])
            per_memory.add(temporal_difference, (state, action, reward, next_state, int(done)))
            
            batch = per_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for i in range(batch_size):
                states.append     (batch[i][1][0])
                actions.append    (batch[i][1][1])
                rewards.append    (batch[i][1][2])
                next_states.append(batch[i][1][3])
                dones.append      (batch[i][1][4])

            critic_optimizer, loss, new_td_errors = backpropagate_critic(
                                        critic_optimizer,
                                        target_critic_model,
                                        (
                                            jnp.asarray(states),
                                            jnp.asarray(actions),
                                            jnp.asarray(rewards),
                                            jnp.asarray(next_states),
                                            jnp.asarray(dones)
                                        )
                                    )
            new_td_errors = np.array(new_td_errors)
            for i in range(batch_size):
                idx = batch[i][0]
                per_memory.update(idx, new_td_errors[i])

            state = next_state

            if global_step%sync_steps==0:
                target_critic_model = target_critic_model.replace(params=critic_optimizer.target.params)

            if debug_render:
                env.render()

            if done:
                print(episode, " - reward :", sum(episode_rewards))
                break
finally:
    env.close()
