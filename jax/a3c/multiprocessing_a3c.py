import os
import sys
import random
import math
import time
import threading
import multiprocessing
import pickle
import gym
import flax
import jax
from jax import numpy as jnp
import numpy as np
import ray; ray.init()



debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99

env_name      = "CartPole-v1"
n_workers     = 4


class ActorNetwork(flax.nn.Module):
    def apply(self, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self, x):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer


env       = gym.make(env_name)
state     = env.reset()
n_actions = env.action_space.n
env.close()

#actor_module     = ActorNetwork.partial(n_actions=n_actions)
#_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
#actor_model      = flax.nn.Model(actor_module, actor_params)

#critic_module    = CriticNetwork.partial()
#_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
#critic_model     = flax.nn.Model(critic_module, critic_params)

#actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)
#critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)


@jax.jit
def actor_inference(model, x):
    return model(x)

@jax.jit
def critic_inference(model, x):
    return model(x)

@jax.jit
def backpropagate_critic(optimizer, props):
    # props[0] - states
    # props[1] - discounted_rewards
    def loss_fn(model):
        values      = model(props[0])
        advantages  = props[1] - values
        return jnp.mean(jnp.square(advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer, critic_model, props):
    # props[0] - states
    # props[1] - discounted_rewards
    # props[2] - actions
    values      = jax.lax.stop_gradient(critic_model(props[0]))
    advantages  = props[1] - values
    def loss_fn(model):
        action_probabilities = model(props[0])
        probabilities        = gather(action_probabilities, props[2])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, advantages))
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss



actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])
critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state.shape])


@ray.remote
def training_worker(parameter_server):
    actor_params, critic_params = parameter_server.get.remote(
                        ['actor_params', 'critic_params'])
    actor_module     = ActorNetwork.partial(n_actions=n_actions)
    critic_module    = CriticNetwork.partial()
    actor_model      = flax.nn.Model(actor_module, actor_params)
    critic_model     = flax.nn.Model(critic_module, critic_params)
    actor_optimizer  = flax.optim.Adam(learning_rate).create(actor_model)
    critic_optimizer = flax.optim.Adam(learning_rate).create(critic_model)

    env = gym.make(env_name)

    try:
        for episode in range(num_episodes):
            state = env.reset()
            states, actions, rewards, dones = [], [], [], []
            while True:
                actor_params, critic_params = parameter_server.get.remote(
                        ['actor_params', 'critic_params'])
                actor_model  = actor_model.replace(params=actor_params)
                critic_model = critic_model.replace(params=critic_params)

                action_probabilities  = actor_inference(actor_optimizer.target, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)
                next_state, reward, done, _ = env.step(int(action))
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(int(done))
                state = next_state
                if done:
                    print("episode {}, reward : {}".format(episode, sum(rewards)))
                    episode_length = len(rewards)
                    discounted_rewards = np.zeros_like(rewards)
                    for t in range(0, episode_length):
                        G_t = 0
                        for idx, j in enumerate(range(t, episode_length)):
                            G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                        discounted_rewards[t] = G_t
                    discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                    discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-10)
                    actor_optimizer, _  = backpropagate_actor(
                        actor_optimizer,
                        critic_optimizer.target,
                        (
                            jnp.asarray(states),
                            jnp.asarray(discounted_rewards),
                            jnp.asarray(actions)
                        )
                    )
                    critic_optimizer, _ = backpropagate_critic(
                        critic_optimizer,
                        (
                            jnp.asarray(states),
                            jnp.asarray(discounted_rewards),
                        )
                    )
                    parameter_server.update.remote(
                            ['actor_params', 'critic_params'],
                            [
                                actor_optimizer.target.params,
                                critic_optimizer.target.params
                            ])
                    episode_count = episode_count + 1
                    break
    finally:
        env.close()


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        self.parameters = dict(zip(keys, values))
    def get(self, keys):
        return [self.parameters[key] for key in keys]
    def update(self, keys, values):
        for key, value in zip(keys, values):
            self.parameters[key] = value

parameter_server = ParameterServer.remote(
        ['actor_params', 'critic_params'],
        [actor_params, critic_params])
result = parameter_server.get.remote(
        ['actor_params', 'critic_params'])
print("got initial result")
parameter_server.update.remote(
        ['actor_params', 'critic_params'],
        [actor_params, critic_params])
print("parameter updated...")
result = parameter_server.get.remote(
        ['actor_params', 'critic_params'])
ray.get(result)
print("got updated parameters")
print(ray.get(result))
exit(0)


for _ in range(n_workers):
    ray.get(training_worker.remote(parameter_server))

