# References:
#  - https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/
#  - https://docs.ray.io/en/master/using-ray-with-gpus.html
#  - https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
#      export XLA_PYTHON_CLIENT_ALLOCATOR=platform

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
import ray

ray.init(num_cpus=4, num_gpus=1)


debug_render  = False
num_episodes  = 1500
learning_rate = 0.001
gamma         = 0.99

env_name      = "CartPole-v1"
n_workers     = 4

env           = gym.make(env_name)
state_shape   = env.reset().shape
n_actions     = env.action_space.n
env.close()


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
    value      = jax.lax.stop_gradient(critic_model(jnp.asarray([props[0]]))[0][0])
    next_value = jax.lax.stop_gradient(critic_model(jnp.asarray([props[1]]))[0][0])
    advantage  = props[2]+(gamma*next_value)*(1-props[3]) - value
    def loss_fn(model, advantage):
        action_probabilities = model(jnp.asarray([props[0]]))[0]
        probability          = action_probabilities[props[4]]
        log_probability      = -jnp.log(probability)
        return log_probability*advantage
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target, advantage)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss

@jax.jit
def backpropagate_critic(optimizer, props):
    # props[0] - state
    # props[1] - next_state
    # props[2] - reward
    # props[3] - done
    next_value = jax.lax.stop_gradient(optimizer.target(jnp.asarray([props[1]]))[0][0])
    def loss_fn(model):
        value      = model(jnp.asarray([props[0]]))[0][0]
        advantage  = props[2]+(gamma*next_value)*(1-props[3]) - value
        return jnp.square(advantage)
    loss, gradients = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer       = optimizer.apply_gradient(gradients)
    return optimizer, loss


@ray.remote(num_gpus=0.15)
def rollout_worker(parameter_server):
    class Brain(object):
        def __init__(self,):
            class ActorNetwork(flax.nn.Module):
                def apply(self_, x, n_actions):
                    dense_layer_1      = flax.nn.Dense(x, 64)
                    activation_layer_1 = flax.nn.relu(dense_layer_1)
                    dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
                    activation_layer_2 = flax.nn.relu(dense_layer_2)
                    output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
                    output_layer       = flax.nn.softmax(output_dense_layer)
                    return output_layer
            class CriticNetwork(flax.nn.Module):
                def apply(self_, x):
                    dense_layer_1      = flax.nn.Dense(x, 64)
                    activation_layer_1 = flax.nn.relu(dense_layer_1)
                    dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
                    activation_layer_2 = flax.nn.relu(dense_layer_2)
                    output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
                    return output_dense_layer

            self.actor_module     = ActorNetwork.partial(n_actions=n_actions)
            _, self.actor_params  = self.actor_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
            self.actor_model      = flax.nn.Model(self.actor_module, self.actor_params)

            self.critic_module    = CriticNetwork.partial()
            _, self.critic_params = self.critic_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
            self.critic_model     = flax.nn.Model(self.critic_module, self.critic_params)

            self.actor_optimizer  = flax.optim.Adam(learning_rate).create(self.actor_model)
            self.critic_optimizer = flax.optim.Adam(learning_rate).create(self.critic_model)
            pass

        def act(self, state):
            action_probabilities = actor_inference(self.actor_optimizer.target, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action               = np.random.choice(n_actions, p=action_probabilities)
            return action

        def learn(self, actor_batch, critic_batch):
            self.actor_optimizer, _ = backpropagate_actor(
                self.actor_optimizer,
                self.critic_optimizer.target,
                (
                    jnp.asarray(actor_batch[0]),
                    jnp.asarray(actor_batch[1]),
                    jnp.asarray(actor_batch[2]),
                    jnp.asarray(actor_batch[3]),
                    jnp.asarray(actor_batch[4])
                )
            )
            self.critic_optimizer, _ = backpropagate_critic(
                self.critic_optimizer,
                (
                    jnp.asarray(critic_batch[0]),
                    jnp.asarray(critic_batch[1]),
                    jnp.asarray(critic_batch[2]),
                    jnp.asarray(critic_batch[3]),
                )
            )

        def get_params(self,):
            return self.actor_optimizer.target.params, self.critic_optimizer.target.params

        def update_params(self, actor_params, critic_params):
            self.actor_model  = self.actor_model.replace (params=actor_params )
            self.critic_model = self.critic_model.replace(params=critic_params)


    brain     = Brain()
    env       = gym.make(env_name)

    try:
        for episode in range(num_episodes):
            state   = env.reset()
            rewards = []
            while True:
                actor_params, critic_params = ray.get(parameter_server.get.remote(['actor_params', 'critic_params']))
                brain.update_params(actor_params, critic_params)

                action = brain.act(state)

                next_state, reward, done, _ = env.step(int(action))

                rewards.append(reward)

                brain.learn(
                        (state, next_state, reward, int(done), action),
                        (state, next_state, reward, int(done))
                    )
                actor_params, critic_params = brain.get_params()
                parameter_server.update.remote(
                    ['actor_params', 'critic_params'], 
                    [actor_params, critic_params])

                state = next_state

                if done:
                    print("episode {}, reward : {}".format(episode, sum(rewards)))
                    break
    except Exception as e:
        print(e)
    finally:
        env.close()
        return 0
    return 0


class ActorNetwork(flax.nn.Module):
    def apply(self_, x, n_actions):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, n_actions)
        output_layer       = flax.nn.softmax(output_dense_layer)
        return output_layer

class CriticNetwork(flax.nn.Module):
    def apply(self_, x):
        dense_layer_1      = flax.nn.Dense(x, 64)
        activation_layer_1 = flax.nn.relu(dense_layer_1)
        dense_layer_2      = flax.nn.Dense(activation_layer_1, 32)
        activation_layer_2 = flax.nn.relu(dense_layer_2)
        output_dense_layer = flax.nn.Dense(activation_layer_2, 1)
        return output_dense_layer

actor_module     = ActorNetwork.partial(n_actions=n_actions)
_, actor_params  = actor_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])
critic_module    = CriticNetwork.partial()
_, critic_params = critic_module.init_by_shape(jax.random.PRNGKey(0), [state_shape])


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

result_ids = [rollout_worker.remote(parameter_server) for _ in range(n_workers)]
while len(result_ids):
    done_id, result_ids = ray.wait(result_ids)
    print(done_id, "task finished to work.")

