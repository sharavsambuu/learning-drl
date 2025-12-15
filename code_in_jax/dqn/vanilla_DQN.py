import os
import random
import math
import gymnasium  as gym
from collections import deque
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax
import time


debug_render  = True
debug         = False
num_episodes  = 200
batch_size    = 64
learning_rate = 0.001
sync_steps    = 100
memory_length = 4000
replay_memory = deque(maxlen=memory_length)

epsilon       = 1.0
epsilon_decay = 0.001
epsilon_max   = 1.0
epsilon_min   = 0.01

gamma         = 0.99
hidden_size   = 64


class DeepQNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=hidden_size//2)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.n_actions)(x)
        return logits


env   = gym.make('CartPole-v1', render_mode='human' if debug_render else None)
state, info = env.reset()

n_actions              = env.action_space.n
dqn_module             = DeepQNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)

params                 = dqn_module.init(jax.random.PRNGKey(0), dummy_input)
q_network_params       = params['params']
target_q_network_params = params['params']

optimizer              = optax.adam(learning_rate)
opt_state              = optimizer.init(q_network_params)


@jax.jit
def policy(params, x):
    predicted_q_values = dqn_module.apply({'params': params}, x)
    max_q_action       = jnp.argmax(predicted_q_values)
    return max_q_action, predicted_q_values


def q_learning_loss(q_value_vec, target_q_value_vec, action, reward, done):
    one_hot_actions = jax.nn.one_hot(action, n_actions)
    q_value         = jnp.sum(one_hot_actions*q_value_vec)
    td_target       = reward + gamma*jnp.max(target_q_value_vec)*(1.-done)
    td_error        = jax.lax.stop_gradient(td_target) - q_value
    return jnp.square(td_error)

q_learning_loss_vmap = jax.vmap(q_learning_loss, in_axes=(0, 0, 0, 0, 0), out_axes=0)


@jax.jit
def train_step(q_network_params, target_q_network_params, opt_state, batch):
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - rewards
    # batch[3] - next_states
    # batch[4] - dones
    def loss_fn(params):
        predicted_q_values = dqn_module.apply({'params': params}, batch[0])
        target_q_values    = dqn_module.apply({'params': target_q_network_params}, batch[3])
        return jnp.mean(
                q_learning_loss_vmap(
                    predicted_q_values,
                    target_q_values,
                    batch[1],
                    batch[2],
                    batch[4]
                )
        )
    loss, gradients = jax.value_and_grad(loss_fn)(q_network_params)
    updates, opt_state = optimizer.update(gradients, opt_state, q_network_params)
    q_network_params = optax.apply_updates(q_network_params, updates)
    return q_network_params, opt_state, loss


global_steps = 0
try:
    for episode in range(num_episodes):
        episode_rewards = []
        state, info = env.reset()

        state = np.array(state, dtype=np.float32)
        while True:
            global_steps = global_steps+1

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action, q_values = policy(q_network_params, jnp.asarray(state, dtype=jnp.float32))
                action = int(action)
                if debug:
                    print("q утгууд :"       , q_values)
                    print("сонгосон action :", action  )

            if epsilon>epsilon_min:
                epsilon = epsilon_min+(epsilon_max-epsilon_min)*math.exp(-epsilon_decay*global_steps)
                if debug:
                    #print("epsilon :", epsilon)
                    pass

            new_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            new_state = np.array(new_state, dtype=np.float32)

            replay_memory.append((state, action, reward, new_state, float(done)))

            # Хангалттай batch цугласан бол DQN сүлжээг сургах
            if (len(replay_memory)>batch_size):
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                q_network_params, opt_state, loss = train_step(
                                            q_network_params,
                                            target_q_network_params,
                                            opt_state,
                                            (   # sample-дсэн batch өгөгдлүүдийг хурдасгуур
                                                # төхөөрөмийн санах ойруу хуулах
                                                jnp.asarray(list(states),      dtype=jnp.float32),
                                                jnp.asarray(list(actions),     dtype=jnp.int32  ),
                                                jnp.asarray(list(rewards),     dtype=jnp.float32),
                                                jnp.asarray(list(next_states), dtype=jnp.float32),
                                                jnp.asarray(list(dones),       dtype=jnp.float32)
                                            )
                                        )

            episode_rewards.append(reward)
            state = new_state

            # Тодорхой алхам тутамд target неорон сүлжээний жингүүдийг сайжирсан хувилбараар солих
            if global_steps%sync_steps==0:
                target_q_network_params = q_network_params
                if debug:
                    print("сайжруулсан жингүүдийг target неорон сүлжээрүү хууллаа")

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(episode_rewards)))
                break
finally:
    env.close()
