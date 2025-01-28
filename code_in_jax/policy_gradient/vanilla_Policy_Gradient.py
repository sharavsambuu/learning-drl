import os
import random
import math
import gymnasium  as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax

debug_render  = True
debug         = False
num_episodes  = 600
learning_rate = 0.001
gamma         = 0.99   # discount factor

class PolicyNetwork(nn.Module): 
    n_actions: int 

    @nn.compact 
    def __call__(self, x): 
        x = nn.Dense(features=64)(x) 
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1) 
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=self.n_actions)(activation_layer_2) 
        output_layer       = nn.softmax(output_dense_layer)
        return output_layer

env   = gym.make('CartPole-v1', render_mode='human') 
state, info = env.reset() 
state = np.array(state, dtype=np.float32) 

n_actions             = env.action_space.n
pg_module             = PolicyNetwork(n_actions=n_actions) 
dummy_input           = jnp.zeros(state.shape)
params                = pg_module.init(jax.random.PRNGKey(0), dummy_input)['params'] 
policy_network_params = params 
optimizer_def         = optax.adam(learning_rate) 
optimizer_state       = optimizer_def.init(policy_network_params) 


@jax.jit
def policy_inference(params, x): 
    return pg_module.apply({'params': params}, x) 

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def train_step(optimizer_state, policy_network_params, batch): 
    # batch[0] - states
    # batch[1] - actions
    # batch[2] - discounted rewards
    def loss_fn(params):
        action_probabilities_list   = pg_module.apply({'params': params}, batch[0]) 
        picked_action_probabilities = gather(action_probabilities_list, batch[1])
        log_probabilities           = jnp.log(picked_action_probabilities)
        losses                      = jnp.multiply(log_probabilities, batch[2])
        return -jnp.sum(losses)
    loss, gradients = jax.value_and_grad(loss_fn)(policy_network_params)
    updates, new_optimizer_state = optimizer_def.update(gradients, optimizer_state, policy_network_params) 
    new_policy_network_params = optax.apply_updates(policy_network_params, updates) 
    return new_optimizer_state, new_policy_network_params, loss 


global_steps = 0
try:
    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], [] 
        state, info = env.reset() 
        state       = np.array(state, dtype=np.float32) 
        while True:
            global_steps = global_steps+1

            action_probabilities  = policy_inference(policy_network_params, jnp.asarray([state])) 
            action_probabilities  = np.array(action_probabilities[0])
            action_probabilities /= action_probabilities.sum()
            action                = np.random.choice(n_actions, p=action_probabilities)

            new_state, reward, terminated, truncated, info = env.step(int(action)) # env.step returns new values
            done = terminated or truncated
            new_state = np.array(new_state, dtype=np.float32) # ensure next_state is float32

            states .append(state )
            actions.append(action)
            rewards.append(reward)
            dones  .append(done  ) 

            state = new_state

            if debug_render:
                env.render()

            if done:
                print("{} - нийт reward : {}".format(episode, sum(rewards)))
                episode_length     = len(rewards)
                discounted_rewards = np.zeros_like(rewards)
                for t in range(0, episode_length):
                    G_t = 0
                    for idx, j in enumerate(range(t, episode_length)):
                        G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j]) 
                    discounted_rewards[t] = G_t
                discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5) 

                optimizer_state, policy_network_params, loss = train_step( 
                    optimizer_state,
                    policy_network_params,
                    (
                        jnp.asarray(states),
                        jnp.asarray(actions),
                        jnp.asarray(discounted_rewards)
                    ))

                break
finally:
    env.close()