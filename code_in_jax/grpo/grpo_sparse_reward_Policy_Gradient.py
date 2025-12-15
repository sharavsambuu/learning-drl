#
# GRPO (MVP) + A2C Baseline (DeepSeek-V3.2 Advantage)  + Entropy Bonus  (Sparse Reward)
#
#    - Sparse reward wrapper on CartPole (reward only at episode end if steps >= threshold)
#    - Group rollouts (K trajectories) per update
#    - DeepSeek-V3.2 group advantage:   A_i = R_i - mean(R)   (NO std division)
#    - A2C-style critic baseline:       adv_t = (G_t - mean(R)) - V(s_t)
#    - Actor update: -E[ log_pi(a|s) * stopgrad(adv_t) + ent_coef * H(pi) ]
#    - Critic update: MSE(G_t, V(s_t))   (critic learns uncentered returns)
#    - Epsilon-greedy mixed with stochastic policy sampling for exploration
#
#

import os
import random
import math
import jax
import optax
import numpy      as np
import gymnasium  as gym
import flax.linen as nn
from   jax        import numpy as jnp


debug_render            = True
debug                   = True
play_frequency          = 30
num_episodes            = 20000
learning_rate           = 0.0005
gamma                   = 0.99
env_name                = "CartPole-v1"
group_size              = 8
max_steps               = 500

sparse_reward_threshold = 90
sparse_reward_value     = 1.0

epsilon_start           = 1.0
epsilon_end             = 0.01
epsilon_decay_episodes  = num_episodes / 2

entropy_coefficient     = 0.01


class SparseCartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps"  : 50,
    }
    def __init__(self, render_mode='human', sparse_reward_threshold=90, sparse_reward_value=1.0):
        super().__init__()
        self.env = gym.make(env_name, render_mode=render_mode if debug_render else None)
        self.sparse_reward_threshold = int(sparse_reward_threshold)
        self.sparse_reward_value     = float(sparse_reward_value)
        self.current_steps           = 0
        self.action_space            = self.env.action_space
        self.observation_space       = self.env.observation_space

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        observation, info  = self.env.reset(seed=seed, options=options)
        return np.array(observation, dtype=np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1

        done_boundary = bool(terminated or truncated)

        # Sparse reward at episode end if agent survived long enough (time-limit OR later failure)
        sparse_reward = 0.0
        if done_boundary and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value

        return np.array(observation, dtype=np.float32), float(sparse_reward), terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()


class ActorNetwork(nn.Module):
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

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        activation_layer_1 = nn.relu(x)
        dense_layer_2      = nn.Dense(features=32)(activation_layer_1)
        activation_layer_2 = nn.relu(dense_layer_2)
        output_dense_layer = nn.Dense(features=1)(activation_layer_2)
        return output_dense_layer


env_array   = [SparseCartPoleEnv(render_mode=None, sparse_reward_threshold=sparse_reward_threshold, sparse_reward_value=sparse_reward_value) for _ in range(group_size)]
env         =  SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=sparse_reward_threshold, sparse_reward_value=sparse_reward_value)

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n


actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

critic_module          = CriticNetwork()
critic_params          = critic_module.init(jax.random.PRNGKey(1), dummy_input)['params']
critic_model_params    = critic_params

actor_optimizer_def    = optax.adam(learning_rate)
critic_optimizer_def   = optax.adam(learning_rate)

actor_optimizer_state  = actor_optimizer_def .init(actor_model_params)
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x).squeeze(-1)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # props[0] - states  (B, S)
    # props[1] - returns (B,)
    def loss_fn(params):
        values = critic_module.apply({'params': params}, props[0]).squeeze(-1)
        td     = props[1] - values
        return jnp.mean(jnp.square(td))
    loss, gradients = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_optimizer_state = critic_optimizer_def.update(gradients, optimizer_state, critic_model_params)
    new_params = optax.apply_updates(critic_model_params, updates)
    return new_optimizer_state, new_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, critic_model_params, props):
    # props[0] - states  (B, S)
    # props[1] - actions (B,)
    # props[2] - returns (B,)
    # props[3] - group_mean_return (scalar)
    def loss_fn(params):
        action_probas = actor_module.apply({'params': params}, props[0])
        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        values        = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]).squeeze(-1))

        # GRPO-style (DeepSeek-V3.2): center by group mean of R_i (no std)
        advantages    = (props[2] - props[3]) - values

        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        loss          = -jnp.mean(logp * jax.lax.stop_gradient(advantages) + entropy_coefficient * entropies)
        return loss

    loss, gradients = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_params, loss


def compute_returns(rewards, done_terms, bootstrap):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = float(bootstrap)
    for t in reversed(range(T)):
        # terminal reward is included, only the bootstrap beyond terminal is masked
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns


def rollout_trajectory(group_member_id, actor_model_params, critic_model_params, seed, epsilon):
    env_item       = env_array[group_member_id]

    # each group member starts from the same seeded reset each episode
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states         = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32)
    actions        = np.zeros(shape=(max_steps,), dtype=np.int32)
    rewards        = np.zeros(shape=(max_steps,), dtype=np.float32)
    done_terms     = np.zeros(shape=(max_steps,), dtype=np.float32)

    step           = 0

    last_state     = state
    last_truncated = False
    last_terminated= False

    for _ in range(max_steps):

        if random.random() < float(epsilon):
            action = env_item.action_space.sample()
        else:
            action_probabilities = actor_inference(actor_model_params, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action               = np.random.choice(n_actions, p=action_probabilities)

        next_state, reward, terminated, truncated, info = env_item.step(int(action))

        done_boundary = bool(terminated or truncated)
        done_term     = bool(terminated)

        next_state = np.array(next_state, dtype=np.float32)

        states    [step, :] = state
        actions   [step   ] = int  (action)
        rewards   [step   ] = float(reward)
        done_terms[step   ] = float(done_term)

        step += 1

        # must advance state
        state = next_state

        last_state      = state
        last_truncated  = bool(truncated)
        last_terminated = bool(terminated)

        if done_boundary:
            break

    trajectory_length = step

    bootstrap = 0.0
    if last_truncated and (not last_terminated):
        bootstrap = float(critic_inference(critic_model_params, jnp.asarray([last_state]))[0])

    returns = compute_returns(
        rewards    [:trajectory_length],
        done_terms [:trajectory_length],
        bootstrap
    )

    group_reward = float(returns[0]) if trajectory_length > 0 else 0.0

    return group_member_id, group_reward, trajectory_length, states, actions, returns


def rollout_group(actor_model_params, critic_model_params, seed, epsilon):
    group_rewards  = np.zeros(shape=(group_size,), dtype=np.float32)
    group_lengths  = np.zeros(shape=(group_size,), dtype=np.int32  )

    group_states   = []
    group_actions  = []
    group_returns  = []

    for group_member_id in range(group_size):
        member_id, group_reward, trajectory_length, states, actions, returns = rollout_trajectory(
            group_member_id     = group_member_id,
            actor_model_params  = actor_model_params,
            critic_model_params = critic_model_params,
            seed                = seed,
            epsilon             = epsilon
        )

        group_rewards[member_id] = group_reward
        group_lengths[member_id] = trajectory_length

        group_states .append(states [:trajectory_length])
        group_actions.append(actions[:trajectory_length])
        group_returns.append(returns)

    group_mean_reward = float(np.mean(group_rewards))

    return group_mean_reward, group_lengths, group_states, group_actions, group_returns


epsilon = epsilon_start

try:
    global_step = 0

    for episode in range(num_episodes):

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        group_mean_reward, group_lengths, group_states, group_actions, group_returns = rollout_group(
            actor_model_params  = actor_model_params ,
            critic_model_params = critic_model_params,
            seed                = episode            ,
            epsilon             = epsilon
        )

        # Flatten variable-length trajectories into one batch (no padding leakage)
        flat_states  = np.concatenate(group_states , axis=0) if len(group_states)  > 0 else np.zeros((0,) + state_shape, dtype=np.float32)
        flat_actions = np.concatenate(group_actions, axis=0) if len(group_actions) > 0 else np.zeros((0,), dtype=np.int32)
        flat_returns = np.concatenate(group_returns, axis=0) if len(group_returns) > 0 else np.zeros((0,), dtype=np.float32)

        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])

            critic_optimizer_state, critic_model_params, critic_loss = backpropagate_critic(
                critic_optimizer_state,
                critic_model_params,
                (
                    jnp.asarray(flat_states , dtype=jnp.float32),
                    jnp.asarray(flat_returns, dtype=jnp.float32),
                )
            )

            actor_optimizer_state, actor_model_params, actor_loss = backpropagate_actor(
                actor_optimizer_state,
                actor_model_params   ,
                critic_model_params  ,
                (
                    jnp.asarray(flat_states      , dtype=jnp.float32),
                    jnp.asarray(flat_actions     , dtype=jnp.int32  ),
                    jnp.asarray(flat_returns     , dtype=jnp.float32),
                    jnp.asarray(group_mean_reward, dtype=jnp.float32),
                )
            )

        if debug and (episode % 10 == 0):
            group_success = float(np.mean([1.0 if r > 0.0 else 0.0 for r in (np.array([gr[0] if len(gr)>0 else 0.0 for gr in group_returns], dtype=np.float32))])) if len(group_returns) > 0 else 0.0
            print(f"Ep {episode:6d} | Steps {global_step:9d} | "
                  f"GroupMeanR {group_mean_reward:9.4f} | "
                  f"Eps {epsilon:6.3f} | "
                  f"ActorLoss {float(actor_loss):9.4f} | CriticLoss {float(critic_loss):9.4f}")

        if episode % play_frequency == 0 and debug_render == True:
            state, info = env.reset(seed=int(episode))
            state       = np.array(state, dtype=np.float32)

            rewards = []
            steps   = 0

            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                next_state = np.array(next_state, dtype=np.float32)

                rewards.append(float(reward))
                steps += 1
                state = next_state

                env.render()

                if done:
                    print(f"Episode {episode}, sparse_reward_sum : {round(np.sum(rewards), 3)}, steps: {steps}, epsilon: {epsilon:.2f}")
                    break

finally:
    env.close()
    for env_item in env_array:
        env_item.close()
