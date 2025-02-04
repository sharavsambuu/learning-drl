import os
import random
import math
import gymnasium as gym
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


debug_render            = True
debug                   = True
play_frequency          = 100
num_episodes            = 20000 
learning_rate           = 0.0005          
gamma                   = 0.99
env_name                = "CartPole-v1"
group_size              = 8               # diverse experiences
max_steps               = 500
sparse_reward_threshold = 90              # Minimum steps to get a reward for sparse env
sparse_reward_value     = 1.0             # Increased sparse reward value
epsilon_start           = 1.0             # Epsilon start value
epsilon_end             = 0.01            # Epsilon end value
epsilon_decay_episodes  = num_episodes/2  # Decay epsilon over some episodes


class SparseCartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps"  : 50,
    }
    def __init__(self, render_mode='human', sparse_reward_threshold=40, sparse_reward_value=10):
        super().__init__()
        self.env = gym.make('CartPole-v1', render_mode=render_mode if debug_render else None)
        self.sparse_reward_threshold = sparse_reward_threshold
        self.sparse_reward_value     = sparse_reward_value
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
        sparse_reward       = 0
        done                = terminated or truncated
        if terminated and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value
            if debug:
                print(f"Sparse reward given: {sparse_reward} at step {self.current_steps}")
        return np.array(observation, dtype=np.float32), sparse_reward, terminated, truncated, info
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


env_array   = [SparseCartPoleEnv(render_mode=None, sparse_reward_threshold=sparse_reward_threshold, sparse_reward_value=sparse_reward_value) for _ in range(group_size)]
env         = SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=sparse_reward_threshold, sparse_reward_value=sparse_reward_value)
state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n


actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params
actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params )


@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.vmap
def gather(probability_vec, action_index):
    return probability_vec[action_index]

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, props):
    # props[0] - states
    # props[1] - actions
    # props[2] - GRPO advantage
    def loss_fn(params):
        actions_probas       = actor_module.apply({'params': params}, props[0])
        probabilities        = gather(actions_probas, props[1])
        log_probabilities    = -jnp.log(probabilities)
        return jnp.mean(jnp.multiply(log_probabilities, props[2]))
    loss, gradients = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_actor_model_params = optax.apply_updates(actor_model_params, updates)
    return new_optimizer_state, new_actor_model_params, loss


def rollout_trajectory(group_member_id, actor_model_params, epsilon): 
    group_reward    = 0.0
    states          = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32)
    actions         = np.zeros(shape=(max_steps,), dtype=np.int32  )
    rewards         = np.zeros(shape=(max_steps,), dtype=np.float32)
    dones           = np.zeros(shape=(max_steps,), dtype=np.float32)
    step            = 0
    env             = env_array[group_member_id]
    state, info     = env.reset()
    state           = np.array(state, dtype=np.float32)
    for _ in range(max_steps):
        if random.random() < epsilon: # Epsilon-greedy action selection
            action = env.action_space.sample()
        else:
            action_probabilities = actor_inference(actor_model_params, jnp.asarray([state]))
            action_probabilities = np.array(action_probabilities[0])
            action               = np.random.choice(n_actions, p=action_probabilities)

        next_state, reward, terminated, truncated, info = env.step(int(action))
        done                    = terminated or truncated
        next_state              = np.array(next_state, dtype=np.float32)
        states        [step, :] = state
        actions       [step   ] = action
        rewards       [step   ] = reward
        dones         [step   ] = float(done)
        step                   +=1
        if done:
            trajectory_length  = step
            discounted_rewards = np.zeros(trajectory_length)
            for t in range(0, trajectory_length):
                G_t = 0.0
                for idx, j in enumerate(range(t, trajectory_length)):
                    G_t += (gamma**idx)*rewards[j]*(1.0-dones[j])
                discounted_rewards[t] = G_t
            discounted_rewards = (discounted_rewards-np.mean(discounted_rewards))/(np.std(discounted_rewards)+1e-8)
            group_reward       = discounted_rewards[0]
            if debug:
                #print(f"GROUP-{group_member_id} REWARD IS {group_reward}, STEPS {trajectory_length}")
                pass
            break
    return group_member_id, group_reward, step, states, actions


def rollout_group(actor_model_params, epsilon): # Added epsilon parameter
    group_rewards        = np.zeros(shape=(group_size,)                         , dtype=np.float32)
    group_lengths        = np.zeros(shape=(group_size,)                         , dtype=np.float32)
    group_states         = np.zeros(shape=(group_size,)+(max_steps,)+state_shape, dtype=np.float32)
    group_actions        = np.zeros(shape=(group_size,)+(max_steps,            ), dtype=np.int32  )
    for group_member_id in range(group_size):
        member_id, group_reward, trajectory_length, states, actions = rollout_trajectory(
            group_member_id=group_member_id,
            actor_model_params=actor_model_params,
            epsilon=epsilon # Pass epsilon to rollout_trajectory
            )
        group_rewards       [member_id   ] = group_reward
        group_lengths       [member_id   ] = trajectory_length
        group_states        [member_id, :] = states
        group_actions       [member_id   ] = actions
    group_advantages = np.array((group_rewards - np.mean(group_rewards)) / max(np.std(group_rewards), np.finfo(np.float32).eps))
    return group_lengths, group_states, group_actions, group_advantages


epsilon = epsilon_start 

try:
    for episode in range(num_episodes):
        # Decay epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        # Rollout multiple trajectories for GRPO advantage extractions, passing epsilon
        group_lengths, group_states, group_actions, group_advantages = rollout_group(actor_model_params=actor_model_params, epsilon=epsilon) 

        # Training on rollout data with advantages
        for member_id in range(group_size):
            states     = group_states    [member_id]
            actions    = group_actions   [member_id]
            advantage  = group_advantages[member_id]
            actor_optimizer_state, actor_model_params, _ = backpropagate_actor(
                actor_optimizer_state,
                actor_model_params   ,
                (
                    jnp.asarray(states    , dtype=jnp.float32),
                    jnp.asarray(actions   , dtype=jnp.int32  ),
                    jnp.asarray(advantage , dtype=jnp.float32)
                )
            )
            pass

        if episode%play_frequency==0 and debug_render==True:
            state, info = env.reset()
            state = np.array(state, dtype=np.float32)
            states, actions, rewards, dones = [], [], [], []
            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state]))
                action_probabilities  = np.array(action_probabilities[0])
                action                = np.random.choice(n_actions, p=action_probabilities)
                next_state, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)
                states .append(state    )
                actions.append(action   )
                rewards.append(reward   )
                dones  .append(int(done))
                state = next_state

                env.render()

                if done:
                    episode_length = len(rewards)
                    discounted_rewards = np.zeros(episode_length)
                    for t in range(0, episode_length):
                        G_t = 0
                        for idx, j in enumerate(range(t, episode_length)):
                            G_t = G_t + (gamma**idx)*rewards[j]*(1-dones[j])
                        discounted_rewards[t] = G_t
                    discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
                    discounted_rewards = discounted_rewards / (np.std(discounted_rewards)+1e-5)
                    actor_optimizer_state, actor_model_params, _  = backpropagate_actor(
                        actor_optimizer_state,
                        actor_model_params   ,
                        (
                            jnp.asarray(states            , dtype=jnp.float32),
                            jnp.asarray(actions           , dtype=jnp.int32  ),
                            jnp.asarray(discounted_rewards, dtype=jnp.float32)
                        )
                    )
                    print(f"Episode {episode}, reward : {round(np.sum(rewards), 2)}, epsilon: {epsilon:.2f}")
                    break
finally:
    env.close()
    for env_item in env_array:
        env_item.close()