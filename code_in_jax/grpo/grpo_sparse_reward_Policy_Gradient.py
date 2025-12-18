#
# GRPO WITH A2C CRITIC BASELINE AND SPARSE REWARD
#
# Зорилго
# Энэхүү код нь CartPole орчинд шагнал ховор өгөгдөх буюу Sparse Reward нөхцөлд 
# GRPO алгоритмыг A2C Critic Baseline аргачлалтай хослуулан сургах явдал юм
#
# Архитектур
# Actor Network нь үйлдлийн магадлалын тархалтыг гаргах MLP бүтэцтэй
# Critic Network нь тухайн төлөвийн ирээдүйн үнэ цэнэ буюу Value function таамаглана
#
# Сургалтын арга барил
# DeepSeek V3.2 загварын Group Advantage тооцоолол буюу группийн дундаж үр дүнгээр 
# төвлөрүүлж түүнээс Critic моделийн утгыг хасах замаар Variance бууруулах
#
# Exploration
# Sparse Reward нөхцөлд амжилтыг хурдан олохын тулд Epsilon Greedy болон 
# Stochastic Policy Sampling хослуулан ашигласан
#
# Тохиргоо
# JAX санах ойн менежмент болон Auto-grad ашиглан CPU болон GPU дээр ажиллана
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


# CONFIGURATION (Сургалтын тохиргоо)

debug_render            = True
debug                   = True
play_frequency          = 30
num_episodes            = 20000
learning_rate           = 0.0005
gamma                   = 0.99
env_name                = "CartPole-v1"
group_size              = 8       # Нэг update алхамд орох Rollout-ийн тоо
max_steps               = 500

# Sparse Reward босго буюу амжилтад тооцох алхмын тоо
sparse_reward_threshold = 90
sparse_reward_value     = 1.0

# Epsilon Decay тохиргоо
epsilon_start           = 1.0
epsilon_end             = 0.01
epsilon_decay_episodes  = num_episodes / 2

entropy_coefficient     = 0.01    # Exploration дэмжих жин


# ENVIRONMENT WRAPPER (Орчны өөрчлөлт)

class SparseCartPoleEnv(gym.Env):
    """
    Шагнал зөвхөн тоглоомын төгсгөлд босго давсан үед өгөгдөх Sparse Reward орчин
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
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
        done_boundary       = bool(terminated or truncated)

        # Sparse reward логик буюу зөвхөн заасан хугацаанд тэссэн бол шагнал өгөх
        sparse_reward = 0.0
        if done_boundary and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value

        return np.array(observation, dtype=np.float32), float(sparse_reward), terminated, truncated, info

    def render(self, mode='human'): return self.env.render()
    def close(self): self.env.close()


# MODEL DEFINITION (Сүлжээний бүтэц)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Үйлдлийн магадлалын тархалт гаргах MLP сүлжээ
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        # State-ийн үнэ цэнийг таамаглах Baseline сүлжээ
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        return nn.Dense(features=1)(x)


# INITIALIZATION (Эхлүүлэлт)

env_array   = [SparseCartPoleEnv(render_mode=None, sparse_reward_threshold=sparse_reward_threshold) for _ in range(group_size)]
env         =  SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=sparse_reward_threshold)

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

# Моделийн параметрүүд болон Optimizer тохиргоо
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


# JAX COMPILATION (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x).squeeze(-1)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # props[0] - states буюу төлөвүүд, props[1] - returns буюу бодит үр дүн
    def loss_fn(params):
        values = critic_module.apply({'params': params}, props[0]).squeeze(-1)
        # MSE Loss буюу Critic моделийн таамаглалын алдаа
        td     = props[1] - values
        return jnp.mean(jnp.square(td))
    
    loss, grads = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_opt_state = critic_optimizer_def.update(grads, optimizer_state, critic_model_params)
    new_params = optax.apply_updates(critic_model_params, updates)
    return new_opt_state, new_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, critic_model_params, props):
    # props[0] - states, props[1] - actions, props[2] - returns, props[3] - group_mean_return
    def loss_fn(params):
        action_probas = actor_module.apply({'params': params}, props[0])
        # Сонгосон action-ийн log probability тооцох
        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        # Critic Baseline утгыг салгаж авах буюу stop_gradient ашиглах
        values        = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]).squeeze(-1))

        # DeepSeek-style GRPO Advantage буюу группийн дунджаар төвлөрүүлж Critic Baseline хасах
        advantages    = (props[2] - props[3]) - values

        # Exploration дэмжих Entropy тооцоолол
        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        # Policy Gradient Loss буюу REINFORCE алгоритмын үндсэн функц
        loss          = -jnp.mean(logp * jax.lax.stop_gradient(advantages) + entropy_coefficient * entropies)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss


# ROLLOUT LOGIC (Өгөгдөл цуглуулах)

def compute_returns(rewards, done_terms, bootstrap):
    # Discounted Returns буюу ирээдүйн шагналын нийлбэрийг тооцох
    T, G = len(rewards), float(bootstrap)
    returns = np.zeros(T, dtype=np.float32)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns


def rollout_trajectory(group_member_id, actor_model_params, critic_model_params, seed, epsilon):
    # Нэг Agent-ийн бүтэн тоглолтыг гүйцэтгэх
    env_item       = env_array[group_member_id]
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards, done_terms = [], [], [], []
    step           = 0
    last_state, last_truncated, last_terminated = state, False, False

    for _ in range(max_steps):
        # Epsilon-Greedy Exploration буюу санамсаргүй эсвэл моделийн дагуу үйлдэл хийх
        if random.random() < float(epsilon):
            action = env_item.action_space.sample()
        else:
            action_probs = actor_inference(actor_model_params, jnp.asarray([state]))
            action       = np.random.choice(n_actions, p=np.array(action_probs[0]))

        next_state, reward, terminated, truncated, _ = env_item.step(int(action))
        
        states    .append(state            )
        actions   .append(int(action      ))
        rewards   .append(float(reward    ))
        done_terms.append(float(terminated))

        step          += 1
        state          = np.array(next_state, dtype=np.float32)
        last_state     = state
        last_truncated, last_terminated = bool(truncated), bool(terminated)

        if terminated or truncated: break

    # Truncated болсон үед ирээдүйн утгыг Critic-ээр таамаглаж Bootstrap хийх
    bootstrap = 0.0
    if last_truncated and (not last_terminated):
        bootstrap = float(critic_inference(critic_model_params, jnp.asarray([last_state]))[0])

    returns_seq = compute_returns(rewards, done_terms, bootstrap)
    # Группын Advantage тооцоонд ашиглах Outcome утга
    group_reward = float(returns_seq[0]) if step > 0 else 0.0

    return group_member_id, group_reward, step, np.array(states), np.array(actions), returns_seq


def rollout_group(actor_model_params, critic_model_params, seed, epsilon):
    # Бүлгээр Rollout хийж GRPO-д зориулсан өгөгдөл цуглуулах
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_returns = [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r = rollout_trajectory(member_id, actor_model_params, critic_model_params, seed, epsilon)
        group_rewards.append(g_reward)
        group_lengths.append(length  )
        group_states .append(s       )
        group_actions.append(a       )
        group_returns.append(r       )

    # Группын дундаж үр дүн буюу GRPO-ийн гол Baseline утга
    group_mean_reward = float(np.mean(group_rewards))

    return group_mean_reward, group_lengths, group_states, group_actions, group_returns


# TRAINING LOOP (Үндсэн сургалт)

epsilon = epsilon_start

try:
    global_step = 0
    print("\n=== STARTING GRPO A2C BASELINE SPARSE TRAINING ===\n")

    for episode in range(num_episodes):
        # Epsilon decay буюу хугацаа өнгөрөх тусам санамсаргүй байдлыг багасгах
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        # Өгөгдөл цуглуулах
        g_mean, g_lens, g_states, g_actions, g_returns = rollout_group(actor_model_params, critic_model_params, episode, epsilon)

        # Batch болгож нэгтгэх буюу variable length түүхүүдийг нийлүүлэх
        flat_states  = np.concatenate(g_states ) if g_states  else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions) if g_actions else np.zeros((0,))
        flat_returns = np.concatenate(g_returns) if g_returns else np.zeros((0,))

        actor_loss, critic_loss = 0.0, 0.0

        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])

            # Critic шинэчлэх буюу Value function сургах алхам
            critic_optimizer_state, critic_model_params, critic_loss = backpropagate_critic(
                critic_optimizer_state, critic_model_params, 
                (jnp.asarray(flat_states), jnp.asarray(flat_returns))
            )

            # Actor шинэчлэх буюу Policy Gradient алгоритмын алхам
            actor_optimizer_state, actor_model_params, actor_loss = backpropagate_actor(
                actor_optimizer_state, actor_model_params, critic_model_params,
                (jnp.asarray(flat_states), jnp.asarray(flat_actions, dtype=jnp.int32), jnp.asarray(flat_returns), jnp.asarray(g_mean))
            )

        # Явцыг хэвлэх
        if debug and (episode % 10 == 0):
            # Success Rate буюу шагнал авсан туршилтуудын хувь хэмжээ
            success_count = np.mean([1.0 if r[0] > 0.0 else 0.0 for r in g_returns if len(r) > 0])
            print(f"Ep {episode:6d} | Steps {global_step:9d} | GroupMeanR {g_mean:9.4f} | Succ {success_count:6.3f} | Eps {epsilon:6.3f}")

        # Visual Validation буюу явцыг нүдээр харах
        if episode % play_frequency == 0 and debug_render:
            state, info = env.reset(seed=int(episode))
            total_r, steps = 0, 0
            while True:
                probs = actor_inference(actor_model_params, jnp.asarray([state]))
                action = np.random.choice(n_actions, p=np.array(probs[0]))
                state, reward, terminated, truncated, _ = env.step(int(action))
                total_r += reward
                steps   += 1
                env.render()
                if terminated or truncated:
                    print(f"   >> Visual Test | Episode {episode} | SparseReward: {total_r} | Steps: {steps} | Epsilon: {epsilon:.2f}")
                    break

finally:
    env.close()
    for e in env_array: e.close()