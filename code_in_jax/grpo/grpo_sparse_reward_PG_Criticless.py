#
# GRPO CRITIC-LESS WITH SPARSE REWARD AND EPSILON EXPLORATION
#
# Зорилго
# Энэхүү код нь CartPole тоглоомыг Sparse Reward буюу шагнал нь зөвхөн амжилттай 
# дууссан үед л өгөгдөх хүндрэлтэй нөхцөлд GRPO алгоритмаар сургах туршилт юм
#
# Архитектур
# Backbone нь 2 давхаргат Dense буюу бүрэн холбоот мэдрэлийн сүлжээ
# Critic модель ашиглахгүй буюу санах ойн хэмнэлттэй REINFORCE загвар
#
# Аргачлал
# Sparse Reward Wrapper
# Шагнал нь алхам бүрт өгөгдөхгүй бөгөөд зөвхөн 90-ээс дээш алхам тэсэж чадвал 
# төгсгөлд нь 1.0 оноо өгөх буюу LLM-ийн reasoning даалгавартай ижил логик
#
# Epsilon-Greedy Exploration
# Sparse reward үед анхны амжилтыг хурдан олохын тулд санамсаргүй үйлдлүүдийг 
# тодорхой магадлалтайгаар гүйцэтгэх буюу орчныг танин мэдэх аргачлал
#
# DeepSeek-style Outcome Group Advantage
# Бүлгийн үр дүнгүүдээс дунджийг нь хасаж Advantage тооцох боловч 
# Standard Deviation-д хуваахгүйгээр шууд утгаар нь ашиглах
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
group_size              = 8       # Нэг update хийхэд ашиглах Rollout-ийн тоо
max_steps               = 500

# Sparse Reward тохиргоо
sparse_reward_threshold = 90      # Шагнал авахын тулд тэсэх ёстой алхмын тоо
sparse_reward_value     = 1.0

# Exploration тохиргоо
epsilon_start           = 1.0     # Анхны санамсаргүй байдал
epsilon_end             = 0.01
epsilon_decay_episodes  = num_episodes / 2

entropy_coefficient     = 0.01    # Тархалтын баялаг байдлыг хангах


# ENVIRONMENT WRAPPER (Орчны өөрчлөлт)

class SparseCartPoleEnv(gym.Env):
    """
    Шагнал алхам бүрт биш зөвхөн заасан босгыг давсан үед өгөгдөх орчин
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
        done_boundary = bool(terminated or truncated)

        # Sparse reward logic буюу зөвхөн төгсгөлд нь амжилтыг үнэлэх
        sparse_reward = 0.0
        if done_boundary and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value

        return np.array(observation, dtype=np.float32), float(sparse_reward), terminated, truncated, info

    def render(self, mode='human'): return self.env.render()
    def close(self): self.env.close()


# MODEL ARCHITECTURE (Моделийн бүтэц)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # State input-ийг боловсруулах давхаргууд
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        # Action Probability буюу үйлдлийн магадлалыг гаргах
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)


# INITIALIZATION (Эхлүүлэлт)

env_array   = [SparseCartPoleEnv(render_mode=None, sparse_reward_threshold=sparse_reward_threshold) for _ in range(group_size)]
env         = SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=sparse_reward_threshold)

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

# Моделийн параметрүүд болон Optimizer
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

actor_optimizer_def    = optax.adam(learning_rate)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)


# JAX COMPILATION (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, props):
    # props[0] - states, props[1] - actions, props[2] - advantages
    def loss_fn(params):
        action_probas = actor_module.apply({'params': params}, props[0])
        # Сонгосон action-ийн магадлалыг салгах
        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        # Critic-less GRPO буюу Advantage-ийг гаднаас шууд оруулж ирэх
        advantages_sg = jax.lax.stop_gradient(props[2])
        # Exploration-ийг дэмжих Entropy тооцоолол
        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        # Policy Gradient Loss буюу REINFORCE алгоритмын үндсэн функц
        pg_loss       = -jnp.mean(logp * advantages_sg)
        ent_bonus     =  jnp.mean(entropies)

        loss          = pg_loss - entropy_coefficient * ent_bonus
        return loss, (pg_loss, ent_bonus)

    (loss, (pg_l, ent_b)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss, pg_l, ent_b


# ROLLOUT LOGIC (Өгөгдөл цуглуулах)

def rollout_trajectory(group_member_id, actor_model_params, seed, epsilon):
    """
    Нэг Agent-ийн бүтэн тоглолтыг гүйцэтгэх функц
    """
    env_item       = env_array[group_member_id]
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards = [], [], []
    step           = 0

    for _ in range(max_steps):
        # Epsilon-Greedy Exploration буюу санамсаргүй эсвэл моделийн дагуу үйлдэл хийх
        if random.random() < float(epsilon):
            action = env_item.action_space.sample()
        else:
            action_probs = actor_inference(actor_model_params, jnp.asarray([state]))
            action       = np.random.choice(n_actions, p=np.array(action_probs[0]))

        next_state, reward, terminated, truncated, _ = env_item.step(int(action))
        
        states .append(state        )
        actions.append(int  (action))
        rewards.append(float(reward))

        step += 1
        state = np.array(next_state, dtype=np.float32)
        if terminated or truncated: break

    # Outcome буюу эцсийн үр дүнг тооцох 
    total_reward = float(np.sum(rewards)) if step > 0 else 0.0
    return group_member_id, total_reward, step, np.array(states), np.array(actions), np.array(rewards)


def rollout_group(actor_model_params, seed, epsilon):
    """
    GRPO бүлгээр Rollout хийж Advantage тооцох
    """
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_rewards_seq = [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r = rollout_trajectory(member_id, actor_model_params, seed, epsilon)
        group_rewards    .append(g_reward)
        group_lengths    .append(length  )
        group_states     .append(s       )
        group_actions    .append(a       )
        group_rewards_seq.append(r       )

    # GRPO-style Advantage буюу бүлгийн дунджаар төвлөрүүлж Advantage тооцох
    group_mean_reward = float(np.mean(group_rewards))
    group_adv_scalar  = (np.array(group_rewards) - group_mean_reward)

    # Туршилтын алхам бүрт Advantage утгыг хуулж өгөх
    group_advantages  = []
    for member_id in range(group_size):
        adv_trj = np.full(shape=(group_lengths[member_id],), fill_value=float(group_adv_scalar[member_id]), dtype=np.float32)
        group_advantages.append(adv_trj)

    return group_mean_reward, group_lengths, group_states, group_actions, group_rewards_seq, group_advantages


# MAIN TRAINING LOOP (Үндсэн сургалт)

epsilon = epsilon_start

try:
    global_step = 0
    print("\n=== STARTING GRPO SPARSE REWARD TRAINING ===\n")

    for episode in range(num_episodes):
        # Epsilon decay буюу хугацаа өнгөрөх тусам санамсаргүй байдлыг багасгах
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        # Өгөгдөл цуглуулах
        g_mean, g_lens, g_states, g_actions, g_rewards_seq, g_advs = rollout_group(actor_model_params, episode, epsilon)

        # Batch болгож нэгтгэх
        flat_states  = np.concatenate(g_states ) if g_states  else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions) if g_actions else np.zeros((0,))
        flat_advs    = np.concatenate(g_advs   ) if g_advs    else np.zeros((0,))

        actor_loss, pg_loss, ent_bonus = 0.0, 0.0, 0.0

        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])
            # Моделийг шинэчлэх алхам
            actor_optimizer_state, actor_model_params, actor_loss, pg_loss, ent_bonus = backpropagate_actor(
                actor_optimizer_state, actor_model_params, 
                (jnp.asarray(flat_states), jnp.asarray(flat_actions, dtype=jnp.int32), jnp.asarray(flat_advs))
            )

        # Явцыг хэвлэх
        if debug and (episode % 10 == 0):
            # Success Rate буюу шагнал авсан туршилтуудын хувь
            group_success = float(np.mean([1.0 if (np.sum(r) > 0.0) else 0.0 for r in g_rewards_seq]))
            print(f"Ep {episode:6d} | Steps {global_step:9d} | GroupMeanR {g_mean:9.4f} | Succ {group_success:6.3f} | Eps {epsilon:6.3f} | ActL {float(actor_loss):8.4f}")

        # Visual Validation буюу явцыг нүдээр харах
        if episode % play_frequency == 0 and debug_render:
            state, info = env.reset(seed=int(episode))
            total_r, steps = 0, 0
            while True:
                probs = actor_inference(actor_model_params, jnp.asarray([state]))
                action = np.random.choice(n_actions, p=np.array(probs[0]))
                state, reward, terminated, truncated, _ = env.step(int(action))
                total_r += reward
                steps += 1
                env.render()
                if terminated or truncated:
                    print(f"   >> Visual Test | Episode {episode} | SparseReward: {total_r} | Steps: {steps}")
                    break

finally:
    env.close()
    for e in env_array: e.close()