#
# PPO-STYLE GRPO WITH CRITIC BASELINE AND SPARSE REWARD EXPLORATION
#
# Зорилго
# Энэхүү код нь CartPole тоглоомын шагнал ховор өгөгдөх буюу Sparse Reward нөхцөлд 
# PPO-style Clipping болон Critic Baseline бүхий GRPO алгоритмыг хэрэгжүүлэх явдал юм
#
# Архитектур
# Actor Network нь MLP бүтэцтэй бөгөөд үйлдлийн магадлалын тархалт гаргана
# Critic Network нь төлөвийн үнэ цэнийг таамаглаж Advantage тооцоололд Baseline болно
# Reference Model нь сургалтын явцад Policy-г хэт хазайхаас хамгаалж KL Penalty өгнө
#
# Аргачлал
# Epsilon-Greedy Mixture Exploration
# Behavior Policy буюу өгөгдөл цуглуулах үед санамсаргүй байдал болон Policy-г 
# хослуулах ингэхдээ PPO Ratio тооцоололд уг холимог тархалтын log probability-г ашиглах
#
# Outcome-Based Group Centering
# DeepSeek style буюу бүлгийн дундаж үр дүнгээр Advantage-ийг төвлөрүүлж 
# дээр нь Critic Baseline ашиглан Variance-ийг дээд зэргээр бууруулах
#
# Proper Truncation Handling
# Хугацаа дуусаж тоглоом зогсох үед ирээдүйн боломжит шагналыг Critic-ээр таамаглаж 
# Bootstrap хийх замаар Return тооцооллыг зөв гүйцэтгэх
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

# Sparse Reward тохиргоо
sparse_reward_threshold = 90      # Шагнал авахын тулд тэсэх ёстой алхмын тоо
sparse_reward_value     = 1.0

# Exploration тохиргоо
epsilon_start           = 1.0     # Анхны санамсаргүй байдлын хэмжээ
epsilon_end             = 0.01
epsilon_decay_episodes  = num_episodes / 2

# Regularization болон PPO тохиргоо
clip_epsilon            = 0.2     # Policy өөрчлөлтийг хязгаарлах Clipping утга
entropy_coefficient     = 0.01    # Сониуч байдлыг дэмжих жин
kl_beta                 = 0.02    # Reference модельтой ижил байхыг шаардах жин

epochs_per_update       = 4       # Багц өгөгдөл дээр давтан суралцах тоо
mini_batch_size         = 256     # Нэг удаагийн update-д орох өгөгдлийн хэмжээ

max_grad_norm           = 0.5     # Градиент хэт өсөхөөс сэргийлэх хязгаар


# ENVIRONMENT WRAPPER (Орчны тохируулга)

class SparseCartPoleEnv(gym.Env):
    """
    Шагнал алхам бүрт биш зөвхөн заасан босгыг давсан үед өгөгдөх Sparse Reward орчин
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

        # Sparse reward logic буюу зөвхөн төгсгөлд нь амжилтыг үнэлэх
        sparse_reward = 0.0
        if done_boundary and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value

        return np.array(observation, dtype=np.float32), float(sparse_reward), terminated, truncated, info

    def render(self, mode='human'): return self.env.render()
    def close(self): self.env.close()


# MODEL ARCHITECTURE (Мэдрэлийн сүлжээ)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Үйлдлийн магадлалын тархалт гаргах MLP бүтэц
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        # State-ийн утгыг таамаглах Baseline сүлжээ
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

rng         = jax.random.PRNGKey(42)

# Модель болон параметрүүдийг үүсгэх
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

# Reference Policy буюу хөлдөөсөн анхны модель
actor_ref_params       = actor_params

critic_module          = CriticNetwork()
critic_params          = critic_module.init(jax.random.PRNGKey(1), dummy_input)['params']
critic_model_params    = critic_params

# Optimizer болон Gradient Clipping тохиргоо
actor_optimizer_def    = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
critic_optimizer_def   = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))

actor_optimizer_state  = actor_optimizer_def .init(actor_model_params )
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)


# JAX COMPILATION (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x).squeeze(-1)

@jax.jit
def logprob_from_probs(action_probas, actions):
    # Сонгосон action-ийн магадлалыг салгаж log probability тооцох
    probs = jnp.take_along_axis(action_probas, actions[:, None], axis=1).squeeze(1)
    probs = jnp.clip(probs, 1e-8, 1.0)
    return jnp.log(probs)

@jax.jit
def entropy_from_probs(action_probas):
    # Тархалтын тодорхой бус байдлыг хэмжих
    return -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1).mean()

@jax.jit
def kl_from_probs(p_new, p_ref):
    # Reference модельтой харьцуулан KL Divergence зөрүүг хэмжих
    return jnp.sum(p_new * (jnp.log(p_new + 1e-8) - jnp.log(p_ref + 1e-8)), axis=1).mean()


# BACKPROPAGATION LOGIC (Сургалтын алхмууд)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # Critic сүлжээг MSE Loss ашиглан шинэчлэх алхам
    def loss_fn(params):
        values = critic_module.apply({'params': params}, props[0]).squeeze(-1)
        return jnp.mean(jnp.square(props[1] - values))
    
    loss, grads = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_opt_state = critic_optimizer_def.update(grads, optimizer_state, critic_model_params)
    new_params = optax.apply_updates(critic_model_params, updates)
    return new_opt_state, new_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, actor_ref_params, critic_model_params, props):
    # props -> states, actions, returns, old_logp_behavior, group_mean_return
    def loss_fn(params):
        action_probas_new = actor_module.apply({'params': params}, props[0])
        logp_new          = logprob_from_probs(action_probas_new, props[1])

        # PPO Ratio буюу шинэ болон Behavior Policy-ийн магадлалын харьцаа
        ratio             = jnp.exp(logp_new - props[3])
        ratio_clipped     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        # Critic Baseline утгыг салгаж авах буюу stop_gradient ашиглах
        values            = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]).squeeze(-1))

        # DeepSeek-style GRPO Advantage буюу бүлгийн дунджаар төвлөрүүлж Critic Baseline хасах
        advantages        = (props[2] - props[4]) - values
        advantages_sg     = jax.lax.stop_gradient(advantages)

        # Policy Gradient Loss тооцоолол
        pg_loss1          = -advantages_sg * ratio
        pg_loss2          = -advantages_sg * ratio_clipped
        pg_loss           = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Regularization буюу KL Penalty болон Entropy Bonus
        entropy           = entropy_from_probs(action_probas_new)
        action_probas_ref = jax.lax.stop_gradient(actor_module.apply({'params': actor_ref_params}, props[0]))
        kl                = kl_from_probs(action_probas_new, action_probas_ref)

        total_loss        = pg_loss + kl_beta * kl - entropy_coefficient * entropy
        return total_loss, (pg_loss, kl, entropy)

    (loss, (pg_l, kl_l, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss, pg_l, kl_l, ent


# ROLLOUT LOGIC (Өгөгдөл цуглуулах)

def compute_returns(rewards, done_terms, bootstrap):
    # Discounted Returns буюу ирээдүйн шагналын нийлбэрийг тооцох
    T, G = len(rewards), float(bootstrap)
    returns = np.zeros(T, dtype=np.float32)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns

def rollout_trajectory(group_member_id, actor_old_params, critic_model_params, seed, epsilon):
    # Нэг Agent-ийн бүтэн тоглолтыг гүйцэтгэх функц
    env_item       = env_array[group_member_id]
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards, done_terms, old_logps = [], [], [], [], []
    step           = 0
    last_state, last_truncated, last_terminated = state, False, False
    eps, uni_prob  = float(epsilon), 1.0 / float(n_actions)

    for _ in range(max_steps):
        # Inference буюу Policy магадлал гаргах
        action_probs_out = actor_inference(actor_old_params, jnp.asarray([state], dtype=jnp.float32))
        action_probs_np  = np.array(action_probs_out[0], dtype=np.float32)

        # Mixture Exploration буюу санамсаргүй үйлдэл болон Policy-г хослуулах
        if random.random() < eps:
            action = env_item.action_space.sample()
        else:
            action = np.random.choice(n_actions, p=action_probs_np)

        # PPO Ratio тооцоход ашиглах Behavior Policy-ийн бодит магадлалыг хадгалах
        pi_a     = float(np.clip(action_probs_np[int(action)], 1e-8, 1.0))
        beh_prob = float(eps * uni_prob + (1.0 - eps) * pi_a)
        old_logp = float(np.log(np.clip(beh_prob, 1e-8, 1.0)))

        next_state, reward, terminated, truncated, _ = env_item.step(int(action))
        
        states    .append(state            )
        actions   .append(int  (action    ))
        rewards   .append(float(reward    ))
        done_terms.append(float(terminated))
        old_logps .append(float(old_logp  ))

        step          += 1
        state          = np.array(next_state, dtype=np.float32)
        last_state     = state
        last_truncated, last_terminated = bool(truncated), bool(terminated)

        if terminated or truncated: break

    # Truncated үед ирээдүйн утгыг Critic-ээр таамаглаж Bootstrap хийх
    bootstrap = 0.0
    if last_truncated and (not last_terminated):
        bootstrap = float(critic_inference(critic_model_params, jnp.asarray([last_state], dtype=jnp.float32))[0])

    returns_seq = compute_returns(rewards, done_terms, bootstrap)
    # Группын Advantage тооцоонд ашиглах Outcome утга
    group_reward = float(returns_seq[0]) if step > 0 else 0.0

    return group_member_id, group_reward, step, np.array(states), np.array(actions), returns_seq, np.array(old_logps)


def rollout_group(actor_old_params, critic_model_params, seed, epsilon):
    # Бүлгээр Rollout хийж GRPO-д зориулсан өгөгдөл цуглуулах
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_returns, group_old_logps = [], [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r, lp = rollout_trajectory(member_id, actor_old_params, critic_model_params, seed, epsilon)
        group_rewards  .append(g_reward)
        group_lengths  .append(length  )
        group_states   .append(s       )
        group_actions  .append(a       )
        group_returns  .append(r       )
        group_old_logps.append(lp      )

    group_mean_reward = float(np.mean(group_rewards))
    return group_mean_reward, group_lengths, group_states, group_actions, group_returns, group_old_logps


# MAIN TRAINING LOOP (Үндсэн цикл)

epsilon = epsilon_start

try:
    global_step = 0
    print("\n=== STARTING PPO-STYLE GRPO WITH CRITIC BASELINE AND SPARSE REWARD ===\n")

    for episode in range(num_episodes):
        # Epsilon decay буюу санамсаргүй байдлыг хугацаа өнгөрөх тусам багасгах
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        # Одоогийн Policy-г сургалтын багц өгөгдөл цуглуулах behavior policy болгож хөлдөөх
        actor_old_params = actor_model_params

        # Өгөгдөл цуглуулах алхам
        g_mean, g_lens, g_states, g_actions, g_returns, g_old_lps = rollout_group(actor_old_params, critic_model_params, episode, epsilon)

        # Batch болгож нэгтгэх
        flat_states  = np.concatenate(g_states ) if g_states  else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions) if g_actions else np.zeros((0,))
        flat_returns = np.concatenate(g_returns) if g_returns else np.zeros((0,))
        flat_old_lps = np.concatenate(g_old_lps) if g_old_lps else np.zeros((0,))

        actor_loss, pg_loss, kl_loss, ent_bonus, critic_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        if flat_states.shape[0] > 0:
            batch_size = int(flat_states.shape[0])
            global_step += batch_size

            states_j, actions_j, returns_j = jnp.asarray(flat_states), jnp.asarray(flat_actions, dtype=jnp.int32), jnp.asarray(flat_returns)
            old_logp_j, mean_r_j = jnp.asarray(flat_old_lps), jnp.asarray(g_mean)

            # Critic шинэчлэх (Batch тутамд нэг удаа)
            critic_optimizer_state, critic_model_params, critic_loss = backpropagate_critic(
                critic_optimizer_state, critic_model_params, (states_j, returns_j)
            )

            # Actor шинэчлэх (Mini-batch ашиглан олон Epoch давтах)
            for _ in range(epochs_per_update):
                rng, perm_rng = jax.random.split(rng)
                indices = jax.random.permutation(perm_rng, jnp.arange(batch_size))

                for start in range(0, batch_size, mini_batch_size):
                    end = min(start + mini_batch_size, batch_size)
                    idx = indices[start:end]

                    actor_optimizer_state, actor_model_params, actor_loss, pg_loss, kl_loss, ent_bonus = backpropagate_actor(
                        actor_optimizer_state, actor_model_params, actor_ref_params, critic_model_params,
                        (states_j[idx], actions_j[idx], returns_j[idx], old_logp_j[idx], mean_r_j)
                    )

        # Явцыг лог болгон хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:6d} | Steps {global_step:9d} | GroupMeanR {g_mean:9.4f} | Eps {epsilon:6.3f} | ActL {float(actor_loss):8.4f} | CritL {float(critic_loss):8.4f}")

        # Visual Validation буюу явцыг нүдээр харах
        if episode % play_frequency == 0 and debug_render:
            state, info = env.reset(seed=int(episode))
            total_r, steps = 0, 0
            while True:
                probs = actor_inference(actor_model_params, jnp.asarray([state], dtype=jnp.float32))
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