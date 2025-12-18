#
# PPO-STYLE GRPO WITH SPARSE REWARD AND CURRICULUM LEARNING
#
# Зорилго
# Энэхүү код нь CartPole тоглоомыг Sparse Reward буюу шагнал ховор өгөгдөх нөхцөлд 
# Curriculum Learning буюу даалгаврын хүндрэлийг шат дараатай нэмэгдүүлэх аргаар 
# GRPO алгоритм ашиглан сургах явдал юм
#
# Архитектур
# Actor Network нь MLP бүтэцтэй бөгөөд Policy Gradient аргаар суралцана
# Reference Model нь сургалтын явцад Policy-г хэт хазайхаас сэргийлж KL Penalty өгнө
# Critic-less буюу Critic модель ашиглахгүйгээр санах ойн хэмнэлтийг хангана
#
# Аргачлал
# GRPO (Group Relative Policy Optimization) бүлэг доторх харьцуулалт
# PPO-style Clipping буюу Policy өөрчлөлтийн харьцааг хязгаарлах
# Curriculum Learning буюу амжилтын хувьд тулгуурлан Reward Threshold-ийг динамикаар өөрчлөх
# Sparse Reward Fix буюу gamma-г 1.0 болгож эрт бууж өгөх үйлдлийг засах
#
# Exploration
# Epsilon-Greedy Mixture буюу Uniform Random болон Policy sampling-ийг хослуулах
# Энэ нь Sparse Reward нөхцөлд анхны амжилтыг олоход маш чухал үүрэгтэй
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


# CONFIGURATION (Үндсэн тохиргоо)

debug_render            = True
debug                   = True
play_frequency          = 50
num_episodes            = 20000

learning_rate           = 0.0005
gamma                   = 1.0       # Sparse reward үед Discount factor-ийг 1 болгох
env_name                = "CartPole-v1"

group_size              = 16        # Rollout хийх бүлгийн хэмжээ
max_steps               = 500

sparse_reward_value     = 1.0

# Curriculum тохиргоо
start_threshold         = 25        # Анх эхлэх алхмын босго
max_threshold           = 495       # Эцсийн зорилтот босго
threshold_step_up       = 5         # Хүндрэлийг нэмэгдүүлэх алхам
threshold_step_down     = 5         # Хүндрэлийг бууруулах алхам

curr_window             = 30        # Амжилтын хувийг тооцох цонх
target_low              = 0.20      # Хүндрэлийг бууруулах доод хязгаар
target_high             = 0.80      # Хүндрэлийг нэмэх дээд хязгаар

# Exploration тохиргоо
epsilon_start           = 1.0
epsilon_end             = 0.05
epsilon_decay_episodes  = 2000

# Regularization тохиргоо
clip_epsilon            = 0.2
entropy_coefficient     = 0.02
kl_beta                 = 0.02

epochs_per_update       = 4
mini_batch_size         = 256

max_grad_norm           = 0.5
use_std_advantage       = True      # Advantage-ийг Scale хийх эсэх
ref_update_freq         = 10


# ENVIRONMENT WRAPPER (Орчны тохируулга)

class SparseCartPoleEnv(gym.Env):
    """
    Даалгаврын босгыг динамикаар өөрчлөх боломжтой Sparse Reward орчин
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode='human', sparse_reward_threshold=25, sparse_reward_value=1.0):
        super().__init__()
        self.env = gym.make(env_name, render_mode=render_mode if debug_render else None)
        self.sparse_reward_threshold = int(sparse_reward_threshold)
        self.sparse_reward_value     = float(sparse_reward_value)
        self.current_steps           = 0
        self.action_space            = self.env.action_space
        self.observation_space       = self.env.observation_space

    def set_threshold(self, new_threshold):
        # Curriculum learning-д зориулсан босго өөрчлөх функц
        self.sparse_reward_threshold = int(new_threshold)

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        observation, info  = self.env.reset(seed=seed, options=options)
        return np.array(observation, dtype=np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1
        done_boundary = bool(terminated or truncated)

        # Sparse reward логик буюу зөвхөн босго давсан үед шагнал өгөх
        sparse_reward = 0.0
        if done_boundary and self.current_steps >= self.sparse_reward_threshold:
            sparse_reward = self.sparse_reward_value

        return np.array(observation, dtype=np.float32), float(sparse_reward), terminated, truncated, info

    def render(self, mode='human'): return self.env.render()
    def close(self): self.env.close()


# MODEL DEFINITION (Мэдрэлийн сүлжээ)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Policy магадлалын тархалт гаргах MLP бүтэц
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)


# INITIALIZATION (Эхлүүлэлт)

current_threshold = int(start_threshold)

env_array   = [SparseCartPoleEnv(render_mode=None, sparse_reward_threshold=current_threshold) for _ in range(group_size)]
env         =  SparseCartPoleEnv(render_mode='human', sparse_reward_threshold=current_threshold)

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

rng         = jax.random.PRNGKey(42)

# Модель болон Optimizer тохиргоо
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

# Reference Model буюу хөлдөөсөн анхны Policy
actor_ref_params       = actor_params

actor_optimizer_def    = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)


# JAX COMPILATION (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def logprob_from_probs(action_probas, actions):
    # Сонгосон action-ийн log probability тооцох
    probs = jnp.take_along_axis(action_probas, actions[:, None], axis=1).squeeze(1)
    probs = jnp.clip(probs, 1e-8, 1.0)
    return jnp.log(probs)

@jax.jit
def entropy_from_probs(action_probas):
    return -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1).mean()

@jax.jit
def kl_from_probs(p_new, p_ref):
    # KL Divergence буюу Reference-ээс зөрөх зөрүүг хэмжих
    return jnp.sum(p_new * (jnp.log(p_new + 1e-8) - jnp.log(p_ref + 1e-8)), axis=1).mean()

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, actor_ref_params, props):
    # props -> states, actions, old_logp, advantages, kl_weight
    kl_weight_dynamic = props[4]

    def loss_fn(params):
        action_probas_new = actor_module.apply({'params': params}, props[0])
        logp_new          = logprob_from_probs(action_probas_new, props[1])

        # PPO Clipping буюу Policy өөрчлөлтийг хязгаарлах
        ratio             = jnp.exp(logp_new - props[2])
        ratio_clipped     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        advantages_sg     = jax.lax.stop_gradient(props[3])

        pg_loss1          = -advantages_sg * ratio
        pg_loss2          = -advantages_sg * ratio_clipped
        pg_loss           = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Regularization буюу KL Penalty болон Entropy Bonus
        entropy           = entropy_from_probs(action_probas_new)
        action_probas_ref = jax.lax.stop_gradient(actor_module.apply({'params': actor_ref_params}, props[0]))
        kl                = kl_from_probs(action_probas_new, action_probas_ref)

        total_loss        = pg_loss + kl_weight_dynamic * kl - entropy_coefficient * entropy
        return total_loss, (pg_loss, kl, entropy)

    (loss, (pg_l, kl_l, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss, pg_l, kl_l, ent


# ROLLOUT LOGIC (Өгөгдөл цуглуулах)

def compute_returns(rewards, done_terms, bootstrap):
    # Discounted Returns тооцоолох (gamma=1.0)
    T, G = len(rewards), float(bootstrap)
    returns = np.zeros(T, dtype=np.float32)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns

def rollout_trajectory(group_member_id, actor_old_params, seed, epsilon):
    # Нэг Agent-ийн бүтэн тоглолтыг гүйцэтгэх
    env_item = env_array[group_member_id]
    env_item.set_threshold(current_threshold)

    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards, done_terms, old_logps = [], [], [], [], []
    step           = 0
    eps, uni_prob  = float(epsilon), 1.0 / float(n_actions)

    for _ in range(max_steps):
        # Inference буюу Policy магадлал гаргах
        action_probs_out = actor_inference(actor_old_params, jnp.asarray([state], dtype=jnp.float32))
        action_probs_np  = np.array(action_probs_out[0], dtype=np.float32)

        # Behavior Policy буюу Mixture Sampling (Epsilon*Uniform + (1-Eps)*Policy)
        if random.random() < eps:
            action = env_item.action_space.sample()
        else:
            action = np.random.choice(n_actions, p=action_probs_np)

        # Behavior Log Probability хадгалах буюу PPO-ийн зөв Ratio тооцоход ашиглах
        pi_a     = float(np.clip(action_probs_np[int(action)], 1e-8, 1.0))
        beh_prob = float(eps * uni_prob + (1.0 - eps) * pi_a)
        old_logp = float(np.log(np.clip(beh_prob, 1e-8, 1.0)))

        next_state, reward, terminated, truncated, _ = env_item.step(int(action))

        states    .append(state            )
        actions   .append(int  (action    ))
        rewards   .append(float(reward    ))
        done_terms.append(float(terminated))
        old_logps .append(float(old_logp  ))

        step += 1
        state = np.array(next_state, dtype=np.float32)
        if terminated or truncated: break

    returns_seq = compute_returns(rewards, done_terms, 0.0)
    total_reward = float(np.sum(rewards)) if step > 0 else 0.0

    return group_member_id, total_reward, step, np.array(states), np.array(actions), returns_seq, np.array(old_logps)


def rollout_group(actor_old_params, seed, epsilon):
    # GRPO Бүлгээр Rollout хийж Advantage тооцох
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_returns, group_old_logps = [], [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r, lp = rollout_trajectory(member_id, actor_old_params, seed, epsilon)
        group_rewards  .append(g_reward)
        group_lengths  .append(length  )
        group_states   .append(s       )
        group_actions  .append(a       )
        group_returns  .append(r       )
        group_old_logps.append(lp      )

    group_mean_reward = float(np.mean(group_rewards))
    group_std_reward  = float(np.std (group_rewards)) + 1e-8

    # Outcome-based advantage буюу бүлгийн дунджаар төвлөрүүлж Scale хийх
    if use_std_advantage:
        group_adv_scalar = (np.array(group_rewards) - group_mean_reward) / group_std_reward
    else:
        group_adv_scalar = (np.array(group_rewards) - group_mean_reward)

    group_advantages = [np.full((group_lengths[i],), group_adv_scalar[i], dtype=np.float32) for i in range(group_size)]

    return group_mean_reward, group_std_reward, group_lengths, group_states, group_actions, group_returns, group_old_logps, group_advantages


# MAIN TRAINING LOOP (Үндсэн сургалт)

epsilon              = epsilon_start
has_succeeded_once   = False        # KL penalty-г идэвхжүүлэх эсэх
success_rate_history = []           # Curriculum-д зориулсан түүх

try:
    global_step = 0
    print("\n=== STARTING GRPO CURRICULUM SPARSE TRAINING ===\n")

    for episode in range(num_episodes):
        # Epsilon decay буюу хугацаа өнгөрөх тусам санамсаргүй байдлыг багасгах
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay_episodes)

        actor_old_params = actor_model_params

        # Өгөгдөл цуглуулах алхам
        g_mean, g_std, g_lens, g_states, g_actions, g_returns, g_old_lps, g_advs = rollout_group(actor_old_params, episode, epsilon)

        # Curriculum болон Success rate тооцоолол
        success_rate_history.append(float(g_mean))
        if len(success_rate_history) > int(curr_window): success_rate_history.pop(0)
        success_rate_avg = float(np.mean(success_rate_history)) if success_rate_history else 0.0

        if g_mean > 0.0: has_succeeded_once = True
        if has_succeeded_once and (episode % ref_update_freq == 0) and (success_rate_avg > 0.10):
            actor_ref_params = actor_old_params

        # Curriculum Control буюу хүндрэлийн түвшинг автоматаар тохируулах логик
        if len(success_rate_history) >= int(curr_window):
            if (success_rate_avg > target_high) and (current_threshold < max_threshold):
                current_threshold = int(min(max_threshold, current_threshold + threshold_step_up))
            elif (success_rate_avg < target_low) and (current_threshold > start_threshold):
                current_threshold = int(max(start_threshold, current_threshold - threshold_step_down))

        # Багц өгөгдөл бэлдэх
        flat_states  = np.concatenate(g_states ) if g_states  else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions) if g_actions else np.zeros((0,))
        flat_old_lps = np.concatenate(g_old_lps) if g_old_lps else np.zeros((0,))
        flat_advs    = np.concatenate(g_advs   ) if g_advs    else np.zeros((0,))

        actor_loss, pg_loss, kl_loss, ent_bonus = 0.0, 0.0, 0.0, 0.0
        # Анхны амжилтад хүртэл KL Penalty-г тэглэж Exploration-ийг дэмжих
        current_kl_beta = float(kl_beta) if has_succeeded_once else 0.0

        if flat_states.shape[0] > 0:
            batch_size = int(flat_states.shape[0])
            global_step += batch_size

            states_j, actions_j, old_logp_j = jnp.asarray(flat_states), jnp.asarray(flat_actions, dtype=jnp.int32), jnp.asarray(flat_old_lps)
            adv_j, kl_beta_j = jnp.asarray(flat_advs), jnp.asarray(current_kl_beta)

            # PPO Epochs буюу нэг цуглуулсан өгөгдөл дээр олон удаа суралцах
            for _ in range(epochs_per_update):
                rng, perm_rng = jax.random.split(rng)
                indices = jax.random.permutation(perm_rng, jnp.arange(batch_size))

                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    if end > batch_size: continue
                    idx = indices[start:end]

                    actor_optimizer_state, actor_model_params, actor_loss, pg_loss, kl_loss, ent_bonus = backpropagate_actor(
                        actor_optimizer_state, actor_model_params, actor_ref_params,
                        (states_j[idx], actions_j[idx], old_logp_j[idx], adv_j[idx], kl_beta_j)
                    )

        # Явцыг хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:5d} | Steps {global_step:8d} | Thr {current_threshold:3d} | SuccAvg {success_rate_avg:5.2f} | Eps {epsilon:5.2f} | ActL {float(actor_loss):8.4f}")

        # Visual Validation буюу явцыг нүдээр харах
        if episode % play_frequency == 0 and debug_render:
            env.set_threshold(current_threshold)
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
                    print(f"   >> Visual Test | Episode {episode} | SparseReward: {total_r} | Steps: {steps} | Thr: {current_threshold}")
                    break

finally:
    env.close()
    for e in env_array: e.close()