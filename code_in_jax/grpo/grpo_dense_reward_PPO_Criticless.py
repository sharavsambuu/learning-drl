#
# CRITIC-LESS GRPO (PPO-STYLE)
#
# Зорилго
# Энэхүү код нь Critic сүлжээ ашиглахгүйгээр PPO буюу Proximal Policy Optimization 
# аргачлалыг GRPO алгоритмтай хослуулан CartPole тоглоомыг сургах туршилт юм
#
# Архитектур
# Actor Network нь энгийн MLP бүтэцтэй
# Reference Network буюу сургалтын явцад хэт хазайхаас сэргийлэх хуучин модель
# Critic байхгүй тул Value Function ашиглахгүй
#
# Аргачлал
# GRPO with PPO Clipping
# Уламжлалт GRPO дээр PPO-ийн Ratio Clipping нэмж сургалтыг тогтворжуулна
#
# Outcome-Based Advantage (DeepSeek-style)
# Бүлгийн дундаж онооноос хэр зөв буюу Standard Deviation хазайлттай байгаагаар
# Advantage тооцож түүнийгээ бүх алхамд ижил жинтэйгээр хуваарилна
#
# KL Regularization
# Reference модельтой харьцуулан хэт өөрчлөгдөхөөс сэргийлж KL Divergence торгууль тооцно
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


# CONFIGURATION

debug_render        = True
debug               = True
play_frequency      = 30        # Хэдэн episode тутамд render хийж шалгах
num_episodes        = 10000

learning_rate       = 0.001
gamma               = 0.99
env_name            = "CartPole-v1"

group_size          = 6         # Нэг update хийхэд цуглуулах туршилтын тоо
max_steps           = 500       # CartPole v1 ийн дээд хязгаар

# PPO болон Regularization тохиргоо
clip_epsilon        = 0.2       # Policy өөрчлөлтийг хязгаарлах хэмжээ
entropy_coefficient = 0.01      # Сониуч байдлыг дэмжих
kl_beta             = 0.02      # Reference модельоос зөрвөл өгөх шийтгэл

epochs_per_update   = 4         # Нэг цуглуулсан өгөгдөл дээр хэдэн удаа давтаж сурах
mini_batch_size     = 256       # Update хийх багцын хэмжээ

max_grad_norm       = 0.5       # Gradient Exploding асуудлаас сэргийлэх
use_std_advantage   = True      # Advantage тооцоход std ашиглах эсэх
ref_update_freq     = 10        # Reference моделийг шинэчлэх давтамж


# MODEL ARCHITECTURE

class ActorNetwork(nn.Module):
    n_actions: int
    
    @nn.compact
    def __call__(self, x):
        # State боловсруулах давхаргууд
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Action Probability гаргах хэсэг
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)


# INITIALIZATION

env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
env         = gym.make(env_name, render_mode='human')

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

rng                    = jax.random.PRNGKey(42)

# Моделийн параметрүүдийг үүсгэх
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape, dtype=jnp.float32)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

# Reference Params буюу KL Divergence тооцоход ашиглах хуулбар
actor_ref_params       = actor_params

actor_optimizer_def    = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adam(learning_rate)
)
actor_optimizer_state  = actor_optimizer_def.init(actor_model_params)


# JAX HELPER FUNCTIONS

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def logprob_from_probs(action_probas, actions):
    # Сонгогдсон action ийн log probability тооцох
    probs = jnp.take_along_axis(action_probas, actions[:, None], axis=1).squeeze(1)
    probs = jnp.clip(probs, 1e-8, 1.0)
    return jnp.log(probs)

@jax.jit
def entropy_from_probs(action_probas):
    # Entropy буюу тархалтын тодорхой бус байдлыг хэмжих
    return -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1).mean()

@jax.jit
def kl_from_probs(p_new, p_ref):
    # KL Divergence буюу хоёр тархалтын зөрүүг хэмжих
    return jnp.sum(p_new * (jnp.log(p_new + 1e-8) - jnp.log(p_ref + 1e-8)), axis=1).mean()

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, actor_ref_params, props):
    # props[0] -> states     (Batch, StateDim)
    # props[1] -> actions    (Batch,)
    # props[2] -> old_logp   (Batch,) - Өгөгдөл цуглуулах үеийн магадлал
    # props[3] -> advantages (Batch,)
    
    def loss_fn(params):
        action_probas_new = actor_module.apply({'params': params}, props[0])
        logp_new          = logprob_from_probs(action_probas_new, props[1])

        # PPO Ratio буюу шинэ болон хуучин магадлалын харьцаа
        ratio             = jnp.exp(logp_new - props[2])
        ratio_clipped     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        advantages_sg     = jax.lax.stop_gradient(props[3])

        # PPO Loss буюу Clipping хийж хэт огцом өөрчлөлтөөс сэргийлэх
        pg_loss1          = -advantages_sg * ratio
        pg_loss2          = -advantages_sg * ratio_clipped
        pg_loss           = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        entropy           = entropy_from_probs(action_probas_new)

        # Reference Model -той харьцуулж KL Penalty тооцох
        action_probas_ref = jax.lax.stop_gradient(
            actor_module.apply({'params': actor_ref_params}, props[0])
        )
        kl                = kl_from_probs(action_probas_new, action_probas_ref)

        # Нийт Loss функц
        total_loss        = pg_loss + kl_beta * kl - entropy_coefficient * entropy
        return total_loss, (pg_loss, kl, entropy)

    (loss, (pg_loss, kl, entropy)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_optimizer_state = actor_optimizer_def.update(gradients, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    
    return new_optimizer_state, new_params, loss, pg_loss, kl, entropy


# ROLLOUT LOGIC

def compute_returns(rewards, done_terms, bootstrap):
    """
    Rewards to Returns буюу ирээдүйн шагналын нийлбэрийг тооцох
    """
    T       = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G       = float(bootstrap)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns


def rollout_trajectory(group_member_id, actor_old_params, seed):
    """
    Нэг Agent ийн бүтэн тоглолтыг гүйцэтгэх функц
    """
    env_item       = env_array[group_member_id]

    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states         = np.zeros(shape=(max_steps,) + state_shape, dtype=np.float32)
    actions        = np.zeros(shape=(max_steps,), dtype=np.int32  )
    rewards        = np.zeros(shape=(max_steps,), dtype=np.float32)
    done_terms     = np.zeros(shape=(max_steps,), dtype=np.float32)
    old_logps      = np.zeros(shape=(max_steps,), dtype=np.float32)

    step           = 0

    for _ in range(max_steps):
        # Inference хийх
        action_probabilities = actor_inference(actor_old_params, jnp.asarray([state], dtype=jnp.float32))
        action_probabilities = np.array(action_probabilities[0], dtype=np.float32)

        # Action sampling
        action               = np.random.choice(n_actions, p=action_probabilities)
        prob_a               = float(np.clip(action_probabilities[int(action)], 1e-8, 1.0))
        old_logp             = float(np.log(prob_a))

        next_state, reward, terminated, truncated, info = env_item.step(int(action))

        done_boundary = bool(terminated or truncated)
        done_term     = bool(terminated)

        next_state    = np.array(next_state, dtype=np.float32)

        # Түүхийг хадгалах
        states    [step, :] = state
        actions   [step   ] = int  (action   )
        rewards   [step   ] = float(reward   )
        done_terms[step   ] = float(done_term)
        old_logps [step   ] = float(old_logp )

        step += 1
        state = next_state

        if done_boundary: break

    trajectory_length = step
    bootstrap         = 0.0

    returns = compute_returns(
        rewards    [:trajectory_length],
        done_terms [:trajectory_length],
        bootstrap
    )

    total_reward = float(np.sum(rewards[:trajectory_length])) if trajectory_length > 0 else 0.0

    return group_member_id, total_reward, trajectory_length, states, actions, returns, old_logps


def rollout_group(actor_old_params, seed):
    """
    GRPO Group Rollout буюу олон хувилбарыг зэрэг туршиж өгөгдөл цуглуулах
    """
    group_total_rewards = np.zeros(shape=(group_size,), dtype=np.float32)
    group_lengths       = np.zeros(shape=(group_size,), dtype=np.int32  )

    group_states        = []
    group_actions       = []
    group_returns       = []
    group_old_logps     = []

    for group_member_id in range(group_size):
        member_id, total_reward, trajectory_length, states, actions, returns, old_logps = rollout_trajectory(
            group_member_id     = group_member_id,
            actor_old_params    = actor_old_params,
            seed                = seed
        )

        group_total_rewards[member_id] = float(total_reward)
        group_lengths      [member_id] = int  (trajectory_length)

        group_states   .append(states   [:trajectory_length])
        group_actions  .append(actions  [:trajectory_length])
        group_returns  .append(returns)
        group_old_logps.append(old_logps[:trajectory_length])

    # Advantage Calculation DeepSeek Style
    group_mean_reward = float(np.mean(group_total_rewards))
    group_std_reward  = float(np.std (group_total_rewards)) + 1e-8

    # Outcome-based advantage буюу эцсийн үр дүнгээр үнэлэх
    # Standard deviation-д хувааж scale хийх нь сургалтыг тогтворжуулна
    if use_std_advantage:
        group_adv_scalar = (group_total_rewards - group_mean_reward) / group_std_reward
    else:
        group_adv_scalar = (group_total_rewards - group_mean_reward)

    group_advantages = []
    for member_id in range(group_size):
        length  = int(group_lengths[member_id])
        adv_trj = np.full(shape=(length,), fill_value=float(group_adv_scalar[member_id]), dtype=np.float32)
        group_advantages.append(adv_trj)

    return group_mean_reward, group_std_reward, group_lengths, group_states, group_actions, group_returns, group_old_logps, group_advantages


# TRAINING LOOP

try:
    global_step = 0
    
    print("\n" + "="*50)
    print(f"  STARTING TRAINING: {env_name}")
    print(f"  Ref Update Freq: {ref_update_freq} | Use Std Adv: {use_std_advantage}")
    print("="*50 + "\n")

    for episode in range(num_episodes):

        # Reference Model Update хийх
        if (kl_beta > 0.0) and (episode % ref_update_freq == 0):
            actor_ref_params = actor_model_params

        # PPO ийн хувьд хуучин policy хэрэгтэй
        actor_old_params = actor_model_params

        group_mean_reward, group_std_reward, group_lengths, group_states, group_actions, group_returns, group_old_logps, group_advantages = rollout_group(
            actor_old_params = actor_old_params,
            seed             = episode
        )

        # Batch үүсгэх
        flat_states     = np.concatenate(group_states    , axis=0) if len(group_states    ) > 0 else np.zeros((0,) + state_shape, dtype=np.float32)
        flat_actions    = np.concatenate(group_actions   , axis=0) if len(group_actions   ) > 0 else np.zeros((0,), dtype=np.int32  )
        flat_old_logp   = np.concatenate(group_old_logps , axis=0) if len(group_old_logps ) > 0 else np.zeros((0,), dtype=np.float32)
        flat_advantages = np.concatenate(group_advantages, axis=0) if len(group_advantages) > 0 else np.zeros((0,), dtype=np.float32)

        actor_loss = 0.0
        pg_loss    = 0.0
        kl_loss    = 0.0
        entropy    = 0.0

        if flat_states.shape[0] > 0:
            batch_size  = int(flat_states.shape[0])
            global_step += batch_size

            states_j    = jnp.asarray(flat_states    , dtype=jnp.float32)
            actions_j   = jnp.asarray(flat_actions   , dtype=jnp.int32  )
            old_logp_j  = jnp.asarray(flat_old_logp  , dtype=jnp.float32)
            adv_j       = jnp.asarray(flat_advantages, dtype=jnp.float32)

            # PPO Epochs буюу нэг өгөгдлийг олон дахин ашиглаж сурах
            for _ in range(epochs_per_update):
                rng, perm_rng = jax.random.split(rng)
                indices = jax.random.permutation(perm_rng, jnp.arange(batch_size))

                # Mini-batch loop
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size

                    # JAX recompilation асуудлаас сэргийлж дутуу batch ийг хаях
                    if end > batch_size: continue

                    mb_idx = indices[start:end]

                    actor_optimizer_state, actor_model_params, actor_loss, pg_loss, kl_loss, entropy = backpropagate_actor(
                        actor_optimizer_state,
                        actor_model_params,
                        actor_ref_params,
                        (
                            states_j  [mb_idx],
                            actions_j [mb_idx],
                            old_logp_j[mb_idx],
                            adv_j     [mb_idx],
                        )
                    )

        # Лог хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:6d} | Steps {global_step:9d} | GroupMeanR {group_mean_reward:9.3f} | Std {group_std_reward:7.3f} | ActLoss {float(actor_loss):9.4f} | PG {float(pg_loss):9.4f} | KL {float(kl_loss):9.4f} | Ent {float(entropy):9.4f}")

        # Visual Validation
        if episode % play_frequency == 0 and debug_render == True:
            state, info = env.reset(seed=int(episode))
            state       = np.array(state, dtype=np.float32)
            rewards     = []

            while True:
                action_probabilities  = actor_inference(actor_model_params, jnp.asarray([state], dtype=jnp.float32))
                action_probabilities  = np.array(action_probabilities[0], dtype=np.float32)
                action                = np.random.choice(n_actions, p=action_probabilities)

                next_state, reward, terminated, truncated, info = env.step(int(action))
                done       = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)

                rewards.append(float(reward))
                state = next_state

                env.render()

                if done:
                    print(f"   >> Test Play Episode {episode}: Reward {sum(rewards)}")
                    break

finally:
    env.close()
    for env_item in env_array:
        env_item.close()