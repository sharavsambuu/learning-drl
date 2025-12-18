#
# PPO-STYLE GRPO WITH CRITIC BASELINE AND REFERENCE-KL
#
# Зорилго
# Энэхүү код нь GRPO алгоритмыг PPO-ийн Ratio Clipping болон A2C Critic Baseline 
# аргачлалтай хослуулан CartPole тоглоомыг сургах дэвшилтэт хувилбар юм
#
# Архитектур
# Actor Network нь MLP бүтэцтэй бөгөөд Action магадлалын тархалтыг гаргана
# Critic Network нь тухайн төлөвийн суурь утгыг таамаглаж Advantage тооцоход тусална
# Reference Policy буюу анхны моделийн хуулбар нь сургалтыг хэт хазайхаас хамгаална
#
# Аргачлал
# PPO Ratio Clipping буюу шинэ болон хуучин Policy-ийн харьцааг хязгаарлах замаар 
# сургалтын тогтвортой байдлыг хангах
#
# DeepSeek-style Group Centering буюу бүлгийн дундаж үр дүнгээр Advantage-ийг 
# төвлөрүүлж түүнээс Critic-ийн Baseline утгыг хасаж Variance-ийг бууруулах
#
# Proper Truncation Handling буюу тоглоом хугацаанаасаа өмнө тасарсан тохиолдолд 
# ирээдүйн боломжит шагналыг Critic-ээр таамаглаж Bootstrap хийх
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

debug_render        = True
debug               = True
play_frequency      = 30        # Хэдэн episode тутамд явцыг нэг удаа харуулах
num_episodes        = 10000

learning_rate       = 0.001
gamma               = 0.99
env_name            = "CartPole-v1"

group_size          = 6         # GRPO бүлгийн хэмжээ буюу нэг удаад зэрэг турших Agent тоо
max_steps           = 500       # Нэг тоглолтын дээд хугацаа

clip_epsilon        = 0.2       # Policy өөрчлөлтийг хязгаарлах PPO утга
entropy_coefficient = 0.01      # Exploration буюу шинэ зүйл туршихыг дэмжих
kl_beta             = 0.02      # Reference модельтой ижил байхыг шаардах жин

epochs_per_update   = 4         # Цуглуулсан багц өгөгдөл дээр давтан суралцах тоо
mini_batch_size     = 256       # Санах ойд нэг удаа ачааллах өгөгдлийн хэмжээ

max_grad_norm       = 0.5       # Градиент хэт өсөхөөс сэргийлэх хязгаар


# MODEL DEFINITION (Сүлжээний бүтэц)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Action сонгох магадлалын тархалт гаргах
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Төлөвийн утга буюу Value function таамаглах
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        return nn.Dense(features=1)(x)


# INITIALIZATION (Эхлүүлэлт)

env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
env         = gym.make(env_name, render_mode='human')

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

rng         = jax.random.PRNGKey(42)

# Моделийн параметрүүдийг үүсгэх
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


# JAX HELPER FUNCTIONS (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x).squeeze(-1)

@jax.jit
def logprob_from_probs(action_probas, actions):
    # Сонгосон action-ийн log probability утгыг салгаж авах
    probs = jnp.take_along_axis(action_probas, actions[:, None], axis=1).squeeze(1)
    probs = jnp.clip(probs, 1e-8, 1.0)
    return jnp.log(probs)

@jax.jit
def entropy_from_probs(action_probas):
    # Тархалтын тодорхой бус байдлыг хэмжих
    return -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1).mean()

@jax.jit
def kl_from_probs(p_new, p_ref):
    # Хоёр тархалтын хоорондох KL Divergence зөрүүг хэмжих
    return jnp.sum(p_new * (jnp.log(p_new + 1e-8) - jnp.log(p_ref + 1e-8)), axis=1).mean()


# BACKPROPAGATION LOGIC (Сургалтын алхмууд)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # Critic сүлжээг MSE Loss ашиглан шинэчлэх
    def loss_fn(params):
        values = critic_module.apply({'params': params}, props[0]).squeeze(-1)
        return jnp.mean(jnp.square(props[1] - values))

    loss, grads = jax.value_and_grad(loss_fn)(critic_model_params)
    updates, new_opt_state = critic_optimizer_def.update(grads, optimizer_state, critic_model_params)
    new_params = optax.apply_updates(critic_model_params, updates)
    return new_opt_state, new_params, loss

@jax.jit
def backpropagate_actor(optimizer_state, actor_model_params, actor_ref_params, critic_model_params, props):
    # props -> states, actions, returns, old_logp, group_mean_return
    def loss_fn(params):
        action_probas_new = actor_module.apply({'params': params}, props[0])
        logp_new          = logprob_from_probs(action_probas_new, props[1])

        # PPO Clipping logic буюу магадлалын харьцааг хязгаарлах
        ratio             = jnp.exp(logp_new - props[3])
        ratio_clipped     = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

        # Critic Baseline-ийг ашиглан Advantage тооцох
        values            = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]).squeeze(-1))
        
        # GRPO-style Advantage буюу бүлгийн дунджаар төвлөрүүлж Critic Baseline-аар засах
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

    (loss, (pg, kl, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss, pg, kl, ent


# ROLLOUT & RETURN LOGIC (Өгөгдөл цуглуулах)

def compute_returns(rewards, done_terms, bootstrap):
    # Алхам бүрийн Discounted Return тооцох
    T, G = len(rewards), float(bootstrap)
    returns = np.zeros(T, dtype=np.float32)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns

def rollout_trajectory(group_member_id, actor_old_params, critic_model_params, seed):
    # Нэг Agent-ийн туршилтыг гүйцэтгэх
    env_item       = env_array[group_member_id]
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards, done_terms, old_logps = [], [], [], [], []
    step           = 0
    last_state, last_truncated, last_terminated = state, False, False

    for _ in range(max_steps):
        # Action sampling болон Log Probability-ийг хадгалах
        probs_out = actor_inference(actor_old_params, jnp.asarray([state], dtype=jnp.float32))
        probs_np  = np.array(probs_out[0], dtype=np.float32)
        
        action    = np.random.choice(n_actions, p=probs_np)
        old_logp  = np.log(np.clip(probs_np[int(action)], 1e-8, 1.0))

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

    # Bootstrap буюу тасарсан тоглоомын ирээдүйн утгыг Critic-ээр таамаглах
    bootstrap = 0.0
    if last_truncated and (not last_terminated):
        bootstrap = float(critic_inference(critic_model_params, jnp.asarray([last_state], dtype=jnp.float32))[0])

    returns_seq = compute_returns(rewards, done_terms, bootstrap)
    # Группын үнэлгээнд ашиглах эхний алхмын утга
    group_reward = float(returns_seq[0]) if step > 0 else 0.0

    return group_member_id, group_reward, step, np.array(states), np.array(actions), returns_seq, np.array(old_logps)


def rollout_group(actor_old_params, critic_model_params, seed):
    # Бүлэг Agent-уудыг зэрэг тоглуулах
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_returns, group_old_logps = [], [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r, lp = rollout_trajectory(member_id, actor_old_params, critic_model_params, seed)
        group_rewards  .append(g_reward)
        group_lengths  .append(length  )
        group_states   .append(s       )
        group_actions  .append(a       )
        group_returns  .append(r       )
        group_old_logps.append(lp      )

    group_mean_reward = float(np.mean(group_rewards))
    return group_mean_reward, group_lengths, group_states, group_actions, group_returns, group_old_logps


# MAIN TRAINING LOOP (Үндсэн цикл)

try:
    global_step = 0
    print("\n=== STARTING PPO-STYLE GRPO WITH CRITIC BASELINE ===\n")

    for episode in range(num_episodes):
        # Одоогийн Policy-г сургалтын багц өгөгдөл цуглуулах behavior policy болгож хөлдөөх
        actor_old_params = actor_model_params

        # Өгөгдөл цуглуулах
        g_mean, g_lens, g_states, g_actions, g_returns, g_old_logps = rollout_group(actor_old_params, critic_model_params, episode)

        # Batch болгож нэгтгэх
        flat_states  = np.concatenate(g_states   ) if g_states    else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions  ) if g_actions   else np.zeros((0,))
        flat_returns = np.concatenate(g_returns  ) if g_returns   else np.zeros((0,))
        flat_old_lps = np.concatenate(g_old_logps) if g_old_logps else np.zeros((0,))

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

                    actor_optimizer_state, actor_model_params, actor_loss, pg_loss, kl_loss, entropy = backpropagate_actor(
                        actor_optimizer_state, actor_model_params, actor_ref_params, critic_model_params,
                        (states_j[idx], actions_j[idx], returns_j[idx], old_logp_j[idx], mean_r_j)
                    )

        # Явцыг лог болгон хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:5d} | Steps {global_step:8d} | GroupMeanR {g_mean:8.2f} | ActLoss {float(actor_loss):8.4f} | KL {float(kl_loss):8.4f} | CritLoss {float(critic_loss):8.4f}")

        # Visual Validation буюу явцыг нүдээр харах
        if episode % play_frequency == 0 and debug_render:
            state, info = env.reset(seed=int(episode))
            total_r = 0
            while True:
                probs = actor_inference(actor_model_params, jnp.asarray([state], dtype=jnp.float32))
                action = np.random.choice(n_actions, p=np.array(probs[0]))
                state, reward, terminated, truncated, _ = env.step(int(action))
                total_r += reward
                env.render()
                if terminated or truncated:
                    print(f"   >> Visual Test | Episode {episode} | Reward: {total_r}")
                    break

finally:
    env.close()
    for e in env_array: e.close()