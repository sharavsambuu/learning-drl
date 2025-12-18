#
# GRPO WITH A2C BASELINE (DEEPSEEK-V3.2 STYLE ADVANTAGE)
#
# Зорилго
# Энэхүү код нь GRPO алгоритмыг A2C буюу Advantage Actor-Critic арга барилтай 
# хослуулан CartPole тоглоомыг сургах MVP хувилбар юм
#
# Архитектур
# Actor Network нь MLP бүтэцтэй бөгөөд Action Probability гаргана
# Critic Network нь тухайн State-ийн утга буюу Value-г таамаглаж Baseline болно
#
# Аргачлал
# DeepSeek-V3.2 Advantage буюу бүлгийн үр дүнгээс дунджийг хасаж Advantage тооцох
# Гэхдээ стандарт хазайлтыг (std) ашиглахгүйгээр шууд утгаар нь тооцож
# дээр нь Critic Baseline-ийг ашиглан Variance-ийг бууруулах оролдлого юм
#
# Сургалтын онцлог
# Critic нь Returns-ийг таамаглаж сурах бөгөөд Actor нь түүнийг ашиглан
# өөрийн Advantage-ийг нарийвчлан тодорхойлно
# Төгсгөлд нь Entropy Bonus ашиглан Exploration-ийг дэмжсэн
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
play_frequency      = 30        # Хэдэн episode тутамд явцыг харуулах
num_episodes        = 10000
learning_rate       = 0.001
gamma               = 0.99      # Ирээдүйн шагналын хөнгөлөлт буюу Discount Factor
env_name            = "CartPole-v1"
group_size          = 6         # GRPO бүлгийн хэмжээ
max_steps           = 500       # Тоглоомын дээд хязгаар

entropy_coefficient = 0.01      # Моделийг хэт нэг хэвийн болохоос сэргийлэх


# MODEL ARCHITECTURE (Моделийн бүтэц)

class ActorNetwork(nn.Module):
    n_actions: int
    @nn.compact
    def __call__(self, x):
        # Action магадлал тооцох сүлжээ
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return nn.softmax(x)

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        # State-ийн утгыг таамаглах буюу Baseline сүлжээ
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        return nn.Dense(features=1)(x)


# INITIALIZATION (Орчин болон Моделийг эхлүүлэх)

env_array   = [gym.make(env_name, render_mode=None) for _ in range(group_size)]
env         = gym.make(env_name, render_mode='human')

state, info = env.reset()
state       = np.array(state, dtype=np.float32)
state_shape = state.shape
n_actions   = env.action_space.n

# Сүлжээнүүдийг цэнэглэх
actor_module           = ActorNetwork(n_actions=n_actions)
dummy_input            = jnp.zeros(state.shape)
actor_params           = actor_module.init(jax.random.PRNGKey(0), dummy_input)['params']
actor_model_params     = actor_params

critic_module          = CriticNetwork()
critic_params          = critic_module.init(jax.random.PRNGKey(1), dummy_input)['params']
critic_model_params    = critic_params

# Optimizer тохиргоо
actor_optimizer_def    = optax.adam(learning_rate)
critic_optimizer_def   = optax.adam(learning_rate)

actor_optimizer_state  = actor_optimizer_def .init(actor_model_params)
critic_optimizer_state = critic_optimizer_def.init(critic_model_params)


# JAX HELPER FUNCTIONS (Тооцооллын функцүүд)

@jax.jit
def actor_inference(params, x):
    return actor_module.apply({'params': params}, x)

@jax.jit
def critic_inference(params, x):
    return critic_module.apply({'params': params}, x).squeeze(-1)

@jax.jit
def backpropagate_critic(optimizer_state, critic_model_params, props):
    # props[0] - states, props[1] - returns
    def loss_fn(params):
        values = critic_module.apply({'params': params}, props[0]).squeeze(-1)
        # MSE Loss буюу таамагласан утга болон бодит үр дүнгийн зөрүү
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
        # Сонгосон action-ийн магадлалыг салгаж авах
        probs         = jnp.take_along_axis(action_probas, props[1][:, None], axis=1).squeeze(1)
        probs         = jnp.clip(probs, 1e-8, 1.0)
        logp          = jnp.log(probs)

        # Baseline утгыг Critic-ээс авах бөгөөд градиент урсгалыг таслах
        values        = jax.lax.stop_gradient(critic_module.apply({'params': critic_model_params}, props[0]).squeeze(-1))

        # GRPO-style Advantage (DeepSeek-V3.2) буюу бүлгийн дунджаар төвлөрүүлж Critic Baseline-аар засах
        advantages    = (props[2] - props[3]) - values

        # Exploration-ийг дэмжих Entropy тооцоолол
        entropies     = -jnp.sum(action_probas * jnp.log(action_probas + 1e-8), axis=1)

        # Actor Loss буюу магадлалыг Advantage-аар үржүүлж Entropy нэмэх
        loss          = -jnp.mean(logp * jax.lax.stop_gradient(advantages) + entropy_coefficient * entropies)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(actor_model_params)
    updates, new_opt_state = actor_optimizer_def.update(grads, optimizer_state, actor_model_params)
    new_params = optax.apply_updates(actor_model_params, updates)
    return new_opt_state, new_params, loss


# ROLLOUT LOGIC (Өгөгдөл цуглуулах хэсэг)

def compute_returns(rewards, done_terms, bootstrap):
    # Алхам бүрийн Discounted Return тооцоолох
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = float(bootstrap)
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G * (1.0 - float(done_terms[t]))
        returns[t] = G
    return returns


def rollout_trajectory(group_member_id, actor_model_params, critic_model_params, seed):
    # Нэг Agent-ийн бүтэн тоглолтыг гүйцэтгэх
    env_item       = env_array[group_member_id]
    state, info    = env_item.reset(seed=int(seed))
    state          = np.array(state, dtype=np.float32)

    states, actions, rewards, done_terms = [], [], [], []
    step           = 0
    last_state     = state
    last_truncated = False
    last_terminated= False

    for _ in range(max_steps):
        # Action sampling хийх
        action_probs = actor_inference(actor_model_params, jnp.asarray([state]))
        action       = np.random.choice(n_actions, p=np.array(action_probs[0]))

        next_state, reward, terminated, truncated, info = env_item.step(int(action))
        
        states.append(state)
        actions.append(int(action))
        rewards.append(float(reward))
        done_terms.append(float(terminated))

        step          += 1
        state          = np.array(next_state, dtype=np.float32)
        last_state     = state
        last_truncated = bool(truncated)
        last_terminated= bool(terminated)

        if terminated or truncated: break

    # Truncated болсон үед ирээдүйн утгыг Critic-ээр таамаглаж Bootstrap хийх
    bootstrap = 0.0
    if last_truncated and (not last_terminated):
        bootstrap = float(critic_inference(critic_model_params, jnp.asarray([last_state]))[0])

    returns_seq = compute_returns(rewards, done_terms, bootstrap)
    # Группын үнэлгээнд ашиглах эхний алхмын Return утга
    group_reward = float(returns_seq[0]) if step > 0 else 0.0

    return group_member_id, group_reward, step, np.array(states), np.array(actions), returns_seq


def rollout_group(actor_model_params, critic_model_params, seed):
    # Бүлгээр Rollout хийж GRPO-д зориулсан Baseline тооцох
    group_rewards, group_lengths = [], []
    group_states, group_actions, group_returns = [], [], []

    for member_id in range(group_size):
        _, g_reward, length, s, a, r = rollout_trajectory(member_id, actor_model_params, critic_model_params, seed)
        group_rewards.append(g_reward)
        group_lengths.append(length  )
        group_states .append(s       )
        group_actions.append(a       )
        group_returns.append(r       )

    # Бүлгийн дундаж үр дүн буюу GRPO-ийн гол Baseline
    group_mean_reward = float(np.mean(group_rewards))

    return group_mean_reward, group_lengths, group_states, group_actions, group_returns


# TRAINING LOOP (Үндсэн сургалт)

try:
    global_step = 0
    print("\n=== STARTING GRPO + A2C BASELINE TRAINING ===\n")

    for episode in range(num_episodes):
        # Өгөгдөл цуглуулах
        g_mean, g_lens, g_states, g_actions, g_returns = rollout_group(actor_model_params, critic_model_params, episode)

        # Batch болгож нэгтгэх
        flat_states  = np.concatenate(g_states , axis=0) if g_states  else np.zeros((0,) + state_shape)
        flat_actions = np.concatenate(g_actions, axis=0) if g_actions else np.zeros((0,))
        flat_returns = np.concatenate(g_returns, axis=0) if g_returns else np.zeros((0,))

        if flat_states.shape[0] > 0:
            global_step += int(flat_states.shape[0])

            # Critic шинэчлэх буюу Value function сургах
            critic_optimizer_state, critic_model_params, critic_loss = backpropagate_critic(
                critic_optimizer_state, critic_model_params,
                (jnp.asarray(flat_states), jnp.asarray(flat_returns))
            )

            # Actor шинэчлэх буюу Policy сургах
            actor_optimizer_state, actor_model_params, actor_loss = backpropagate_actor(
                actor_optimizer_state, actor_model_params, critic_model_params,
                (jnp.asarray(flat_states), jnp.asarray(flat_actions, dtype=jnp.int32), jnp.asarray(flat_returns), jnp.asarray(g_mean))
            )

        # Явцыг хэвлэх
        if debug and (episode % 10 == 0):
            print(f"Ep {episode:5d} | Steps {global_step:8d} | GroupMeanR {g_mean:8.2f} | ActorL {float(actor_loss):8.4f} | CriticL {float(critic_loss):8.4f}")

        # Visual Validation буюу тоглолтыг харах
        if episode % play_frequency == 0 and debug_render:
            state, info = env.reset(seed=int(episode))
            total_r = 0
            while True:
                probs = actor_inference(actor_model_params, jnp.asarray([state]))
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