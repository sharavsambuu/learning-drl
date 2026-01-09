#
# INTERNAL RL - ROBOTIC CWRRT, PPO(GRPO) TUNING (FETCHPUSH TOKEN ENV)
#
# RoboticCWRRT, Robotic Cross-Window Residual Recurrent Transformer, Sharavsambuu.G (2026/01/09)
#
# ЛАВЛАГАА:
#   - Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning
#     https://arxiv.org/abs/2512.20605
#
#
# ЗОРИЛГО:
#   FetchPushv4 орчинг Autoregressive token generation хэлбэрт оруулж,
#   Temporal Abstraction / Internal RL санааг практик байдлаар шалгах туршилт.
#   Роботын нарийн үйлдлүүдийг (continuous control) дискрет токенууд руу хөрвүүлж,
#   LLM шиг дараалсан байдлаар сургах замаар урт хугацааны төлөвлөлт (planning) хийлгэх.
#
#   1) Action      : Continuous (physics) -> Discrete tokens (0..255) per action dim
#   2) Observation : Dict -> flat vector (28,)
#   3) Task        : K дараалсан sub-goal (pinpad style) ашиглаж HRL-д илүү тохиромжтой болгох
#
#
# АРХИТЕКТУР (RoboticCWRRT):
#   Энэ модель нь Recurrent Transformer бүтэцтэй бөгөөд хоёр төрлийн санах ойтой:
#     - ssum (Intent ): Урт хугацааны зорилго буюу бодол. THINK үед шинэчлэгдэж, ACT үед froze хийгдэнэ.
#     - mem  (Context): Богино хугацааны контекст буюу сүүлийн хэдэн алхмын мэдээлэл.
#                       Attention механизм нь зөвхөн одоогийн оролтыг биш энэ санах ойг (Key/Value)
#                       ашиглан тооцоолол хийнэ.
#
#   Thinking vs Acting:
#     - THINK үе: freeze_thought=False. Модель ssum вектор буюу зорилгоо шинэчилнэ.
#     - ACT   үе: freeze_thought=True.  Модель ssum-г өөрчлөхгүйгээр, түүнд захирагдан үйлдлүүдээ (token-ууд) үүсгэнэ.
#
#
# СУРГАЛТЫН ҮЕ ШАТУУД:
#   PHASE 1 - SFT (Warm Start):
#     - Heuristic policy ашиглан роботын үндсэн хөдөлгөөнүүдийг цуглуулж,
#       Supervised Learning (Cross-Entropy Loss)-ээр моделийг эхлүүлж сургана.
#
#   PHASE 2 - PPO + GRPO:
#     - Group Rollout (GROUP_SIZE) цуглуулна.
#     - GRPO Advantage: A_i = score_i - mean(score_group). 
#       Энэ нь baseline network ашиглахгүйгээр variance багасгах арга юм.
#     - PPO Clipped Objective + KL Penalty (Reference Policy) + Entropy Bonus.
#
#


import os
import math
import time
import random
import numpy              as np
import gymnasium          as gym
import gymnasium_robotics 
import jax
import optax
import jax.numpy          as jnp
import flax.linen         as nn
from   flax.training      import train_state


# GLOBAL CONFIGURATION

SEED                 = 42

# Environment Settings
ENV_ID               = "FetchPush-v4"
MAX_EPISODE_STEPS    = 200

# Action Tokenization (Continuous -> Discrete)
ACTION_DIM           = 4
ACTION_BINS          = 256

# Task Difficulty (Sequential Sub-goals)
K_SEQUENTIAL_GOALS   = 3
GOAL_DIST_THRESHOLD  = 0.05
SPARSE_FINAL_REWARD  = True

# Internal RL / Temporal Abstraction
MACRO_STEP           = 10     # Нэг intent (ssum) барих хугацаа

# PPO + GRPO Hyperparameters
UPDATES              = 200
GROUP_SIZE           = 16     # Нэг update-д цуглуулах episode-ийн тоо
PPO_EPOCHS           = 3
MINI_BATCH_SIZE      = 16
CLIP_EPS             = 0.2
ENTROPY_COEFF        = 0.01

# KL Regularization (Dynamic)
KL_BETA              = 0.04
TARGET_KL            = 0.05
KL_ALPHA             = 1.2

# Optimizer & Model Settings
LR                   = 3e-4
GRPO_LR              = 1e-5
MAX_GRAD_NORM        = 1.0

EMBED_DIM            = 256
NUM_HEADS            = 4
MEMORY_LEN           = 16     # Attention Key/Value buffer size

# Warm Start Settings
SFT_ENABLE           = True
SFT_FLAG             = "sft_done_fetch_internal_rl.flag"
SFT_EPISODES         = 200
SFT_EPOCHS           = 3
SFT_BATCH_SIZE       = 256


# JAX Memory Allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# ENV WRAPPERS

class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Observation Dict-ийг хавтгай вектор (flat vector) болгон хувиргана.
      - "observation"   : (25,) биеийн байрлал + объект
      - "desired_goal"  : (3,)  хүрэх ёстой цэг
      - "achieved_goal" : (3,)  одоо байгаа цэг (хасагдсан, учир нь obs дотор бий)
    Гаралт: concat(observation, desired_goal) -> shape (28,)
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        # Tokenizer-ийг гаднаас хандах боломжтой болгох
        self.tokenizer = getattr(env, "tokenizer", None)

    def observation(self, obs_dict):
        return np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"]],
            dtype=np.float32,
        )


class ActionTokenizer:
    """
    Continuous action <-> Discrete token хөрвүүлэгч.
    Continuous domain : [-1, 1]
    Token domain      : [0, ACTION_BINS-1]
    """
    def __init__(self, bins=256):
        self.bins = int(bins)

    def encode(self, continuous_action):
        clipped = np.clip(continuous_action, -1.0, 1.0)
        norm    = (clipped + 1.0) / 2.0
        return (norm * (self.bins - 1)).astype(np.int32)

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float32)
        norm   = tokens / (self.bins - 1)
        return (norm * 2.0) - 1.0


class TokenActionWrapper(gym.ActionWrapper):
    """
    Орчинг MultiDiscrete токен хүлээж авдаг болгох wrapper.
    Үндсэн орчин: Box(-1, 1, (4,), float32)
    Энэ wrapper : MultiDiscrete([bins] * 4)
    """
    def __init__(self, env, bins=256):
        super().__init__(env)
        self.tokenizer    = ActionTokenizer(bins=bins)
        act_dim           = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete([bins] * act_dim)

    def action(self, token_action):
        return self.tokenizer.decode(np.array(token_action)).astype(np.float32)


class SequentialGoalsWrapper(gym.Wrapper):
    """
    FetchPush даалгаврыг Pinpad буюу дараалсан K зорилготой болгох.
    - Reset хийхэд: K ширхэг зорилго үүснэ (goal0 нь үндсэн, goal1..K нь offset).
    - Episode дундуур: Агент эхний зорилгод хүрвэл дараагийнх нь идэвхжинэ.
    - Шагнал (Reward): 
        SPARSE_FINAL_REWARD=True үед зөвхөн хамгийн сүүлийн зорилгод хүрэхэд +1 авна.
        Бусад үед 0 байна (GRPO сурахад хэцүү ч, pinpad бүтэц нь тусална).
    """
    def __init__(self, env, k=3, dist_threshold=0.05, sparse_final_reward=True):
        super().__init__(env)
        self.k                   = int(k)
        self.dist_threshold      = float(dist_threshold)
        self.sparse_final_reward = bool(sparse_final_reward)
        self._goals              = []
        self._goal_idx           = 0

    def _update_obs(self, obs):
        obs                 = dict(obs)
        obs["desired_goal"] = self._goals[self._goal_idx].copy()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        goals     = [base_goal.copy()]
        
        # Санамсаргүй offset нэмж дараагийн зорилгуудыг үүсгэх
        for _ in range(self.k - 1):
            offset = np.random.uniform(-0.10, 0.10, size=3).astype(np.float32)
            g      = base_goal + offset
            g[2]   = max(0.42, float(g[2])) # Ширээнээс доош орохгүй байх
            goals.append(g.astype(np.float32))

        self._goals    = goals
        self._goal_idx = 0

        info = dict(info)
        info["seq_goal_index"    ] = self._goal_idx
        info["seq_goals"         ] = np.stack(self._goals, axis=0)
        info["distance_threshold"] = self.dist_threshold

        return self._update_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired  = self._goals[self._goal_idx]
        dist     = float(np.linalg.norm(achieved - desired))

        reached  = dist < self.dist_threshold

        info = dict(info)
        info["seq_goal_index"    ] = self._goal_idx
        info["seq_reached"       ] = reached
        info["seq_dist"          ] = dist
        info["distance_threshold"] = self.dist_threshold

        if reached:
            # Дараагийн зорилго руу шилжих
            if self._goal_idx < self.k - 1:
                self._goal_idx        += 1
                info["seq_goal_index"] = self._goal_idx
                info["is_success"    ] = 0.0

                terminated = False
                reward = 0.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info
            else:
                # Бүх зорилго биелсэн
                info["is_success"] = 1.0
                terminated         = True
                reward             = 1.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info

        info["is_success"] = 0.0
        reward             = 0.0 if self.sparse_final_reward else float(base_reward)
        return self._update_obs(obs), reward, terminated, truncated, info


def create_env(render_mode=None):
    """
    Env Stack үүсгэх функц:
      FetchPush-v4
        -> SequentialGoalsWrapper        (Pinpad logic    )
        -> TokenActionWrapper            (Discrete Actions)
        -> TransformerObservationWrapper (Flat Observation)
    """
    env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=MAX_EPISODE_STEPS)
    env = SequentialGoalsWrapper(
        env,
        k                   = K_SEQUENTIAL_GOALS ,
        dist_threshold      = GOAL_DIST_THRESHOLD,
        sparse_final_reward = SPARSE_FINAL_REWARD
    )
    env = TokenActionWrapper(env, bins=ACTION_BINS)
    env = TransformerObservationWrapper(env)
    return env


# MODEL ARCHITECTURE (Recurrent Transformer)

class CWRRTCell(nn.Module):
    """
    Recurrent Transformer Cell.
    Үүрэг:
      1. ssum (Intent)-ийг оролт дээр Alpha gate-ээр нэмэх (Residual injection).
      2. Sliding Window Memory (mem) ашиглан Attention хийх.
      3. ssum-ийг Lambda gate-ээр шинэчлэх (EMA style).
    """
    embed_dim: int
    num_heads: int
    mem_len  : int = MEMORY_LEN

    @nn.compact
    def __call__(self, carry, x):
        mem, ssum = carry  
        # mem shape : (Batch, MEM_LEN, Embed)
        # ssum shape: (Batch, Embed)

        # Intent Injection (Top-down control)
        # alpha нь сурч болох параметр, intent хэр их нөлөөлөхийг шийднэ.
        alpha = nn.sigmoid(self.param("alpha", nn.initializers.zeros, (self.embed_dim,)))
        x_in  = x + (ssum * alpha)

        # Recurrent Attention Mechanism
        # Query                 : Current input (x_in)
        # Key/Value             : Memory history (mem)
        # Sliding Window Update : mem-ийг зүүн тийш шилжүүлж, шинэ x_in-ийг төгсгөлд нь нэмнэ.
        
        # Шинэ memory үүсгэх
        new_mem_entry = x_in[:, None, :] # (B, 1, E)
        updated_mem   = jnp.concatenate([mem[:, 1:, :], new_mem_entry], axis=1)
        
        # Attention Calculation
        # Q = x_in (B, 1, E), KV = updated_mem (B, MEM_LEN, E)
        y = nn.LayerNorm()(x_in)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=y[:, None, :], 
            inputs_kv=updated_mem
        )
        y = y.squeeze(1) # (B, E)
        
        x_mid = x_in + y # Residual

        # MLP Block
        y     = nn.LayerNorm()(x_mid)
        y     = nn.Dense(self.embed_dim * 4)(y)
        y     = nn.gelu(y)
        y     = nn.Dense(self.embed_dim)(y)
        x_out = x_mid + y # Residual

        # Intent Update (State Tracking)
        # lambda нь хуучин intent болон шинэ мэдээллийг холих харьцаа.
        lam = nn.sigmoid(self.param("lambda", nn.initializers.zeros, (self.embed_dim,)))
        new_ssum = (ssum * lam) + (x_out * (1.0 - lam))

        return (updated_mem, new_ssum), x_out


class RoboticCWRRT(nn.Module):
    """
    Single-step Policy Network.
    Input:
      obs      : (B, 28)
      carry    : (mem, ssum)
      time_idx : int (Macro step доторх цаг хугацааны байрлал)
    Output:
      logits   : (B, ActionDim, ActionBins)
      next_carry
    """
    action_dim : int = ACTION_DIM
    action_bins: int = ACTION_BINS
    embed_dim  : int = EMBED_DIM
    mem_len    : int = MEMORY_LEN

    def setup(self):
        self.input_proj  = nn.Dense(self.embed_dim)
        self.time_embed  = nn.Embed(MACRO_STEP + 1, self.embed_dim) # Temporal embedding
        self.cell        = CWRRTCell(embed_dim=self.embed_dim, num_heads=NUM_HEADS, mem_len=self.mem_len)
        self.action_head = nn.Dense(self.action_dim * self.action_bins)

    def init_carry(self, batch_size):
        # Санах ойг тэгээр дүүргэж эхлүүлэх
        return (
            jnp.zeros((batch_size, self.mem_len, self.embed_dim)), # mem history
            jnp.zeros((batch_size, self.embed_dim)),               # ssum intent
        )

    def __call__(self, obs, carry, time_idx, freeze_thought=False):
        mem, ssum = carry
        
        # Observation Embedding + Time Embedding
        x     = self.input_proj(obs)
        t_emb = self.time_embed(jnp.clip(time_idx, 0, MACRO_STEP))
        x     = x + t_emb

        (new_mem, new_ssum), x_out = self.cell((mem, ssum), x)

        # Freeze Logic:
        # freeze_thought=True үед new_ssum-ийг хаяж, хуучин ssum-ийг хадгална.
        # Гэхдээ new_mem (context) үргэлж шинэчлэгдэнэ (sliding window).
        is_frozen = jnp.asarray(freeze_thought)
        if is_frozen.ndim > 0:
            is_frozen = is_frozen[..., None]  # Broadcast handle

        next_ssum  = jnp.where(is_frozen, ssum, new_ssum)
        next_carry = (new_mem, next_ssum)

        logits     = self.action_head(x_out)
        logits     = logits.reshape(obs.shape[0], self.action_dim, self.action_bins)

        return logits, next_carry

    def forward_sequence(self, obs_btf, freeze_btf):
        """
        Scan ашиглан хугацааны дарааллаар тооцоолол хийх (unroll).
        Training үед ашиглагдана.
        """
        B, T, _ = obs_btf.shape
        carry0  = self.init_carry(B)

        obs_T = jnp.swapaxes(obs_btf, 0, 1)      # (T, B, F)
        frz_T = jnp.swapaxes(freeze_btf, 0, 1)   # (T, B)
        
        # Time index үүсгэх (Macro step доторх цикл)
        # Энд энгийнээр 0..9, 0..9 гэж давтагдана гэж үзье эсвэл зүгээр л T mod MACRO
        time_T = jnp.arange(T) % MACRO_STEP
        time_T = jnp.tile(time_T[:, None], (1, B)) # (T, B)

        def scan_fn(carry, inp):
            obs_t, frz_t, t_idx = inp
            logits_t, carry     = self.__call__(obs_t, carry, t_idx, freeze_thought=frz_t)
            return carry, logits_t

        _, logits_T = jax.lax.scan(scan_fn, carry0, (obs_T, frz_T, time_T))
        logits_btav = jnp.swapaxes(logits_T, 0, 1)
        return logits_btav


# METRICS & UTILS

@jax.jit
def logprob_bt_from_logits(logits_btav, actions_bta):
    """
    Log Probability тооцоолох.
    Action dimension бүрийн logprob-ийг нэмнэ (independence assumption).
    """
    logp  = jax.nn.log_softmax(logits_btav, axis=-1)
    taken = jnp.take_along_axis(logp, actions_bta[..., None], axis=-1).squeeze(-1)
    return jnp.sum(taken, axis=-1)

@jax.jit
def entropy_bt_from_logits(logits_btav):
    """
    Entropy тооцоолох (Exploration хэмжүүр).
    """
    p   = jax.nn.softmax(logits_btav, axis=-1)
    lp  = jax.nn.log_softmax(logits_btav, axis=-1)
    ent = -jnp.sum(p * lp, axis=-1)   # (B,T,A)
    return jnp.mean(ent, axis=-1)     # (B,T)

@jax.jit
def kl_bt_from_logits(logits_new_btav, logits_ref_btav):
    """
    KL Divergence тооцоолох (Policy өөрчлөлтийг хянах).
    """
    p_new  = jax.nn.softmax(logits_new_btav, axis=-1)
    lp_new = jax.nn.log_softmax(logits_new_btav, axis=-1)
    lp_ref = jax.nn.log_softmax(logits_ref_btav, axis=-1)
    kl     = jnp.sum(p_new * (lp_new - lp_ref), axis=-1)  # (B,T,A)
    return jnp.mean(kl, axis=-1)                          # (B,T)


# TRAINING STATE

class Trainer:
    """
    Model Parameters, Optimizer State болон Random Key хадгалах класс.
    """
    def __init__(self, seed=SEED):
        self.model = RoboticCWRRT()
        self.rng   = jax.random.PRNGKey(seed)

        # Dummy input-ээр моделийг эхлүүлж параметрүүдийг үүсгэх
        dummy_obs   = jnp.zeros((1, 28), dtype=jnp.float32)
        dummy_carry = self.model.init_carry(1)
        dummy_t     = jnp.array([0], dtype=jnp.int32)
        
        params = self.model.init(self.rng, dummy_obs, dummy_carry, dummy_t, freeze_thought=False)["params"]

        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR),
            ),
        )

    def set_lr(self, lr):
        # Learning Rate өөрчлөх (SFT -> RL шилжихэд хэрэгтэй)
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = self.state.params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(lr),
            ),
        )


# PHASE 1, SFT (Warm Start)

def generate_sft_dataset(env, episodes=SFT_EPISODES):
    """
    Heuristic (дүрэмд суурилсан) аргаар сургалтын өгөгдөл цуглуулах.
    FetchPush дээр:
      1. Gripper блокноос хол бол -> Блок руу дөхнө.
      2. Блок дээр очсон бол -> Блокийг зорилго руу түлхэнэ.
    """
    obs_list = []
    act_list = []

    for _ in range(episodes):
        obs, info = env.reset()
        for _t in range(MAX_EPISODE_STEPS):
            grip_pos  = obs[:3]
            block_pos = obs[3:6]
            goal_pos  = obs[-3:]

            dist_block = np.linalg.norm(grip_pos - block_pos)

            if dist_block > 0.06:
                target = block_pos.copy()
                target[2] += 0.02 # Блокны дээгүүр очих
            else:
                target = goal_pos.copy()

            # Cartesian хөдөлгөөнийг тооцоолох (P-controller)
            delta = (target - grip_pos) * 5.0
            delta = np.clip(delta, -1.0, 1.0)

            # Gripper төлөв (-1.0 = хаалттай)
            cont_action = np.append(delta, -1.0).astype(np.float32)

            # Continuous action -> Discrete tokens
            tokens = env.tokenizer.encode(cont_action).astype(np.int32)

            obs_list.append(obs.astype(np.float32))
            act_list.append(tokens.astype(np.int32))

            obs, r, term, trunc, info = env.step(tokens)
            if term or trunc:
                break

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int32)


def sft_train(trainer, obs_all, act_all):
    """
    Supervised Fine-Tuning (Cross Entropy Loss).
    RL эхлэхээс өмнө моделийг суурь мэдлэгтэй болгоно.
    """
    N   = len(obs_all)
    idx = np.arange(N)

    @jax.jit
    def sft_step(state, obs_b, act_b, rng):
        def loss_fn(p):
            carry0    = trainer.model.init_carry(obs_b.shape[0])
            # SFT дээр temporal structure чухал биш тул t_idx=0, freeze=False өгнө
            # Эсвэл scan ашиглаж болно, энд хялбарчлав.
            t_idx     = jnp.zeros((obs_b.shape[0],), dtype=jnp.int32)
            logits, _ = state.apply_fn({"params": p}, obs_b, carry0, t_idx, freeze_thought=False)

            logp      = jax.nn.log_softmax(logits, axis=-1)
            one_hot   = jax.nn.one_hot(act_b, ACTION_BINS)
            loss      = -jnp.sum(one_hot * logp, axis=-1).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state       = state.apply_gradients(grads=grads)
        return state, loss

    for ep in range(SFT_EPOCHS):
        np.random.shuffle(idx)
        losses = []

        for i in range(0, N - SFT_BATCH_SIZE + 1, SFT_BATCH_SIZE):
            b = idx[i:i + SFT_BATCH_SIZE]

            obs_b = jnp.asarray(obs_all[b], dtype=jnp.float32)
            act_b = jnp.asarray(act_all[b], dtype=jnp.int32)

            trainer.rng, step_rng = jax.random.split(trainer.rng)
            trainer.state, loss   = sft_step(trainer.state, obs_b, act_b, step_rng)
            losses.append(loss)

        print(f"[SFT] Epoch {ep} | Loss: {float(jnp.mean(jnp.array(losses))):.6f}")


# PHASE 2, INTERNAL RL ROLLOUTS

def score_episode(info):
    """
    Episode-ийн үнэлгээ (Reward Function).
    Дараалсан зорилгуудыг (Pinpad) биелүүлсэн байдлаар оноо өгнө.
    """
    s = float(info.get("seq_goal_index", 0)) * 10.0
    if float(info.get("is_success", 0.0)) > 0.5:
        s += 50.0 # Эцсийн зорилгод хүрсэн бол том шагнал
    return s


def collect_group_rollouts(env, trainer, group_size=GROUP_SIZE):
    """
    RL сургалтанд зориулж өгөгдөл цуглуулах (Rollout).
    Temporal Abstraction логик:
      1. THINK алхам: 
         - freeze=False.
         - Модель ssum (Intent)-ээ шинэчилнэ.
      2. ACT алхмууд (MACRO_STEP хугацаанд):
         - freeze=True.
         - Модель ssum-ээ өөрчлөхгүй барьж байгаад үйлдлээ хийнэ.
         - Энэ нь моделийг нэг БОДОЛ-оор олон үйлдэл хийхэд сургана.
    """
    trajs  = []
    scores = []

    for _ in range(group_size):
        obs, info = env.reset()
        carry = trainer.model.init_carry(1)

        obs_seq   = []
        act_seq   = []
        oldlp_seq = []
        frz_seq   = []

        done = False
        t = 0

        while (not done) and (t < MAX_EPISODE_STEPS):
            obs_j = jnp.asarray(obs[None, :], dtype=jnp.float32)
            
            # THINK PHASE
            # Intent (ssum) шинэчлэх, үйлдэл хийхгүй (эсвэл эхний үйлдлийг авч болно)
            # Энд шууд ACT руу шилжиж байгаа ч, эхний алхам дээр ssum шинэчлэгдэнэ.
            
            # MACRO BLOCK LOOP
            # Нэг intent дор хэд хэдэн алхам хийх
            
            # Эхний алхам дээр бодно (Unfreeze)
            should_freeze = False
            
            for m_step in range(MACRO_STEP):
                if t >= MAX_EPISODE_STEPS: break
                
                trainer.rng, k1 = jax.random.split(trainer.rng)
                t_idx_j         = jnp.array([m_step], dtype=jnp.int32)
                
                # Inference хийх
                logits, carry = trainer.state.apply_fn(
                    {"params": trainer.state.params}, 
                    obs_j, carry, t_idx_j, 
                    freeze_thought=should_freeze
                )
                
                # Дараагийн алхмуудад бодлоо царцаана
                should_freeze = True

                # Үйлдэл сонгох (Sample from logits)
                logp = jax.nn.log_softmax(logits, axis=-1)
                keys = jax.random.split(k1, ACTION_DIM)

                actions = []
                for a in range(ACTION_DIM):
                    tok = jax.random.categorical(keys[a], logp[0, a], axis=-1)
                    actions.append(tok)
                act = jnp.stack(actions, axis=0).astype(jnp.int32)

                # Reference Logprob хадгалах (Ratio тооцоход хэрэгтэй)
                taken_lp = jnp.take_along_axis(logp[0], act[:, None], axis=-1).squeeze(-1)
                old_lp   = jnp.sum(taken_lp)

                obs_seq  .append(obs.astype(np.float32))
                act_seq  .append(np.array(act, dtype=np.int32))
                oldlp_seq.append(float(old_lp))
                frz_seq  .append(should_freeze) # True except for first step of macro

                obs, r, term, trunc, info = env.step(np.array(act, dtype=np.int32))
                t += 1

                if term or trunc:
                    done = True
                    break
                
                obs_j = jnp.asarray(obs[None, :], dtype=jnp.float32)

        trajs.append({
            "obs"     : np.asarray(obs_seq  , dtype=np.float32),
            "actions" : np.asarray(act_seq  , dtype=np.int32  ),
            "old_logp": np.asarray(oldlp_seq, dtype=np.float32),
            "freeze"  : np.asarray(frz_seq  , dtype=np.bool_  ),
        })

        scores.append(score_episode(info))

    return trajs, np.asarray(scores, dtype=np.float32)


def compute_grpo_advantages(scores, group_size=GROUP_SIZE):
    """
    GRPO (Group Relative Policy Optimization) Advantage.
    Baseline NN (Value Function) сургах шаардлагагүйгээр
    тухайн бүлэг доторх дундажаас хэр сайн байснаар нь үнэлнэ.
    """
    mean = float(np.mean(scores))
    adv  = (scores - mean).astype(np.float32)
    adv  = np.clip(adv, -5.0, 5.0) # Хэт өндөр утгаас сэргийлэх
    return adv, mean


# GRPO / PPO UPDATE

def pad_batch(trajs):
    """
    Ялгаатай урттай episode-уудыг нэг Batch болгож Padding хийх.
    Mask tensor үүсгэж, loss тооцохдоо padding хэсгийг хасна.
    """
    B       = len(trajs)
    max_len = max(int(t["obs"].shape[0]) for t in trajs)

    obs   = np.zeros((B, max_len, 28), dtype=np.float32)
    acts  = np.zeros((B, max_len, 4 ), dtype=np.int32  )
    oldlp = np.zeros((B, max_len    ), dtype=np.float32)
    frz   = np.zeros((B, max_len    ), dtype=np.bool_  )
    mask  = np.zeros((B, max_len    ), dtype=np.float32)

    for i, t in enumerate(trajs):
        T = int(t["obs"].shape[0])
        if T == 0: continue

        obs  [i, :T] = t["obs"     ]
        acts [i, :T] = t["actions" ]
        oldlp[i, :T] = t["old_logp"]
        frz  [i, :T] = t["freeze"  ]
        mask [i, :T] = 1.0

    return (
        jnp.asarray(obs  , dtype=jnp.float32),
        jnp.asarray(acts , dtype=jnp.int32  ),
        jnp.asarray(oldlp, dtype=jnp.float32),
        jnp.asarray(frz  , dtype=jnp.bool_  ),
        jnp.asarray(mask , dtype=jnp.float32),
    )


@jax.jit
def ppo_grpo_update_step(state, ref_params, obs, acts, oldlp, adv, frz, mask, kl_beta):
    """
    PPO Loss Function:
      1. Surrogate Loss: Policy хэр их өөрчлөгдсөнийг Advantage-аар үржүүлнэ.
      2. KL Penalty: Хуучин Policy-оос хэт холдохгүй байх.
      3. Entropy Bonus: Санамсаргүй байдлыг дэмжиж, гацахаас сэргийлэх.
    """
    def loss_fn(p):
        # Одоогийн policy-ийн logits
        logits_new = state.apply_fn({"params": p}, obs, frz, method=RoboticCWRRT.forward_sequence)
        # Reference policy-ийн logits (KL-д зориулж)
        logits_ref = state.apply_fn({"params": ref_params}, obs, frz, method=RoboticCWRRT.forward_sequence)

        newlp_bt   = logprob_bt_from_logits(logits_new, acts)
        ratio      = jnp.exp(newlp_bt - oldlp)

        # PPO Clipping logic
        adv_bt     = adv[:, None]
        unclipped  = ratio * adv_bt
        clipped    = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_bt
        surr       = jnp.minimum(unclipped, clipped)

        # Aux losses
        kl_bt      = kl_bt_from_logits(logits_new, logits_ref)
        ent_bt     = entropy_bt_from_logits(logits_new)

        # Masking padding regions
        m          = mask
        surr_mean  = jnp.sum(surr   * m) / (jnp.sum(m) + 1e-8)
        kl_mean    = jnp.sum(kl_bt  * m) / (jnp.sum(m) + 1e-8)
        ent_mean   = jnp.sum(ent_bt * m) / (jnp.sum(m) + 1e-8)

        loss       = -surr_mean + (kl_beta * kl_mean) - (ENTROPY_COEFF * ent_mean)
        return loss, (kl_mean, ent_mean)

    (loss, (kl_mean, ent_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, kl_mean, ent_mean


def ppo_grpo_epoch(trainer, ref_params, trajs, advs, kl_beta):
    """
    Цуглуулсан өгөгдөл дээр нэг удаагийн сургалт явуулах (Epoch).
    Өгөгдлийг Mini-batch болгон хувааж, градиент шинэчлэлт хийнэ.
    """
    idx = np.arange(len(trajs))
    np.random.shuffle(idx)

    for i in range(0, len(trajs), MINI_BATCH_SIZE):
        mb_idx   = idx[i:i + MINI_BATCH_SIZE]
        mb_trajs = [trajs[j] for j in mb_idx]
        mb_adv   = jnp.asarray(advs[mb_idx], dtype=jnp.float32)

        obs, acts, oldlp, frz, mask = pad_batch(mb_trajs)

        trainer.state, loss, kl_m, ent_m = ppo_grpo_update_step(
            trainer.state, ref_params, obs, acts, oldlp, mb_adv, frz, mask, kl_beta
        )

    return float(loss), float(kl_m), float(ent_m)


# MAIN EXECUTION

def main():
    np.random.seed(SEED)
    random.seed(SEED)

    env     = create_env(render_mode=None)
    trainer = Trainer(seed=SEED)

    # RL сургалтын үед бага LR ашиглана
    trainer.set_lr(GRPO_LR)

    # Warm Start (SFT Phase)
    # Хэрэв flag байхгүй бол SFT сургалтыг эхлүүлнэ.
    if SFT_ENABLE and (not os.path.exists(SFT_FLAG)):
        print("[SFT] Building heuristic dataset...")

        sft_env          = create_env(render_mode=None)
        obs_all, act_all = generate_sft_dataset(sft_env, episodes=SFT_EPISODES)
        sft_env.close()

        print(f"[SFT] Steps: {len(obs_all)} | Batch: {SFT_BATCH_SIZE}")
        sft_train(trainer, obs_all, act_all)

        with open(SFT_FLAG, "w") as f:
            f.write("done")

    # Frozen Reference Policy (KL Penalty-д зориулж хадгалах)
    frozen_ref = trainer.state.params
    kl_beta    = float(KL_BETA)

    print("\n" + "=" * 72)
    print("  INTERNAL RL (FetchPush Token Env) - PPO+GRPO")
    print(f"  Updates: {UPDATES} | Group: {GROUP_SIZE} | Macro: {MACRO_STEP} | MaxT: {MAX_EPISODE_STEPS}")
    print("=" * 72 + "\n")

    # RL Loop (Iterative Improvement)
    for upd in range(1, UPDATES + 1):
        # Өгөгдөл цуглуулах
        trajs, scores    = collect_group_rollouts(env, trainer, group_size=GROUP_SIZE)
        advs, mean_score = compute_grpo_advantages(scores, group_size=GROUP_SIZE)

        kl_v  = 0.0
        ent_v = 0.0

        # Policy Update (PPO)
        for _ in range(PPO_EPOCHS):
            loss, kl_v, ent_v = ppo_grpo_epoch(trainer, frozen_ref, trajs, advs, kl_beta)

        # Dynamic KL Control (KL хэт ихэсвэл Beta-г нэмнэ, багасвал хасна)
        if kl_v > TARGET_KL * 1.5:
            kl_beta *= KL_ALPHA
        elif kl_v < TARGET_KL / 1.5:
            kl_beta /= KL_ALPHA

        # Log
        if upd % 5 == 0 or upd == 1:
            print(
                f"[UPD {upd:4d}] "
                f"MeanScore: {mean_score:7.2f} | Best: {float(np.max(scores)):7.2f} | "
                f"KL: {kl_v:.4f} | Ent: {ent_v:.4f} | Beta: {kl_beta:.4f}"
            )

        # Visualization
        if upd % 5 == 0:
            viz = create_env(render_mode="human")
            collect_group_rollouts(viz, trainer, group_size=1)
            viz.close()

    env.close()


if __name__ == "__main__":
    main()