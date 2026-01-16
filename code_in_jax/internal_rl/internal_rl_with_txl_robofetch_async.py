#
# HIERARCHICAL БҮТЭЦТЭЙ INTERNAL RL ТУРШИЛТ
#
# АРХИТЕКТУРЫН ТАЙЛБАР:
# Энэ код нь FetchPush-v4 робот орчинд өгөгдсөн даалгаврыг биелүүлэх зорилготой 2 түвшний агент юм.
# Дараах үндсэн хэсгүүдээс бүрдэнэ:
#
# 1. Meta-Controller (Дээд түвшний удирдлага):
#    - GRU болон Episodic Slot Memory (Урт хугацааны санах ой) ашиглан орчныг ажиглаж,
#      стратегийн зорилго буюу intention (u_prop)-ийг тодорхойлно.
#    - Мөн хэзээ шинэ зорилго өгөх вэ гэдгийг шийдэх (switch) болон өмнөх зорилгыг
#      үргэлжлүүлэх хувь хэмжээг (beta) тооцоолно.
#
# 2. Episodic Slot Memory (Санах ой):
#    - Мэдээллийг Key-Value хэлбэрээр хадгалах бөгөөд санах ойн үүрнүүд (slots) нь
#      age болон хүч strength гэсэн үзүүлэлтүүдтэй.
#    - Ашиглагдаагүй удсан эсвэл чухал биш мэдээллийг шинэ мэдээллээр дарж бичих (eviction)
#      механизмтай тул санах ой үр ашигтай ажиллана.
#
# 3. Residual Intervention Decoder (Хөрвүүлэгч):
#    - Meta-Controller-ийн гаргасан u_int зорилгыг Worker-ийн шууд ойлгох хэмжээст
#      вектор (delta) болгон хөрвүүлнэ. Hypernetwork-тай төстэй зарчмаар ажиллана.
#
# 4. TXL Worker (Доод түвшний гүйцэтгэгч):
#    - Transformer-XL (TXL) архитектуртай төстэй, өмнөх алхмуудын мэдээллээ (FIFO memory) хадгалдаг.
#    - Орчны төлөв болон Decoder-ээс ирсэн delta мэдээллийг нэгтгэн шууд роботын үе мөчний
#      хөдөлгөөнийг (action) гаргана.
#
# МАШИН СУРГАЛТЫН ПРОЦЕСС:
# - Phase 1 (SFT): 
#   Worker-ийг эхлээд энгийн алгоритмаар цуглуулсан өгөгдөл дээр
#   Supervised Fine-Tuning (SFT) хийж роботыг хөдөлгөх анхан шатны чадвартай болгоно.
# - Phase 2 (RL): 
#   Worker-ийн жинг царцааж (frozen), зөвхөн Meta-Controller болон Decoder-ийг
#   PPO (Proximal Policy Optimization) болон GRPO (Group Relative Policy Optimization) аргаар сургана.
#
#


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
import time
import random
import numpy as np

import gymnasium as gym
import gymnasium_robotics
from gymnasium.vector import AsyncVectorEnv

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# Hyperparams 
SEED                 = 42

ENV_ID               = "FetchPush-v4"
MAX_EPISODE_STEPS    = 200

ACTION_DIM           = 4
ACTION_BINS          = 256

# Олон зорилго дараалан биелүүлэх тохиргоо (Sequential goals)
K_SEQUENTIAL_GOALS   = 3
GOAL_DIST_THRESHOLD  = 0.05
SPARSE_FINAL_REWARD  = True

# Meta-Controller-ийн шийдвэр гаргах интервал
MACRO_STEP           = 10

# PPO сургалтын тохиргоонууд
UPDATES              = 200
GROUP_SIZE           = 16
PPO_EPOCHS           = 3
MINI_BATCH_SIZE      = 16

CLIP_EPS             = 0.2

KL_BETA              = 0.04
TARGET_KL            = 0.05
KL_ALPHA             = 1.2

LR_BASE_SFT          = 3e-4
LR_META_RL           = 1e-4
MAX_GRAD_NORM        = 1.0

# Worker-ийн хэмжээсүүд
EMBED_DIM            = 256
NUM_HEADS            = 4
MEMORY_LEN           = 16
WORKER_LAYERS        = 2

# Meta болон Санах ойн хэмжээсүүд
U_DIM                = 128
GRU_H_DIM            = 256

NUM_SLOTS            = 32
SLOT_TOPK            = 1
SLOT_TEMP            = 50.0

EPI_STRENGTH_DECAY   = 0.995
EPI_AGE_PENALTY      = 0.02
EPI_STRENGTH_BOOST   = 0.50
EPI_WRITE_ALPHA      = 0.50

# Санах ойн хуучин мэдээллийг устгах (Eviction) тохиргоо
EVICT_AGE_BOOST      = 0.05   # Age өсөх буюу хуучрах тусам дарж бичигдэх магадлал ихэснэ
EVICT_STR_PENALTY    = 0.50   # strength(чухал) мэдээллийг дарж бичихгүй байх тохиргоо

# Decoder-ийн хэмжээс
LRANK                = 32

# Meta-ийн гаралтын хязгаарууд
SWITCH_TAU           = 0.50
BETA_MIN             = 0.05
BETA_MAX             = 0.995

# Алдааны функцийн (Loss) коэффициентүүд
SWITCH_ENTROPY_COEFF = 0.01
U_ENTROPY_COEFF      = 0.001

TARGET_SW_RATE       = 0.10
SW_RATE_COEFF        = 1.0

BETA_MEAN_TARGET     = 0.85
BETA_MEAN_COEFF      = 0.2
BETA_VAR_COEFF       = 0.1

DECODER_PG_COEFF     = 0.10

# Эхний шатны сургалтын (SFT) тохиргоо
SFT_ENABLE           = True
SFT_FLAG             = "sft_done_fetch_integrated_fixed.flag"
SFT_EPISODES         = 100 #3000
SFT_EPOCHS           = 10
SFT_BATCH_SIZE       = 256


# Environment-ээс ирж буй мэдээллийг (observation) нэг урт вектор болгон хувиргана.
# Ингэснээр Transformer бүтэцрүү оруулахад амар болно.

class TransformerObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        self.tokenizer         = getattr(env, "tokenizer", None)

    def observation(self, obs_dict):
        return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]], dtype=np.float32)


# Тасралтгүй (continuous) үйлдлийг дискрет токен (token) болгож хувиргана.
# Ингэснээр classification хэлбэрээр үйлдлийг таамаглах боломжтой.

class ActionTokenizer:
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


# ActionTokenizer-ийг Gym орчинд хэрэгжүүлэх Wrapper

class TokenActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=256):
        super().__init__(env)
        self.tokenizer = ActionTokenizer(bins=bins)
        act_dim        = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete([bins] * act_dim)

    def action(self, token_action):
        return self.tokenizer.decode(np.array(token_action)).astype(np.float32)


# Роботод хэд хэдэн зорилгыг дараалан өгөх Wrapper.
# Зөвхөн хамгийн сүүлийн зорилгод хүрсэн үед л reward=1 өгнө.

class SequentialGoalsWrapper(gym.Wrapper):
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

        goals = [base_goal.copy()]
        for _ in range(self.k - 1):
            offset = np.random.uniform(-0.10, 0.10, size=3).astype(np.float32)
            g      = base_goal + offset
            g[2]   = max(0.42, float(g[2]))
            goals.append(g.astype(np.float32))

        self._goals    = goals
        self._goal_idx = 0

        info                       = dict(info)
        info["seq_goal_index"    ] = self._goal_idx
        info["distance_threshold"] = self.dist_threshold

        return self._update_obs(obs), info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired  = self._goals[self._goal_idx]
        dist     = float(np.linalg.norm(achieved - desired))
        reached  = dist < self.dist_threshold

        info                   = dict(info)
        info["seq_goal_index"] = self._goal_idx
        info["seq_reached"   ] = reached
        info["seq_dist"      ] = dist

        if reached:
            if self._goal_idx < self.k - 1:
                self._goal_idx        += 1
                info["seq_goal_index"] = self._goal_idx
                info["is_success"    ] = 0.0
                terminated             = False
                reward                 = 0.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info
            else:
                info["is_success"] = 1.0
                terminated         = True
                reward             = 1.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info

        info["is_success"] = 0.0
        reward             = 0.0 if self.sparse_final_reward else float(base_reward)
        return self._update_obs(obs), reward, terminated, truncated, info

def make_single_env(rank, render_mode=None):
    def _thunk():
        env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=MAX_EPISODE_STEPS)
        env = SequentialGoalsWrapper(
            env,
            k                   = K_SEQUENTIAL_GOALS,
            dist_threshold      = GOAL_DIST_THRESHOLD,
            sparse_final_reward = SPARSE_FINAL_REWARD
        )
        env = TokenActionWrapper(env, bins=ACTION_BINS)
        env = TransformerObservationWrapper(env)
        env.reset(seed=SEED + rank)
        return env
    return _thunk

def create_vec_env(n_envs, render_mode=None):
    fns = [make_single_env(i, render_mode=render_mode) for i in range(n_envs)]
    return AsyncVectorEnv(fns)


# PPO-д ашиглагдах магадлалын (logprob) болон Entropy тооцоолох JAX функцүүд

@jax.jit
def gaussian_logprob_bt(x, mean, logstd):
    var = jnp.exp(2.0 * logstd)
    return -0.5 * jnp.sum(((x - mean) ** 2) / (var + 1e-8) + 2.0 * logstd + jnp.log(2.0 * jnp.pi), axis=-1)

@jax.jit
def gaussian_entropy(logstd):
    return jnp.sum(logstd + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))

@jax.jit
def gaussian_kl_bt(mean_new, logstd_new, mean_ref, logstd_ref):
    var_new = jnp.exp(2.0 * logstd_new)
    var_ref = jnp.exp(2.0 * logstd_ref)
    term1   = (logstd_ref - logstd_new)
    term2   = (var_new + (mean_new - mean_ref) ** 2) / (2.0 * var_ref + 1e-8)
    return jnp.sum(term1 + term2 - 0.5, axis=-1)

@jax.jit
def bernoulli_logprob(action01, prob):
    p = jnp.clip(prob, 1e-6, 1.0 - 1e-6)
    a = action01.astype(jnp.float32)
    return a * jnp.log(p) + (1.0 - a) * jnp.log(1.0 - p)

@jax.jit
def bernoulli_entropy(prob):
    p = jnp.clip(prob, 1e-6, 1.0 - 1e-6)
    return -(p * jnp.log(p) + (1.0 - p) * jnp.log(1.0 - p))

@jax.jit
def bernoulli_kl(p_new, p_ref):
    pn = jnp.clip(p_new, 1e-6, 1.0 - 1e-6)
    pr = jnp.clip(p_ref, 1e-6, 1.0 - 1e-6)
    return pn * (jnp.log(pn) - jnp.log(pr)) + (1.0 - pn) * (jnp.log(1.0 - pn) - jnp.log(1.0 - pr))


# Episode явцад чухал мэдээллүүдийг хадгалах санах ойн блок.
# Уншихдаа similarity (төстэй байдал) ашиглах ба бичихдээ чухал биш хуучин мэдээллийг дарж бичнэ.

class EpisodicSlotMemoryBlock(nn.Module):
    num_slots      : int   = NUM_SLOTS
    embed_dim      : int   = GRU_H_DIM
    strength_decay : float = EPI_STRENGTH_DECAY
    age_penalty    : float = EPI_AGE_PENALTY
    strength_boost : float = EPI_STRENGTH_BOOST
    write_alpha    : float = EPI_WRITE_ALPHA
    write_temp     : float = SLOT_TEMP
    write_topk     : int   = SLOT_TOPK

    @nn.compact
    def __call__(self, query_vec, write_vec, write_strength, mem_state):
        keys, vals, age, strength = mem_state
        B, K, D                   = keys.shape

        # Унших үйлдэл: Асуулга (query) болон түлхүүрүүдийн (keys) төстэй байдлыг тооцоолох
        qn  = query_vec / (jnp.linalg.norm(query_vec, axis=-1, keepdims=True) + 1e-6)
        kn  = keys      / (jnp.linalg.norm(keys,      axis=-1, keepdims=True) + 1e-6)

        sim_r  = jnp.sum(kn * qn[:, None, :], axis=-1)
        # Age ихтэй бол оноо хасагдаж, strength сайн мэдээлэл бол оноо нэмэгдэнэ
        logits = sim_r + (self.strength_boost * jnp.log(jnp.clip(strength, 1e-3, 1.0))) - (self.age_penalty * age)

        w_read   = jax.nn.softmax(logits, axis=-1)
        read_out = jnp.sum(w_read[:, :, None] * vals, axis=1)

        # Бичих үйлдэл: Хамгийн тохиромжтой үүрийг (slot) сонгох
        wk_n  = write_vec / (jnp.linalg.norm(write_vec, axis=-1, keepdims=True) + 1e-6)

        sim_w = jnp.sum(kn * wk_n[:, None, :], axis=-1)

        # Eviction буюу дарж бичих логик: Age ихтэй, Strength багатай үүрийг сонгохыг илүүд үзнэ
        evict_score = (EVICT_AGE_BOOST * jnp.log1p(age)) - (EVICT_STR_PENALTY * strength)

        write_logits = (sim_w * self.write_temp) + evict_score
        write_soft   = jax.nn.softmax(write_logits, axis=-1)

        if self.write_topk <= 1:
            top_idx    = jnp.argmax(write_soft, axis=-1)
            write_hard = jax.nn.one_hot(top_idx, K, dtype=jnp.float32)
        else:
            top_idx    = jnp.argsort(write_soft, axis=-1)[:, -self.write_topk:]
            write_hard = jnp.sum(jax.nn.one_hot(top_idx, K, dtype=jnp.float32), axis=1)
            write_hard = write_hard / (jnp.sum(write_hard, axis=-1, keepdims=True) + 1e-6)

        # Straight-Through Estimator (STE) ашиглан градиент дамжуулах
        write_w = jax.lax.stop_gradient(write_hard - write_soft) + write_soft

        # Санах ойг шинэчлэх
        ws       = jnp.clip(write_strength, 0.0, 1.0)
        eff_rate = (write_w * ws) * self.write_alpha
        rate_exp = eff_rate[:, :, None]

        keys_new = (1.0 - rate_exp) * keys + rate_exp * wk_n[:, None, :]
        vals_new = (1.0 - rate_exp) * vals + rate_exp * write_vec[:, None, :]

        # age-ийг бүх үүрэнд нэмэх ба бичилт хийгдсэн үүрний age-ийг 0 болгоно
        age_new = age + 1.0
        age_new = age_new * (1.0 - write_w)

        # strength-ийг багасгах ба бичилт хийгдсэн үүрэнд strength утгыг нэмнэ
        str_new = strength * self.strength_decay
        str_new = str_new + (write_w * ws) * (1.0 - str_new)
        str_new = jnp.clip(str_new, 0.0, 1.0)

        return read_out, (keys_new, vals_new, age_new, str_new)


# Дээд түвшний удирдлага. 
# Богино хугацааны санах ой (GRU) болон урт хугацааны санах ойг (Episodic) хослуулна.

class EpisodicMetaController(nn.Module):
    u_dim     : int = U_DIM
    h_dim     : int = GRU_H_DIM
    num_slots : int = NUM_SLOTS

    def setup(self):
        self.obs_proj    = nn.Dense(128)
        self.ssum_proj   = nn.Dense(128)

        self.encoder     = nn.GRUCell(features=self.h_dim)

        self.epi_memory  = EpisodicSlotMemoryBlock(num_slots=self.num_slots, embed_dim=self.h_dim)
        self.write_gate  = nn.Dense(1)

        self.to_mean     = nn.Dense(self.u_dim)
        self.u_logstd    = self.param("u_logstd", nn.initializers.constant(-1.0), (self.u_dim,))

        self.beta_head   = nn.Dense(1)
        self.switch_head = nn.Dense(1)

    def init_hidden(self, batch_size):
        return jnp.zeros((batch_size, self.h_dim), dtype=jnp.float32)

    def init_memory(self, batch_size):
        k = jnp.zeros((batch_size, self.num_slots, self.h_dim), dtype=jnp.float32)
        v = jnp.zeros((batch_size, self.num_slots, self.h_dim), dtype=jnp.float32)
        a = jnp.ones ((batch_size, self.num_slots), dtype=jnp.float32) * 100.0
        s = jnp.zeros((batch_size, self.num_slots), dtype=jnp.float32)
        return (k, v, a, s)

    def __call__(
        self,
        h_prev,
        mem_prev,
        obs,
        ssum,
        u_int_prev,
        step_in_macro,
        forced_action=None,
        force_macro_switch=True,
        rng=None
    ):
        # Env state болон Worker-ийн товч мэдээллийг (ssum) нэгтгэнэ
        xo   = nn.tanh(self.obs_proj(obs))
        xs   = nn.tanh(self.ssum_proj(ssum))
        x_in = jnp.concatenate([xo, xs], axis=-1)

        # GRU ашиглан богино хугацааны state шинэчилнэ
        h_t, _ = self.encoder(h_prev, x_in)

        # Episodic санах ойтой харилцах
        w_str = nn.sigmoid(self.write_gate(h_t))

        mem_read, mem_next = self.epi_memory(
            query_vec      = h_t,
            write_vec      = h_t,
            write_strength = w_str,
            mem_state      = mem_prev
        )

        # Цаашдын шийдвэр гаргах толгойнууд (Heads)
        ctx    = jnp.concatenate([h_t, mem_read], axis=-1)
        mean   = self.to_mean(ctx)
        logstd = self.u_logstd

        # Өмнөх зорилгыг хэр зэрэг хадгалах вэ гэдгийг тодорхойлох (Beta)
        beta_l = self.beta_head(ctx).squeeze(-1)
        beta01 = nn.sigmoid(beta_l)
        beta   = BETA_MIN + (BETA_MAX - BETA_MIN) * beta01

        # Шинэ зорилго руу шилжих эсэхийг шийдэх (Switch)
        sw_l   = self.switch_head(ctx).squeeze(-1)
        sw_prob= nn.sigmoid(sw_l)

        # Машин сургалтын явцад teacher forcing хийх эсвэл шинээр action сонгох
        if forced_action is not None:
            u_prop, sw = forced_action
            switch     = sw.astype(jnp.int32)
        else:
            if rng is None:
                u_prop  = mean
                switch  = (sw_prob > SWITCH_TAU).astype(jnp.int32)
            else:
                rng, k1, k2 = jax.random.split(rng, 3)
                eps         = jax.random.normal(k1, shape=mean.shape)
                u_prop      = mean + jnp.exp(logstd)[None, :] * eps
                switch      = jax.random.bernoulli(k2, sw_prob).astype(jnp.int32)

        # Макро алхмын эхлэл дээр заавал шинэ зорилго өгөхийг албадлах
        if force_macro_switch:
            is_boundary = (step_in_macro == 0).astype(jnp.int32)
            switch      = jnp.maximum(switch, is_boundary)

        # Өмнөх болон шинэ зорилгыг нэгтгэх
        sw_f     = switch[:, None].astype(jnp.float32)
        beta_eff = beta[:, None] * (1.0 - sw_f)

        u_int = (beta_eff * u_int_prev) + ((1.0 - beta_eff) * u_prop)

        aux = {
            "mean"      : mean,
            "logstd"    : logstd,
            "beta"      : beta,
            "beta_eff"  : beta_eff.squeeze(-1),
            "sw_prob"   : sw_prob,
        }

        return h_t, mem_next, u_prop, switch, u_int, aux, rng


# Meta-ийн гаргасан зорилгыг (u_int) Worker-ийн ойлгох вектор (delta) болгож хувиргана.

class ResidualIntervention(nn.Module):
    embed_dim: int = EMBED_DIM
    rank     : int = LRANK
    u_dim    : int = U_DIM

    def setup(self):
        self.mlp1 = nn.Dense(self.embed_dim)
        self.mlp2 = nn.Dense(self.embed_dim)

        self.to_A = nn.Dense(self.embed_dim * self.rank)
        self.to_B = nn.Dense(self.rank * self.embed_dim)
        self.to_b = nn.Dense(self.embed_dim)

    def __call__(self, u_int, ssum):
        x = nn.tanh(self.mlp1(u_int))
        x = nn.tanh(self.mlp2(x))

        Bsz = u_int.shape[0]

        A = jnp.tanh(self.to_A(x).reshape(Bsz, self.embed_dim, self.rank)) * 0.5
        B = jnp.tanh(self.to_B(x).reshape(Bsz, self.rank, self.embed_dim)) * 0.5
        b = jnp.tanh(self.to_b(x)) * 0.5

        y     = jnp.matmul(ssum[:, None, :], A).squeeze(1)
        delta = jnp.matmul(y[:, None, :], B).squeeze(1) + b
        return delta


# Transformer-ийн нэг блок. Өмнөх алхмуудын мэдээллийг (mem) ашиглан одоогийн төлөвийг баяжуулна.

class TXLWorkerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mem, deterministic=True):
        kv   = jnp.concatenate([mem, x[:, None, :]], axis=1)

        y    = nn.LayerNorm()(x)
        kvn  = nn.LayerNorm()(kv)
        attn = nn.MultiHeadDotProductAttention(
            num_heads     = self.num_heads,
            dropout_rate  = 0.0,
            deterministic = deterministic
        )(inputs_q=y[:, None, :], inputs_kv=kvn).squeeze(1)

        x = x + attn

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim)(y)
        x = x + y
        return x


# Доод түвшний гүйцэтгэгч буюу Worker.
# Env мэдээлэл болон Decoder-ээс ирсэн delta-г ашиглан шууд үйлдэл (action) гаргана.

class TXLWorkerPolicy(nn.Module):
    action_dim : int = ACTION_DIM
    action_bins: int = ACTION_BINS
    embed_dim  : int = EMBED_DIM
    mem_len    : int = MEMORY_LEN
    num_layers : int = WORKER_LAYERS

    def setup(self):
        self.input_proj  = nn.Dense(self.embed_dim)
        self.time_embed  = nn.Embed(MACRO_STEP + 1, self.embed_dim)
        self.blocks      = [TXLWorkerBlock(self.embed_dim, NUM_HEADS) for _ in range(self.num_layers)]
        self.action_head = nn.Dense(self.action_dim * self.action_bins)

    def init_carry(self, batch_size):
        return (
            jnp.zeros((batch_size, self.mem_len, self.embed_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, self.embed_dim), dtype=jnp.float32),
        )

    def __call__(self, obs, carry, step_in_macro, delta, deterministic=True):
        mem, ssum = carry

        x     = self.input_proj(obs)
        t_emb = self.time_embed(jnp.clip(step_in_macro, 0, MACRO_STEP))
        # Decoder-ээс ирсэн delta-г энд нэмж өгнө
        x     = x + t_emb + delta

        for blk in self.blocks:
            x = blk(x, mem, deterministic=deterministic)

        # Worker-ийн богино хугацааны санах ойг (FIFO) шинэчлэх
        mem_new = jnp.concatenate([mem[:, 1:, :], x[:, None, :]], axis=1)

        # Товч мэдээллийг (ssum) EMA (Exponential Moving Average) ашиглан шинэчлэх
        lam      = 0.90
        ssum_new = (ssum * lam) + (x * (1.0 - lam))

        logits = self.action_head(x).reshape(obs.shape[0], self.action_dim, self.action_bins)
        return logits, (mem_new, ssum_new), x


# Бүх модулиудыг нэгтгэсэн агент
class HierarchicalAgent:
    def __init__(self):
        self.worker = TXLWorkerPolicy()
        self.meta   = EpisodicMetaController()
        self.interv = ResidualIntervention()


# Машин сургах класс, агентын бүх жинг эхлүүлж, optimizer-тэй холбоно.

class Trainer:
    def __init__(self, seed=SEED):
        self.rng   = jax.random.PRNGKey(seed)
        self.agent = HierarchicalAgent()

        self.rng, k1, k2, k3, k4 = jax.random.split(self.rng, 5)

        dummy_obs   = jnp.zeros((1, 28), dtype=jnp.float32)
        dummy_step  = jnp.array([0], dtype=jnp.int32)

        dummy_carry = self.agent.worker.init_carry(1)
        dummy_delta = jnp.zeros((1, EMBED_DIM), dtype=jnp.float32)

        worker_params = self.agent.worker.init(k1, dummy_obs, dummy_carry, dummy_step, dummy_delta, True)["params"]

        dummy_h   = self.agent.meta.init_hidden(1)
        dummy_mem = self.agent.meta.init_memory(1)
        dummy_u   = jnp.zeros((1, U_DIM), dtype=jnp.float32)

        meta_params = self.agent.meta.init(
            k2,
            dummy_h,
            dummy_mem,
            dummy_obs,
            dummy_carry[1],
            dummy_u,
            dummy_step,
            forced_action=None,
            force_macro_switch=True,
            rng=k3
        )["params"]

        interv_params = self.agent.interv.init(k4, dummy_u, dummy_carry[1])["params"]

        self.worker_state = train_state.TrainState.create(
            apply_fn = self.agent.worker.apply,
            params   = worker_params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR_BASE_SFT),
            )
        )

        self.meta_state = train_state.TrainState.create(
            apply_fn = None,
            params   = {"meta": meta_params, "interv": interv_params},
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR_META_RL),
            )
        )


# Worker-ийг эхний ээлжинд сургах (SFT) зорилгоор энгийн логик ашиглан өгөгдөл цуглуулах.

def generate_sft_dataset(env, episodes=SFT_EPISODES, print_every_steps=20000, print_every_eps=50):
    obs_list = []
    act_list = []

    total_steps = 0
    t0 = time.time()

    for ep in range(int(episodes)):
        obs, info = env.reset()
        obs       = np.asarray(obs, dtype=np.float32)

        ep_steps = 0

        for _t in range(MAX_EPISODE_STEPS):
            grip_pos  = obs[:3]
            block_pos = obs[3:6]
            goal_pos  = obs[-3:]

            dist_block = np.linalg.norm(grip_pos - block_pos)

            if dist_block > 0.06:
                target = block_pos.copy()
                target[2] += 0.02
            else:
                target = goal_pos.copy()

            delta = (target - grip_pos) * 5.0
            delta = np.clip(delta, -1.0, 1.0)

            cont_action = np.append(delta, -1.0).astype(np.float32)
            tokens      = env.tokenizer.encode(cont_action).astype(np.int32)

            obs_list.append(obs.astype(np.float32))
            act_list.append(tokens.astype(np.int32))

            obs, r, term, trunc, info = env.step(tokens)
            obs = np.asarray(obs, dtype=np.float32)

            ep_steps += 1
            total_steps += 1

            if (print_every_steps is not None) and (total_steps % int(print_every_steps) == 0):
                dt = max(1e-6, time.time() - t0)
                sps = total_steps / dt
                remain_steps = max(0, int(episodes) * MAX_EPISODE_STEPS - total_steps)
                eta_sec = remain_steps / max(1e-6, sps)
                print(f"[SFT-GEN] steps={total_steps:,} | sps={sps:.1f} | ETA={eta_sec/60.0:.1f}m")

            if term or trunc:
                break

        if (print_every_eps is not None) and ((ep + 1) % int(print_every_eps) == 0):
            dt = max(1e-6, time.time() - t0)
            sps = total_steps / dt
            print(f"[SFT-GEN] ep={ep+1}/{int(episodes)} | ep_steps={ep_steps} | total_steps={total_steps:,} | sps={sps:.1f}")

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int32)


# Цуглуулсан өгөгдөл дээр Worker-ийг SFT-ээр сургана. Meta болон Decoder оролцохгүй.

def sft_train(trainer, obs_all, act_all):
    N   = int(len(obs_all))
    idx = np.arange(N)

    @jax.jit
    def sft_step(state, obs_b, act_b):
        def loss_fn(p):
            B      = obs_b.shape[0]
            carry0 = trainer.agent.worker.init_carry(B)

            step0  = jnp.zeros((B,), dtype=jnp.int32)
            delta0 = jnp.zeros((B, EMBED_DIM), dtype=jnp.float32)

            logits, _, _ = trainer.agent.worker.apply(
                {"params": p},
                obs_b,
                carry0,
                step0,
                delta0,
                False
            )

            logp = jax.nn.log_softmax(logits, axis=-1)
            oh   = jax.nn.one_hot(act_b, ACTION_BINS)
            loss = -jnp.sum(oh * logp, axis=-1).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    for ep in range(int(SFT_EPOCHS)):
        np.random.shuffle(idx)
        losses = []

        for i in range(0, N - SFT_BATCH_SIZE + 1, SFT_BATCH_SIZE):
            b     = idx[i:i + SFT_BATCH_SIZE]
            obs_b = jnp.asarray(obs_all[b], dtype=jnp.float32)
            act_b = jnp.asarray(act_all[b], dtype=jnp.int32)

            trainer.worker_state, loss = sft_step(trainer.worker_state, obs_b, act_b)
            losses.append(loss)

        print(f"[SFT] Epoch {ep:2d} | Loss: {float(jnp.mean(jnp.array(losses))):.6f}")

def score_from_info_dict(info_d):
    if info_d is None:
        return 0.0
    seq_idx = float(info_d.get("seq_goal_index", 0.0))
    succ    = float(info_d.get("is_success", 0.0))
    return (seq_idx * 10.0) + (50.0 if succ > 0.5 else 0.0)

def extract_final_infos(infos, B):
    final_info = [None] * B

    if isinstance(infos, dict):
        if "final_info" in infos:
            fi = infos["final_info"]
            try:
                for i in range(B):
                    final_info[i] = fi[i]
            except Exception:
                pass
        elif "final_infos" in infos:
            fi = infos["final_infos"]
            try:
                for i in range(B):
                    final_info[i] = fi[i]
            except Exception:
                pass

    elif isinstance(infos, (list, tuple)):
        for i in range(min(B, len(infos))):
            final_info[i] = infos[i]

    return final_info


# Агентын нэг алхмын шийдвэр гаргах үйл явц (Rollout үед дуудагдана)

@jax.jit
def policy_step_vec(worker_params, meta_params, interv_params, obs_j, carry, h, mem, u_int, step_j, rng):
    worker = TXLWorkerPolicy()
    meta   = EpisodicMetaController()
    interv = ResidualIntervention()

    rng, k_meta, k_act = jax.random.split(rng, 3)

    ssum_j = carry[1]

    h2, mem2, u_prop, sw, u_int2, aux, _ = meta.apply(
        {"params": meta_params},
        h, mem, obs_j, ssum_j, u_int, step_j,
        forced_action      = None,
        force_macro_switch = True,
        rng                = k_meta
    )

    delta = interv.apply({"params": interv_params}, u_int2, ssum_j)

    logits, carry2, _ = worker.apply(
        {"params": worker_params},
        obs_j, carry, step_j, delta,
        False
    )

    logp_w = jax.nn.log_softmax(logits, axis=-1)
    B      = obs_j.shape[0]

    keys = jax.random.split(k_act, B * ACTION_DIM).reshape(B, ACTION_DIM, 2)

    def sample_tok(k, lp_vec):
        return jax.random.categorical(k, lp_vec, axis=-1)

    acts = jax.vmap(lambda kb, lpb: jax.vmap(sample_tok)(kb, lpb))(keys, logp_w).astype(jnp.int32)

    lp_u  = gaussian_logprob_bt(u_prop, aux["mean"], aux["logstd"])
    lp_sw = bernoulli_logprob(sw, aux["sw_prob"])
    lp_m  = lp_u + lp_sw

    return h2, mem2, u_int2, carry2, acts, u_prop, sw, lp_m, aux["sw_prob"], aux["beta"], rng


# Олон орчноос зэрэг өгөгдөл цуглуулах (Rollout)

def collect_rollouts(vec_env, trainer):
    obs, infos = vec_env.reset(seed=SEED)
    obs        = np.asarray(obs, dtype=np.float32)
    B          = obs.shape[0]

    carry = trainer.agent.worker.init_carry(B)
    h     = trainer.agent.meta.init_hidden(B)
    mem   = trainer.agent.meta.init_memory(B)
    u_int = jnp.zeros((B, U_DIM), dtype=jnp.float32)

    step_m = np.zeros((B,), dtype=np.int32)

    obs_buf   = [[] for _ in range(B)]
    ssum_buf  = [[] for _ in range(B)]
    step_buf  = [[] for _ in range(B)]
    uprop_buf = [[] for _ in range(B)]
    sw_buf    = [[] for _ in range(B)]
    decm_buf  = [[] for _ in range(B)]
    act_buf   = [[] for _ in range(B)]
    oldlp_buf = [[] for _ in range(B)]

    done_mask = np.zeros((B,), dtype=bool)
    final_scores = np.zeros((B,), dtype=np.float32)

    worker_params = trainer.worker_state.params
    meta_params   = trainer.meta_state.params["meta"  ]
    interv_params = trainer.meta_state.params["interv"]

    steps = 0
    while (not np.all(done_mask)) and (steps < MAX_EPISODE_STEPS):
        obs_j  = jnp.asarray(obs, dtype=jnp.float32)
        step_j = jnp.asarray(step_m, dtype=jnp.int32)

        ssum_j = np.asarray(carry[1], dtype=np.float32)

        h, mem, u_int, carry, acts, u_prop, sw, lp_m, sw_prob, beta, trainer.rng = policy_step_vec(
            worker_params, meta_params, interv_params,
            obs_j, carry, h, mem, u_int, step_j, trainer.rng
        )

        acts_np = np.asarray(acts, dtype=np.int32  )
        sw_np   = np.asarray(sw  , dtype=np.int32  )
        lp_np   = np.asarray(lp_m, dtype=np.float32)

        # decision mask, зөвхөн макро алхмын эхлэл дээр эсвэл шинэ зорилго өгсөн үед л суралцах mask
        decm_np = np.logical_or((step_m == 0), (sw_np > 0.5)).astype(np.float32)

        for i in range(B):
            if not done_mask[i]:
                obs_buf  [i].append(obs[i])
                ssum_buf [i].append(ssum_j[i])
                step_buf [i].append(int(step_m[i]))
                uprop_buf[i].append(np.asarray(u_prop[i], dtype=np.float32))
                sw_buf   [i].append(int(sw_np[i]))
                decm_buf [i].append(float(decm_np[i]))
                act_buf  [i].append(acts_np[i])
                oldlp_buf[i].append(float(lp_np[i]))

        obs, _, term, trunc, infos = vec_env.step(acts_np)
        obs = np.asarray(obs, dtype=np.float32)

        step_done = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)

        if np.any(step_done):
            finfos = extract_final_infos(infos, B)

            just_finished = step_done & (~done_mask)
            for i in np.where(just_finished)[0]:
                final_scores[i] = float(score_from_info_dict(finfos[i]))

            dm = jnp.asarray(just_finished)

            h     = jnp.where(dm[:, None], jnp.zeros_like(h), h)
            u_int = jnp.where(dm[:, None], jnp.zeros_like(u_int), u_int)

            mem = (
                jnp.where(dm[:, None, None], jnp.zeros_like(mem[0]), mem[0]),
                jnp.where(dm[:, None, None], jnp.zeros_like(mem[1]), mem[1]),
                jnp.where(dm[:, None      ], jnp.ones_like (mem[2]) * 100.0, mem[2]),
                jnp.where(dm[:, None      ], jnp.zeros_like(mem[3]), mem[3]),
            )

            carry = (
                jnp.where(dm[:, None, None], jnp.zeros_like(carry[0]), carry[0]),
                jnp.where(dm[:, None      ], jnp.zeros_like(carry[1]), carry[1]),
            )

            step_m[just_finished] = 0

        done_mask = done_mask | step_done
        steps    += 1
        step_m    = (step_m + 1) % MACRO_STEP

    if np.any(~done_mask):
        finfos = extract_final_infos(infos, B)
        for i in np.where(~done_mask)[0]:
            final_scores[i] = float(score_from_info_dict(finfos[i]))

    trajs = []
    for i in range(B):
        trajs.append({
            "obs"      : np.asarray(obs_buf  [i], dtype=np.float32),
            "ssum"     : np.asarray(ssum_buf [i], dtype=np.float32),
            "step"     : np.asarray(step_buf [i], dtype=np.int32  ),
            "u_prop"   : np.asarray(uprop_buf[i], dtype=np.float32),
            "sw"       : np.asarray(sw_buf   [i], dtype=np.int32  ),
            "decmask"  : np.asarray(decm_buf [i], dtype=np.float32),
            "act"      : np.asarray(act_buf  [i], dtype=np.int32  ),
            "old_logp" : np.asarray(oldlp_buf[i], dtype=np.float32),
        })

    return trajs, final_scores


# GRPO (Group Relative Policy Optimization) аргаар advantage тооцоолох
def compute_grpo_advantages(scores):
    mean = float(np.mean(scores))
    adv  = (scores - mean).astype(np.float32)
    adv  = np.clip(adv, -5.0, 5.0)
    return adv, mean

def pad_batch(trajs):
    B = len(trajs)
    max_t = 0
    for tr in trajs:
        max_t = max(max_t, int(tr["obs"].shape[0]))

    obs   = np.zeros((B, max_t, 28        ), dtype=np.float32)
    ssum  = np.zeros((B, max_t, EMBED_DIM ), dtype=np.float32)
    step  = np.zeros((B, max_t            ), dtype=np.int32  )

    uprop = np.zeros((B, max_t, U_DIM     ), dtype=np.float32)
    sw    = np.zeros((B, max_t            ), dtype=np.int32  )
    decm  = np.zeros((B, max_t            ), dtype=np.float32)

    act   = np.zeros((B, max_t, ACTION_DIM), dtype=np.int32  )
    oldlp = np.zeros((B, max_t            ), dtype=np.float32)

    mask  = np.zeros((B, max_t            ), dtype=np.float32)

    for i, tr in enumerate(trajs):
        T = int(tr["obs"].shape[0])
        if T == 0:
            continue

        obs  [i, :T] = tr["obs"     ]
        ssum [i, :T] = tr["ssum"    ]
        step [i, :T] = tr["step"    ]

        uprop[i, :T] = tr["u_prop"  ]
        sw   [i, :T] = tr["sw"      ]
        decm [i, :T] = tr["decmask" ]

        act  [i, :T] = tr["act"     ]
        oldlp[i, :T] = tr["old_logp"]

        mask [i, :T] = 1.0

    return (
        jnp.asarray(obs,   dtype=jnp.float32),
        jnp.asarray(ssum,  dtype=jnp.float32),
        jnp.asarray(step,  dtype=jnp.int32  ),
        jnp.asarray(uprop, dtype=jnp.float32),
        jnp.asarray(sw,    dtype=jnp.int32  ),
        jnp.asarray(decm,  dtype=jnp.float32),
        jnp.asarray(act,   dtype=jnp.int32  ),
        jnp.asarray(oldlp, dtype=jnp.float32),
        jnp.asarray(mask,  dtype=jnp.float32),
    )


# Teacher forcing ашиглан Meta-Controller-ийн өмнөх санах ой, төлөвүүдийг дахин сэргээх (Scan)

@jax.jit
def meta_replay_scan(params_meta, obs_bt, ssum_bt, step_bt, uprop_bt, sw_bt):
    meta = EpisodicMetaController()

    B, T, _ = obs_bt.shape

    h0     = meta.init_hidden(B)
    mem0   = meta.init_memory(B)
    u_int0 = jnp.zeros((B, U_DIM), dtype=jnp.float32)

    obs_T   = jnp.swapaxes(obs_bt,   0, 1)
    ssum_T  = jnp.swapaxes(ssum_bt,  0, 1)
    step_T  = jnp.swapaxes(step_bt,  0, 1)
    uprop_T = jnp.swapaxes(uprop_bt, 0, 1)
    sw_T    = jnp.swapaxes(sw_bt,    0, 1)

    def scan_fn(carry, x):
        h_prev, mem_prev, u_prev = carry
        obs_t, ssum_t, step_t, uprop_t, sw_t = x

        forced = (uprop_t, sw_t)

        h_t, mem_t, _, _, u_int_t, aux, _ = meta.apply(
            {"params": params_meta},
            h_prev,
            mem_prev,
            obs_t,
            ssum_t,
            u_prev,
            step_t,
            forced_action      = forced,
            force_macro_switch = True,
            rng                = None
        )

        out = (aux["mean"], aux["sw_prob"], aux["beta"], u_int_t)
        return (h_t, mem_t, u_int_t), out

    (_, _, _), outs = jax.lax.scan(scan_fn, (h0, mem0, u_int0), (obs_T, ssum_T, step_T, uprop_T, sw_T))
    mean_T, swprob_T, beta_T, uint_T = outs

    return (
        jnp.swapaxes(mean_T,   0, 1),
        jnp.swapaxes(swprob_T, 0, 1),
        jnp.swapaxes(beta_T,   0, 1),
        jnp.swapaxes(uint_T,   0, 1),
    )


# Worker-ийн гаргасан үйлдлүүдийн магадлалыг (logprob) градиенттэйгаар дахин тооцоолох

@jax.jit
def worker_logprob_rescan(worker_params, interv_params, obs_bt, step_bt, u_int_bt, act_bt, mask_bt):
    worker = TXLWorkerPolicy()
    interv = ResidualIntervention()

    B, T, _ = obs_bt.shape
    carry0 = worker.init_carry(B)

    obs_T  = jnp.swapaxes(obs_bt,   0, 1)
    step_T = jnp.swapaxes(step_bt,  0, 1)
    uint_T = jnp.swapaxes(u_int_bt, 0, 1)
    act_T  = jnp.swapaxes(act_bt,   0, 1)
    m_T    = jnp.swapaxes(mask_bt,  0, 1)

    def scan_fn(carry, x):
        obs_t, step_t, u_t, a_t, m_t = x
        mem_old, ssum_old = carry

        m_t = m_t.astype(bool)

        delta = interv.apply({"params": interv_params}, u_t, ssum_old)
        logits, (mem_new, ssum_new), _ = worker.apply(
            {"params": worker_params}, obs_t, carry, step_t, delta, False
        )

        logp     = jax.nn.log_softmax(logits, axis=-1)
        sel      = jnp.take_along_axis(logp, a_t[..., None], axis=-1).squeeze(-1)
        lpw_calc = jnp.sum(sel, axis=-1)

        m_vec = m_t[:, None      ]
        m_mem = m_t[:, None, None]

        mem_final  = jnp.where(m_mem, mem_new , mem_old)
        ssum_final = jnp.where(m_vec, ssum_new, ssum_old)

        lpw_final  = jnp.where(m_t, lpw_calc, 0.0)

        return (mem_final, ssum_final), lpw_final

    _, lpw_T = jax.lax.scan(scan_fn, carry0, (obs_T, step_T, uint_T, act_T, m_T))
    return jnp.swapaxes(lpw_T, 0, 1)


# PPO аргаар Meta болон Decoder-ийн жингүүдийг шинэчлэх алхам

@jax.jit
def ppo_update_step(meta_state, ref_params_bundle, worker_params, batch, adv_ep, kl_beta):
    obs, ssum, step, uprop, sw, decm, act, oldlp, mask = batch

    def loss_fn(p_bundle):
        p_meta   = p_bundle["meta"  ]
        p_interv = p_bundle["interv"]

        # доогийн параметрүүдээр Meta-г дахин ажиллуулж (Scan) гаралтуудыг авах
        mean_bt, swprob_bt, beta_bt, u_int_bt = meta_replay_scan(p_meta, obs, ssum, step, uprop, sw)

        # Reference (хуучин) параметрүүдээр бас ажиллуулж, KL тооцоход ашиглана
        mean_ref_bt, swprob_ref_bt, _, _ = meta_replay_scan(ref_params_bundle["meta"], obs, ssum, step, uprop, sw)

        # Mask бэлтгэх, m_dec нь зөвхөн шийдвэр гаргасан алхмууд (Decision)
        m_all = mask
        m_dec = mask * decm

        # LogProb тооцоолол
        logstd_new = p_meta["u_logstd"]
        logstd_ref = ref_params_bundle["meta"]["u_logstd"]

        lp_u_new  = gaussian_logprob_bt(uprop, mean_bt, logstd_new)
        lp_sw_new = bernoulli_logprob(sw, swprob_bt)
        lp_new    = lp_u_new + lp_sw_new

        ratio     = jnp.exp(lp_new - oldlp)

        # PPO Surrogate Loss (NaN үүсэхээс сэргийлж jnp.where ашиглах хэрэгтэй)
        adv_bt    = adv_ep[:, None]
        unclipped = ratio * adv_bt
        clipped   = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_bt
        surr      = jnp.minimum(unclipped, clipped)

        # Хоосон (padding) алхмууд дээр NaN гарч болзошгүй тул
        # шууд үржүүлэхийн оронд jnp.where ашиглан шүүж авна.
        surr_mean = jnp.sum(jnp.where(m_dec > 0, surr, 0.0)) / (jnp.sum(m_dec) + 1e-8)

        # KL Divergence тооцоолол
        kl_u    = gaussian_kl_bt(mean_bt, logstd_new, mean_ref_bt, logstd_ref)
        kl_sw   = bernoulli_kl(swprob_bt, swprob_ref_bt)
        kl_bt   = kl_u + kl_sw
        # KL дээр мөн адил jnp.where ашиглана
        kl_mean = jnp.sum(jnp.where(m_dec > 0, kl_bt, 0.0)) / (jnp.sum(m_dec) + 1e-8)

        # Switch Rate Penalty (Зорилго солих давтамжийг тохируулах)
        sw_mean = jnp.sum(jnp.where(m_all > 0, swprob_bt, 0.0)) / (jnp.sum(m_all) + 1e-8)
        sw_pen  = (sw_mean - TARGET_SW_RATE) ** 2

        # Beta Penalty (Beta тархалтыг хэт туйлшрахаас сэргийлэх)
        beta_mean = jnp.sum(jnp.where(m_all > 0, beta_bt, 0.0)) / (jnp.sum(m_all) + 1e-8)
        beta_var  = jnp.sum(jnp.where(m_all > 0, (beta_bt - beta_mean) ** 2, 0.0)) / (jnp.sum(m_all) + 1e-8)
        beta_pen  = (BETA_MEAN_COEFF * (beta_mean - BETA_MEAN_TARGET) ** 2) + (BETA_VAR_COEFF * beta_var)

        # Entropy Bonus (explore хийхийг дэмжих)
        ent_u     = gaussian_entropy(logstd_new)
        ent_sw    = jnp.sum(jnp.where(m_all > 0, bernoulli_entropy(swprob_bt), 0.0)) / (jnp.sum(m_all) + 1e-8)
        ent_bonus = (U_ENTROPY_COEFF * ent_u) + (SWITCH_ENTROPY_COEFF * ent_sw)

        # Decoder-ийн сургалт (Worker-ийн үйлдлээс градиент авах)
        lpw_bt = worker_logprob_rescan(worker_params, p_interv, obs, step, u_int_bt, act, mask)
        # Decoder PG дээр мөн адил шүүлт хийнэ
        dec_pg = jnp.sum(jnp.where(m_dec > 0, lpw_bt * adv_bt, 0.0)) / (jnp.sum(m_dec) + 1e-8)

        # Нийт Loss функц
        loss = (-surr_mean) + (kl_beta * kl_mean) + (SW_RATE_COEFF * sw_pen) + beta_pen - ent_bonus - (DECODER_PG_COEFF * dec_pg)

        aux = (kl_mean, ent_u, ent_sw, sw_mean, beta_mean, beta_var, dec_pg, surr_mean)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(meta_state.params)
    meta_state = meta_state.apply_gradients(grads=grads)
    return meta_state, loss, aux


def ppo_epoch(trainer, ref_params_bundle, trajs, advs, kl_beta):
    idx = np.arange(len(trajs))
    np.random.shuffle(idx)

    last_stats = None

    for i in range(0, len(trajs), MINI_BATCH_SIZE):
        mb_idx   = idx[i:i + MINI_BATCH_SIZE]
        mb_trajs = [trajs[j] for j in mb_idx]
        mb_adv   = jnp.asarray(advs[mb_idx], dtype=jnp.float32)

        batch = pad_batch(mb_trajs)

        trainer.meta_state, loss, aux = ppo_update_step(
            trainer.meta_state,
            ref_params_bundle,
            trainer.worker_state.params,
            batch,
            mb_adv,
            kl_beta
        )

        last_stats = (loss, aux)

    return last_stats


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    trainer = Trainer(seed=SEED)

    if SFT_ENABLE and (not os.path.exists(SFT_FLAG)):
        print("[SFT] Heuristic өгөгдөл үүсгэж байна...")
        sft_env          = make_single_env(0, render_mode=None)()
        obs_all, act_all = generate_sft_dataset(sft_env, episodes=SFT_EPISODES)
        sft_env.close()

        print(f"[SFT] N={len(obs_all)} | Batch={SFT_BATCH_SIZE}")
        sft_train(trainer, obs_all, act_all)

        #with open(SFT_FLAG, "w") as f:
        #    f.write("done")

    vec_env = create_vec_env(GROUP_SIZE, render_mode=None)
    kl_beta = float(KL_BETA)

    print("\n" + "=" * 72)
    print("  INTERNAL RL - Worker Frozen, Meta+Decoder PPO+GRPO (EPISODIC META)")
    print(f"  Updates: {UPDATES} | Group: {GROUP_SIZE} | MacroMax: {MACRO_STEP}")
    print(f"  Embed: {EMBED_DIM} | U_DIM: {U_DIM} | GRU_H: {GRU_H_DIM} | Slots: {NUM_SLOTS}")
    print("=" * 72 + "\n")

    for upd in range(1, UPDATES + 1):
        ref_params_bundle = trainer.meta_state.params

        trajs, scores     = collect_rollouts(vec_env, trainer)

        advs, mean_score  = compute_grpo_advantages(scores)

        loss, aux = None, None
        for _ in range(PPO_EPOCHS):
            loss, aux = ppo_epoch(trainer, ref_params_bundle, trajs, advs, kl_beta)

        kl_v, ent_u, ent_sw, sw_m, beta_m, beta_v, dec_pg, surr_m = [float(x) for x in aux]

        if kl_v > TARGET_KL * 1.5:
            kl_beta *= KL_ALPHA
        elif kl_v < TARGET_KL / 1.5:
            kl_beta /= KL_ALPHA

        if upd % 5 == 0 or upd == 1:
            best = float(np.max(scores)) if len(scores) else 0.0
            print(
                f"[UPD {upd:4d}] "
                f"Score: {mean_score:6.2f} | Best: {best:6.2f} | "
                f"Loss: {float(loss):.4f} | KL: {kl_v:.4f} | KL_B: {kl_beta:.3f} | "
                f"SW: {sw_m:.2f} | Beta: {beta_m:.2f} | DecPG: {dec_pg:.3f}"
            )

    vec_env.close()
    print("DONE")

if __name__ == "__main__":
    main()