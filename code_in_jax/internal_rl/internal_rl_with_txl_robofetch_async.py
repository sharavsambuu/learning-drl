#
# INTERNAL RL - ROBOTIC FETCH, META-CONTROLLER + RESIDUAL INTERVENTION 
#
#
#   АРХИТЕКТУРЫН ТАЙЛБАР:
#
#   Энэ код нь FetchPush-v4 робот орчинд "Дотоод RL" (Internal RL) аргачлалаар
#   шатлалтай (Hierarchical) удирдлага хэрэгжүүлж үзэх туршилт юм.
#
#
#   БҮРЭЛДЭХҮҮН ХЭСГҮҮД:
#
#   1. BaseARPolicy (Worker - Autoregressive):
#      - Роботын үе мөчний хөдөлгөөнийг (Joint Actions) шууд удирдана.
#      - Transformer-XL (TXL) архитектуртай төстэй, өнгөрсөн санах ойтой (Mem).
#      - RL шатанд жингүүд нь царцсан (frozen) байна.
#      - Гаднаас орж ирэх Delta (Δ) вектороор дамжуулан "бодлоо" өөрчилнө.
#
#   2. MetaController (Manager - Episodic Memory):
#      - Env-г ажиглаж, стратегийн зорилго (Intention u_prop) гаргана.
#      - Episodic Slot Memory ашиглан урт хугацааны туршлагыг хадгална.
#      - Switch Head          : Шинэ команд өгөх эсэхийг шийднэ.
#      - Temporal Integration : Өмнөх бодлоо Beta (β) хувиар хадгалж, шинэ командтай холино.
#
#   3. ResidualIntervention (Decoder):
#      - Meta-гаас гарсан u_int кодыг Low-Rank матриц (A, B) ашиглан
#        Worker-ийн ойлгох хэл буюу Delta (Δ) вектор руу хөрвүүлнэ.
#
#
#   МАШИН СУРГАЛТЫН ҮЕ ШАТ (PHASES):
#
#     PHASE 1: SFT (Supervised Fine-Tuning)
#       - Worker-ийг Heuristic өгөгдөл дээр сургаж, анхан шатны хөдөлгөөний эвсэл суулгана.
#
#     PHASE 2: Internal RL (PPO + GRPO)
#       - Worker-ийг царцааж, зөвхөн Meta болон Decoder-ийг сургана.
#       - PPO (Proximal Policy Optimization) ашиглан тогтвортой сургана.
#       - GRPO (Group Relative Policy Optimization) ашиглан Advantage тооцно.
#
#


import os
# JAX санах ойн менежментийг сайжруулах
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import math
import time
import random
import numpy               as np
import gymnasium           as gym
import gymnasium_robotics
from   gymnasium.vector    import SyncVectorEnv
import jax
import jax.numpy           as jnp
import jax.tree_util       as jtu  # JAX v0.6+ нийцтэй байдах хэрэглэнэ
import flax.linen          as nn
import optax
from   flax.training       import train_state



# ЕРӨНХИЙ ТОХИРГОО (HYPERPARAMETERS)

SEED                 = 42

ENV_ID               = "FetchPush-v4"
MAX_EPISODE_STEPS    = 200

ACTION_DIM           = 4
ACTION_BINS          = 256

# Даалгаврын тохиргоо (Sequential Goals)
K_SEQUENTIAL_GOALS   = 3

# Meta Controller шийдвэр гаргах давтамж
MACRO_STEP           = 10

# Машин сургалтын параметрүүд
UPDATES              = 500
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

# Моделийн хэмжээсүүд
EMBED_DIM            = 256
NUM_HEADS            = 4
MEMORY_LEN           = 16
WORKER_LAYERS        = 2

U_DIM                = 128
GRU_H_DIM            = 256
LRANK                = 32

# Санах ойн тохиргоо (Episodic Memory)
NUM_SLOTS            = 32
SLOT_TOPK            = 1
SLOT_TEMP            = 50.0

EPI_STRENGTH_DECAY   = 0.995
EPI_AGE_PENALTY      = 0.02
EPI_STRENGTH_BOOST   = 0.50
EPI_WRITE_ALPHA      = 0.50

EVICT_AGE_BOOST      = 0.05
EVICT_STR_PENALTY    = 0.50

# Meta гаралтын хязгаарууд
SWITCH_TAU           = 0.50
BETA_MIN             = 0.05
BETA_MAX             = 0.995

# Loss функцийн коэффицентүүд
SWITCH_ENTROPY_COEFF = 0.01
U_ENTROPY_COEFF      = 0.001

TARGET_SW_RATE       = 0.10
SW_RATE_COEFF        = 1.0

BETA_MEAN_TARGET     = 0.85
BETA_MEAN_COEFF      = 0.2
BETA_VAR_COEFF       = 0.1

DECODER_PG_COEFF     = 0.10

# Тогтвортой байдлыг хангах хязгаар утгууд (Numerical Stability Clamps)
U_MEAN_SCALE         = 2.0
U_PROP_SCALE         = 2.0
U_INT_CLIP           = 3.0
DELTA_CLIP           = 2.0

# SFT (Supervised Fine-Tuning)
SFT_ENABLE           = True
SFT_FLAG             = "sft_done_fetch.flag"
SFT_EPISODES         = 3000
SFT_EPOCHS           = 10
SFT_BATCH_SIZE       = 256

# Тоглуулж харах утгууд (Periodic Play)
PLAY_ENABLE          = True
PLAY_EVERY           = 10     # Хэдэн update тутамд render хийх вэ
PLAY_EPISODES        = 2
PLAY_SLEEP_SEC       = 0.0



# ENV WRAPPERS 

class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Observation Dict-ийг хавтгай вектор (flat vector) болгон хувиргана.
    NaN болон Infinity утгуудаас хамгаалах (Sanitization) давхаргатай.
    """
    def __init__(self, env, clip_abs=10.0):
        super().__init__(env)
        self.clip_abs          = float(clip_abs)
        self.observation_space = gym.spaces.Box(
            low   =-np.inf,
            high  = np.inf,
            shape = (28,),
            dtype = np.float32,
        )
        self.tokenizer = getattr(env, "tokenizer", None)

    def observation(self, obs_dict):
        x = np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"]],
            axis=0,
        ).astype(np.float32)

        # NaN/Inf утгуудыг 0-ээр солих (Маш чухал хамгаалалт)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.clip_abs > 0:
            x = np.clip(x, -self.clip_abs, self.clip_abs).astype(np.float32)

        return x


class ActionTokenizer:
    """
    Continuous action <-> Discrete token хөрвүүлэгч.
    Domain: [-1, 1] <-> [0, BINS-1]
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
    """
    def __init__(self, env, bins=256):
        super().__init__(env)
        self.tokenizer    = ActionTokenizer(bins=bins)
        act_dim           = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete([bins] * act_dim)

    def action(self, token_action):
        # Action-г хязгаарт нь барих (Safety clip)
        token_action = np.clip(np.asarray(token_action, dtype=np.int32), 0, self.tokenizer.bins - 1)
        return self.tokenizer.decode(token_action).astype(np.float32)


class SequentialGoalsWrapper(gym.Wrapper):
    """
    FetchPush даалгаврыг Pinpad буюу дараалсан K зорилготой болгох.
    NaN болон "нисдэг блок" (physics explosion)-оос хамгаалсан логиктой.
    """
    def __init__(self, env, k=3, shaping_enable=True, sparse_final_reward=True, **kwargs):
        super().__init__(env)
        self.k                   = int(k)
        self.shaping_enable      = bool(shaping_enable)
        self.sparse_final_reward = bool(sparse_final_reward)
        
        # Тохиргоонуудыг хадгалах
        self.goal_threshold0      = kwargs.get("goal_threshold0"     , 0.10)
        self.goal_threshold1      = kwargs.get("goal_threshold1"     , 0.10)
        self.goal_threshold_final = kwargs.get("goal_threshold_final", 0.12)
        
        self.goal_jitter          = kwargs.get("goal_jitter"         , 0.02)
        self.z_floor              = kwargs.get("z_floor"             , 0.42)
        self.z_ceil               = kwargs.get("z_ceil"              , 0.80)
        self.xy_clip              = kwargs.get("xy_clip"             , 0.12)
        
        self.shaping_scale        = kwargs.get("shaping_scale"       , 1.0 )
        self.shaping_clip         = kwargs.get("shaping_clip"        , 0.05)
        self.subgoal_bonus        = kwargs.get("subgoal_bonus"       , 0.20)
        self.push_offset          = kwargs.get("push_offset"         , 0.06)

        self._goals               = None
        self._goal_idx            = 0
        self._base_goal           = None
        self._prev_dist           = None

    def _set_current_goal_in_obs(self, obs):
        obs                 = dict(obs)
        obs["desired_goal"] = self._goals[self._goal_idx].copy()
        return obs

    def _safe_goal(self, g):
        g = np.asarray(g, dtype=np.float32).copy()
        if self._base_goal is not None:
            g[0] = np.clip(g[0], self._base_goal[0] - self.xy_clip, self._base_goal[0] + self.xy_clip)
            g[1] = np.clip(g[1], self._base_goal[1] - self.xy_clip, self._base_goal[1] + self.xy_clip)
        g[2] = float(np.clip(g[2], self.z_floor, self.z_ceil))
        return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = np.asarray(obs["desired_goal"], dtype=np.float32)
        self._base_goal = self._safe_goal(base)

        goals = [self._base_goal.copy()]
        for _ in range(self.k - 2):
            off = np.random.uniform(-self.goal_jitter, self.goal_jitter, size=3).astype(np.float32)
            goals.append(self._safe_goal(self._base_goal + off))
        goals.append(self._base_goal.copy())

        self._goals     = goals
        self._goal_idx  = 0
        self._prev_dist = None

        obs = self._set_current_goal_in_obs(obs)
        info = dict(info)
        info["seq_goal_index"] = self._goal_idx
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        obs_vec = np.asarray(obs["observation"], dtype=np.float32)
        grip_pos, block_pos = obs_vec[0:3], obs_vec[3:6]

        # Fail-safe, Блок ширээнээс унавал шууд дуусгана
        if block_pos[2] < 0.40:
            info["is_success"] = 0.0
            return self._set_current_goal_in_obs(obs), -1.0, True, truncated, info

        # Зорилгын зайг тооцох
        if self._goal_idx == 0:
            target = block_pos.copy(); target[2] = max(target[2], self.z_floor)
            dist, thresh = float(np.linalg.norm(grip_pos - target)), self.goal_threshold0
        elif self._goal_idx < (self.k - 1):
            ach = np.asarray(obs["achieved_goal"], dtype=np.float32)
            v   = (self._goals[self.k - 1] - ach).astype(np.float32)
            pp  = ach - (v / (np.linalg.norm(v) + 1e-6)) * self.push_offset
            pp[2] = max(pp[2], self.z_floor)
            dist, thresh = float(np.linalg.norm(grip_pos - pp)), self.goal_threshold1
        else:
            ach = np.asarray(obs["achieved_goal"], dtype=np.float32)
            dist, thresh = float(np.linalg.norm(ach - self._goals[self._goal_idx])), self.goal_threshold_final

        info = dict(info)
        info["seq_goal_index"] = self._goal_idx
        info["seq_dist"      ] = dist

        # Reward Shaping
        shaping_r = 0.0
        if self.shaping_enable and (self._prev_dist is not None):
            shaping_r = float(np.clip((self._prev_dist - dist) * self.shaping_scale, -self.shaping_clip, self.shaping_clip))
        self._prev_dist = dist

        # Зорилгод хүрсэн эсэх
        if dist <= thresh:
            if self._goal_idx < self.k - 1:
                self._goal_idx += 1
                info["seq_goal_index"], info["is_success"] = self._goal_idx, 0.0
                return self._set_current_goal_in_obs(obs), shaping_r + self.subgoal_bonus, False, truncated, info
            else:
                info["is_success"] = 1.0
                r = 1.0 if self.sparse_final_reward else float(base_reward)
                return self._set_current_goal_in_obs(obs), r, True, truncated, info

        info["is_success"] = 0.0
        return self._set_current_goal_in_obs(obs), shaping_r, False, truncated, info



# ENV UTILS

def make_single_env(rank, render_mode=None):
    def _thunk():
        env = gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=MAX_EPISODE_STEPS)
        env = SequentialGoalsWrapper(env, k=K_SEQUENTIAL_GOALS, shaping_enable=True)
        env = TokenActionWrapper(env, bins=ACTION_BINS)
        env = TransformerObservationWrapper(env, clip_abs=10.0)
        env.reset(seed=SEED + rank)
        return env
    return _thunk

def create_vec_env(n_envs, render_mode=None):
    return SyncVectorEnv([make_single_env(i, render_mode=render_mode) for i in range(n_envs)])



# JAX MATH UTILS 

@jax.jit
def gaussian_logprob_bt(x, mean, logstd):
    # LogStd-ийг хязгаарлавал NaN үүсэхээс сэргийлнэ
    logstd = jnp.clip(logstd, -5.0, 2.0)
    var    = jnp.exp(2.0 * logstd)
    return -0.5 * jnp.sum(((x - mean) ** 2) / (var + 1e-8) + 2.0 * logstd + jnp.log(2.0 * jnp.pi), axis=-1)

@jax.jit
def gaussian_entropy(logstd):
    logstd = jnp.clip(logstd, -5.0, 2.0)
    return jnp.sum(logstd + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))

@jax.jit
def gaussian_kl_bt(mean_new, logstd_new, mean_ref, logstd_ref):
    logstd_new = jnp.clip(logstd_new, -5.0, 2.0)
    logstd_ref = jnp.clip(logstd_ref, -5.0, 2.0)
    var_new    = jnp.exp(2.0 * logstd_new)
    var_ref    = jnp.exp(2.0 * logstd_ref)
    term1      = logstd_ref - logstd_new
    term2      = (var_new + (mean_new - mean_ref) ** 2) / (2.0 * var_ref + 1e-8)
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



# NEURAL NETWORKS

class EpisodicSlotMemoryBlock(nn.Module):
    """
    Episodic Memory Block. Query-Key механизмаар ажиллана.
    Eviction policy: Хуучин бөгөөд ашиглалт багатай үүрийг дарж бичнэ.
    """
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
        B, K, D      = keys.shape
        
        # Read Operation
        qn           = query_vec / (jnp.linalg.norm(query_vec, axis=-1, keepdims=True) + 1e-6)
        kn           = keys      / (jnp.linalg.norm(keys,      axis=-1, keepdims=True) + 1e-6)
        
        sim_r        = jnp.sum(kn * qn[:, None, :], axis=-1)
        logits       = sim_r + (self.strength_boost * jnp.log(jnp.clip(strength, 1e-3, 1.0))) - (self.age_penalty * age)
        w_read       = jax.nn.softmax(logits, axis=-1)
        read_out     = jnp.sum(w_read[:, :, None] * vals, axis=1)

        # Write Operation
        wk_n         = write_vec / (jnp.linalg.norm(write_vec, axis=-1, keepdims=True) + 1e-6)
        sim_w        = jnp.sum(kn * wk_n[:, None, :], axis=-1)
        evict_sc     = (EVICT_AGE_BOOST * jnp.log1p(age)) - (EVICT_STR_PENALTY * strength)
        
        write_logits = (sim_w * self.write_temp) + evict_sc
        write_soft   = jax.nn.softmax(write_logits, axis=-1)
        
        top_idx      = jnp.argmax(write_soft, axis=-1)
        write_hard   = jax.nn.one_hot(top_idx, K, dtype=jnp.float32)
        write_w      = jax.lax.stop_gradient(write_hard - write_soft) + write_soft

        ws           = jnp.clip(write_strength, 0.0, 1.0)
        eff_rate     = (write_w * ws) * self.write_alpha
        rate_exp     = eff_rate[:, :, None]

        keys_new     = (1.0 - rate_exp) * keys + rate_exp * wk_n[:, None, :]
        vals_new     = (1.0 - rate_exp) * vals + rate_exp * write_vec[:, None, :]
        
        # Age update, Бичигдсэн үүрний насыг тэглэх, бусдыг нэмэх
        age_new      = (age + 1.0) * (1.0 - write_w)
        
        # Strength update
        str_new      = strength * self.strength_decay + (write_w * ws) * (1.0 - strength * self.strength_decay)
        str_new      = jnp.clip(str_new, 0.0, 1.0)

        return read_out, (keys_new, vals_new, age_new, str_new)


class EpisodicMetaController(nn.Module):
    """
    Meta-Controller. GRU + Episodic Memory ашиглан шийдвэр гаргана.
    """
    u_dim     : int = U_DIM
    h_dim     : int = GRU_H_DIM
    num_slots : int = NUM_SLOTS

    def setup(self):
        self.obs_proj    = nn.Dense(128)
        self.ssum_proj   = nn.Dense(128)
        self.uint_proj   = nn.Dense(128)
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
        return (
            jnp.zeros((batch_size, self.num_slots, self.h_dim), dtype=jnp.float32), # keys
            jnp.zeros((batch_size, self.num_slots, self.h_dim), dtype=jnp.float32), # vals
            jnp.ones ((batch_size, self.num_slots), dtype=jnp.float32) * 100.0,     # age
            jnp.zeros((batch_size, self.num_slots), dtype=jnp.float32)              # strength
        )

    def __call__(self, h_prev, mem_prev, obs, ssum, u_int_prev, step_in_macro, forced_action=None, force_macro_switch=True, rng=None):
        x_in   = jnp.concatenate([nn.tanh(self.obs_proj(obs)), nn.tanh(self.ssum_proj(ssum)), nn.tanh(self.uint_proj(u_int_prev))], axis=-1)
        h_t, _ = self.encoder(h_prev, x_in)
        
        w_str = nn.sigmoid(self.write_gate(h_t))
        mem_read, mem_next = self.epi_memory(query_vec=h_t, write_vec=h_t, write_strength=w_str, mem_state=mem_prev)
        
        ctx = jnp.concatenate([h_t, mem_read], axis=-1)

        mean   = jnp.tanh(self.to_mean(ctx)) * U_MEAN_SCALE
        logstd = jnp.clip(self.u_logstd, -5.0, 2.0)
        
        beta    = BETA_MIN + (BETA_MAX - BETA_MIN) * nn.sigmoid(self.beta_head(ctx).squeeze(-1))
        sw_prob = nn.sigmoid(self.switch_head(ctx).squeeze(-1))

        # Double Tanh-аас сэргийлэх
        if forced_action is not None:
            # Teacher Forcing, Гаднаас ирсэн утгыг шууд авна (аль хэдийн Tanh хийгдсэн)
            u_prop, sw = forced_action
            switch     = sw.astype(jnp.int32)
        else:
            if rng is None:
                u_prop = mean
                switch = (sw_prob > SWITCH_TAU).astype(jnp.int32)
            else:
                rng, k1, k2 = jax.random.split(rng, 3)
                eps    = jax.random.normal(k1, shape=mean.shape)
                u_prop = jnp.tanh(mean + jnp.exp(logstd)[None, :] * eps) * U_PROP_SCALE
                switch = jax.random.bernoulli(k2, sw_prob).astype(jnp.int32)

        if force_macro_switch:
            switch = jnp.maximum(switch, (step_in_macro == 0).astype(jnp.int32))

        beta_eff = beta[:, None] * (1.0 - switch[:, None].astype(jnp.float32))
        u_int    = (beta_eff * u_int_prev) + ((1.0 - beta_eff) * u_prop)
        u_int    = jnp.clip(u_int, -U_INT_CLIP, U_INT_CLIP)

        aux = {"mean": mean, "logstd": logstd, "beta": beta, "sw_prob": sw_prob}
        return h_t, mem_next, u_prop, switch, u_int, aux, rng


class ResidualIntervention(nn.Module):
    """
    Intention (u_int) -> Delta (Action modifier). Hypernetwork.
    """
    embed_dim : int = EMBED_DIM
    rank      : int = LRANK

    def setup(self):
        self.mlp1 = nn.Dense(self.embed_dim)
        self.mlp2 = nn.Dense(self.embed_dim)
        self.to_A = nn.Dense(self.embed_dim * self.rank)
        self.to_B = nn.Dense(self.rank * self.embed_dim)
        self.to_b = nn.Dense(self.embed_dim)

    def __call__(self, u_int, ssum):
        x     = nn.tanh(self.mlp2(nn.tanh(self.mlp1(u_int))))
        Bsz   = u_int.shape[0]
        
        A     = jnp.tanh(self.to_A(x).reshape(Bsz, self.embed_dim, self.rank)) * 0.5
        B     = jnp.tanh(self.to_B(x).reshape(Bsz, self.rank, self.embed_dim)) * 0.5
        b     = jnp.tanh(self.to_b(x)) * 0.5
        
        y     = jnp.matmul(ssum[:, None, :], A).squeeze(1)
        delta = jnp.matmul(y[:, None, :], B).squeeze(1) + b
        
        return jnp.clip(delta, -DELTA_CLIP, DELTA_CLIP)


class TXLWorkerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, mem, deterministic=True):
        kv   = jnp.concatenate([mem, x[:, None, :]], axis=1)
        y    = nn.LayerNorm()(x)
        kvn  = nn.LayerNorm()(kv)
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dropout_rate=0.0, deterministic=deterministic)(inputs_q=y[:, None, :], inputs_kv=kvn).squeeze(1)
        x    = x + attn
        y    = nn.Dense(self.embed_dim)(nn.gelu(nn.Dense(self.embed_dim * 4)(nn.LayerNorm()(x))))
        return x + y


class TXLWorkerPolicy(nn.Module):
    """
    Worker Policy (Transformer-XL). Base Policy.
    """
    action_dim : int = ACTION_DIM
    embed_dim  : int = EMBED_DIM
    num_layers : int = WORKER_LAYERS

    def setup(self):
        self.input_proj  = nn.Dense(self.embed_dim)
        self.time_embed  = nn.Embed(MACRO_STEP + 1, self.embed_dim)
        self.blocks      = [TXLWorkerBlock(self.embed_dim, NUM_HEADS) for _ in range(self.num_layers)]
        self.action_head = nn.Dense(self.action_dim * ACTION_BINS)

    def init_carry(self, batch_size):
        return (
            jnp.zeros((batch_size, MEMORY_LEN, self.embed_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, self.embed_dim), dtype=jnp.float32)
        )

    def __call__(self, obs, carry, step_in_macro, delta, deterministic=True):
        mem, ssum = carry
        
        # NaN Sanitization
        obs   = jnp.nan_to_num(obs  , nan=0.0, posinf=0.0, neginf=0.0)
        delta = jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        
        x     = self.input_proj(obs) + self.time_embed(jnp.clip(step_in_macro, 0, MACRO_STEP)) + delta
        x     = jnp.clip(x, -10.0, 10.0)
        
        for blk in self.blocks:
            x = blk(x, mem, deterministic=deterministic)
        
        mem_new  = jnp.concatenate([mem[:, 1:, :], x[:, None, :]], axis=1)
        ssum_new = (ssum * 0.90) + (jnp.clip(x, -10.0, 10.0) * 0.10)
        
        logits = self.action_head(x).reshape(obs.shape[0], self.action_dim, ACTION_BINS)
        return logits, (mem_new, ssum_new), x


class HierarchicalAgent:
    def __init__(self):
        self.worker = TXLWorkerPolicy()
        self.meta   = EpisodicMetaController()
        self.interv = ResidualIntervention()

class Trainer:
    def __init__(self, seed=SEED):
        self.rng   = jax.random.PRNGKey(seed)
        self.agent = HierarchicalAgent()
        self.rng, k1, k2, k3, k4 = jax.random.split(self.rng, 5)
        
        dummy_obs   = jnp.zeros((1, 28), dtype=jnp.float32)
        dummy_carry = self.agent.worker.init_carry(1)
        
        # Initialize Worker
        worker_params = self.agent.worker.init(k1, dummy_obs, dummy_carry, jnp.array([0]), jnp.zeros((1, EMBED_DIM)), True)["params"]
        
        # Initialize Meta
        meta_params = self.agent.meta.init(
            k2, self.agent.meta.init_hidden(1), self.agent.meta.init_memory(1),
            dummy_obs, dummy_carry[1], jnp.zeros((1, U_DIM)), jnp.array([0]), None, True, k3
        )["params"]
        
        # Initialize Intervention
        interv_params = self.agent.interv.init(k4, jnp.zeros((1, U_DIM)), dummy_carry[1])["params"]

        # Train States
        self.worker_state = train_state.TrainState.create(
            apply_fn = self.agent.worker.apply, params=worker_params,
            tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR_BASE_SFT))
        )
        self.meta_state = train_state.TrainState.create(
            apply_fn = None, params={"meta": meta_params, "interv": interv_params},
            tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR_META_RL))
        )



# PHASE 1: SFT (SUPERVISED FINE-TUNING)

def generate_sft_dataset(env, episodes=SFT_EPISODES, print_every_eps=50):
    obs_list, act_list = [], []
    total_steps, t0 = 0, time.time()
    
    for ep in range(int(episodes)):
        obs, _ = env.reset()
        for _ in range(MAX_EPISODE_STEPS):
            grip, block, goal = obs[:3], obs[3:6], obs[-3:]
            target = block.copy(); target[2] += 0.02
            if np.linalg.norm(grip - block) <= 0.06: target = goal.copy()
            
            delta  = np.clip((target - grip) * 5.0, -1.0, 1.0)
            tokens = env.tokenizer.encode(np.append(delta, -1.0).astype(np.float32)).astype(np.int32)
            
            obs_list.append(obs.astype(np.float32))
            act_list.append(tokens)
            
            obs, _, term, trunc, _ = env.step(tokens)
            total_steps += 1
            if term or trunc: break
        
        if (ep + 1) % print_every_eps == 0:
            dt = max(1e-6, time.time() - t0)
            print(f"[SFT-GEN] ep={ep+1}/{int(episodes)} | total_steps={total_steps:,} | sps={total_steps/dt:.1f}")

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int32)

def sft_train(trainer, obs_all, act_all):
    N, idx = len(obs_all), np.arange(len(obs_all))
    
    @jax.jit
    def sft_step(state, obs_b, act_b):
        def loss_fn(p):
            B = obs_b.shape[0]; carry0 = trainer.agent.worker.init_carry(B)
            logits, _, _ = trainer.agent.worker.apply(
                {"params": p}, obs_b, carry0, jnp.zeros((B,), dtype=jnp.int32), jnp.zeros((B, EMBED_DIM)), False
            )
            return -jnp.sum(jax.nn.one_hot(act_b, ACTION_BINS) * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    for ep in range(SFT_EPOCHS):
        np.random.shuffle(idx)
        losses = []
        for i in range(0, N - SFT_BATCH_SIZE + 1, SFT_BATCH_SIZE):
            b = idx[i:i + SFT_BATCH_SIZE]
            trainer.worker_state, loss = sft_step(
                trainer.worker_state, jnp.asarray(obs_all[b], dtype=jnp.float32), jnp.asarray(act_all[b], dtype=jnp.int32)
            )
            losses.append(loss)
        print(f"[SFT] Epoch {ep} | Loss: {float(jnp.mean(jnp.array(losses))):.4f}")



# PHASE 2: RL ROLLOUT & PPO UPDATE 

@jax.jit
def policy_step_vec(worker_params, meta_params, interv_params, obs_j, carry, h, mem, u_int, step_j, rng):
    rng, k1, k2 = jax.random.split(rng, 3)
    meta, worker, interv = EpisodicMetaController(), TXLWorkerPolicy(), ResidualIntervention()
    
    h2, mem2, u_prop, sw, u_int2, aux, _ = meta.apply(
        {"params": meta_params}, h, mem, obs_j, carry[1], u_int, step_j,
        forced_action=None, force_macro_switch=True, rng=k1
    )
    delta = interv.apply({"params": interv_params}, u_int2, carry[1])
    logits, carry2, _ = worker.apply({"params": worker_params}, obs_j, carry, step_j, delta, False)
    
    logp_w = jax.nn.log_softmax(logits, axis=-1)
    B = obs_j.shape[0]
    keys = jax.random.split(k2, B * ACTION_DIM).reshape(B, ACTION_DIM, 2)
    acts = jax.vmap(lambda kb, lpb: jax.vmap(lambda k, lp: jax.random.categorical(k, lp, axis=-1))(kb, lpb))(keys, logp_w).astype(jnp.int32)
    
    lp_m = gaussian_logprob_bt(u_prop, aux["mean"], aux["logstd"]) + bernoulli_logprob(sw, aux["sw_prob"])
    return h2, mem2, u_int2, carry2, acts, u_prop, sw, lp_m, aux["sw_prob"], aux["beta"], rng

def collect_rollouts(vec_env, trainer):
    obs, infos = vec_env.reset(seed=SEED)
    obs = np.asarray(obs, dtype=np.float32)
    B = obs.shape[0]
    
    carry  = trainer.agent.worker.init_carry (B)
    h      = trainer.agent.meta  .init_hidden(B)
    mem    = trainer.agent.meta  .init_memory(B)
    u_int  = jnp.zeros((B, U_DIM), dtype=jnp.float32)
    step_m = np.zeros((B,), dtype=np.int32)
    
    buffers      = {k: [[] for _ in range(B)] for k in ["obs", "ssum", "step", "uprop", "sw", "decm", "act", "oldlp"]}
    done_mask    = np.zeros((B,), dtype=bool)
    final_scores = np.zeros((B,), dtype=np.float32)

    steps = 0
    while (not np.all(done_mask)) and (steps < MAX_EPISODE_STEPS):
        h, mem, u_int, carry, acts, u_prop, sw, lp_m, _, _, trainer.rng = policy_step_vec(
            trainer.worker_state.params, trainer.meta_state.params["meta"], trainer.meta_state.params["interv"],
            jnp.asarray(obs), carry, h, mem, u_int, jnp.asarray(step_m), trainer.rng
        )
        
        acts_np, sw_np, lp_np = np.asarray(acts), np.asarray(sw), np.asarray(lp_m)
        decm_np = np.logical_or((step_m == 0), (sw_np > 0.5)).astype(np.float32)
        
        for i in range(B):
            if not done_mask[i]:
                for k, v in zip(buffers.keys(), [obs[i], carry[1][i], step_m[i], u_prop[i], sw_np[i], decm_np[i], acts_np[i], lp_np[i]]):
                    buffers[k][i].append(np.asarray(v))
        
        obs, _, term, trunc, infos = vec_env.step(acts_np)
        obs = np.asarray(obs, dtype=np.float32)
        step_done = np.asarray(term) | np.asarray(trunc)
        
        if np.any(step_done):
            finfos = infos.get("final_info", [None]*B)
            just_finished = step_done & (~done_mask)
            for i in np.where(just_finished)[0]:
                fi = finfos[i] if finfos[i] else {k: v[i] for k, v in infos.items() if k not in ["final_info", "final_observation"]}
                if fi: final_scores[i] = (float(fi.get("seq_goal_index", 0)) * 10.0) + (50.0 if fi.get("is_success", 0) > 0.5 else 0.0)
            
            dm = jnp.asarray(just_finished)
            h = jnp.where(dm[:, None], jnp.zeros_like(h), h)
            u_int = jnp.where(dm[:, None], jnp.zeros_like(u_int), u_int)
            mem = (
                jnp.where(dm[:, None, None], jnp.zeros_like(mem[0])      , mem[0]),
                jnp.where(dm[:, None, None], jnp.zeros_like(mem[1])      , mem[1]),
                jnp.where(dm[:, None]      , jnp.ones_like (mem[2])*100.0, mem[2]),
                jnp.where(dm[:, None]      , jnp.zeros_like(mem[3])      , mem[3])
            )
            carry = (jnp.where(dm[:, None, None], jnp.zeros_like(carry[0]), carry[0]), jnp.where(dm[:, None], jnp.zeros_like(carry[1]), carry[1]))
            step_m[just_finished] = 0
            
        done_mask |= step_done
        steps     += 1
        step_m    = (step_m + 1) % MACRO_STEP

    if np.any(~done_mask):
         for i in np.where(~done_mask)[0]:
             fi = {k: v[i] for k, v in infos.items() if k not in ["final_info", "final_observation"]}
             if fi: final_scores[i] = (float(fi.get("seq_goal_index", 0)) * 10.0) + (50.0 if fi.get("is_success", 0) > 0.5 else 0.0)

    trajs = [{k: np.array(v[i]) for k, v in buffers.items()} for i in range(B)]
    return trajs, final_scores

def compute_grpo_advantages(scores):
    mean = float(np.mean(scores))
    std  = float(np.std(scores)) + 1e-8
    return np.clip((scores - mean) / std, -5.0, 5.0).astype(np.float32), mean

def pad_batch(trajs):
    B, max_t = len(trajs), max([t["obs"].shape[0] for t in trajs])
    batch = {k: np.zeros((B, max_t) + v.shape[1:], dtype=v.dtype) for k, v in trajs[0].items()}
    batch["mask"] = np.zeros((B, max_t), dtype=np.float32)
    for i, tr in enumerate(trajs):
        T = tr["obs"].shape[0]
        if T > 0:
            for k, v in tr.items(): batch[k][i, :T] = v
            batch["mask"][i, :T] = 1.0
    return [jnp.asarray(batch[k]) for k in ["obs", "ssum", "step", "uprop", "sw", "decm", "act", "oldlp", "mask"]]

@jax.jit
def meta_replay_scan(params_meta, obs_bt, ssum_bt, step_bt, uprop_bt, sw_bt, mask_bt):
    meta = EpisodicMetaController(); B = obs_bt.shape[0]
    h0, mem0, u0 = meta.init_hidden(B), meta.init_memory(B), jnp.zeros((B, U_DIM))
    
    in_T = [jnp.swapaxes(x, 0, 1) for x in [obs_bt, ssum_bt, step_bt, uprop_bt, sw_bt, mask_bt]]
    
    def scan_fn(carry, x):
        h_prev, mem_prev, u_prev = carry
        obs_t, ssum_t, step_t, uprop_t, sw_t, m_t = x
        
        # Masking, Padding-тай хэсгийн оролтыг 0 болгож NaN-аас сэргийлэх
        m_h       = (m_t > 0)[:, None]
        m_kv      = m_h[:, None]
        obs_safe  = jnp.where(m_h, obs_t, 0.0)
        ssum_safe = jnp.where(m_h, ssum_t, 0.0)
        
        h_next, mem_next, _, _, u_int_next, aux, _ = meta.apply(
            {"params": params_meta}, h_prev, mem_prev, obs_safe, ssum_safe, u_prev, step_t,
            forced_action=(uprop_t, sw_t), force_macro_switch=True, rng=None
        )
        
        h_final = jnp.where(m_h, h_next, h_prev)
        u_final = jnp.where(m_h, u_int_next, u_prev)
        mem_final = (
            jnp.where(m_kv, mem_next[0], mem_prev[0]), jnp.where(m_kv, mem_next[1], mem_prev[1]),
            jnp.where(m_h, mem_next[2], mem_prev[2]), jnp.where(m_h, mem_next[3], mem_prev[3])
        )
        
        return (h_final, mem_final, u_final), (aux["mean"], aux["sw_prob"], aux["beta"], u_final)

    _, outs = jax.lax.scan(scan_fn, (h0, mem0, u0), tuple(in_T))
    return [jnp.swapaxes(x, 0, 1) for x in outs]

@jax.jit
def worker_logprob_rescan(worker_params, interv_params, obs_bt, step_bt, u_int_bt, act_bt, mask_bt):
    worker, interv = TXLWorkerPolicy(), ResidualIntervention(); B = obs_bt.shape[0]
    carry0 = worker.init_carry(B)
    in_T = [jnp.swapaxes(x, 0, 1) for x in [obs_bt, step_bt, u_int_bt, act_bt, mask_bt]]

    def scan_fn(carry, x):
        obs_t, step_t, u_t, a_t, m_t = x
        mem_old, ssum_old = carry
        m_mem, m_vec = (m_t > 0)[:, None, None], (m_t > 0)[:, None]
        
        u_t_safe = jnp.where(m_vec, u_t, 0.0)
        delta = interv.apply({"params": interv_params}, u_t_safe, ssum_old)
        logits, (mem_new, ssum_new), _ = worker.apply({"params": worker_params}, obs_t, carry, step_t, delta, False)
        
        logp = jax.nn.log_softmax(logits, axis=-1)
        lpw = jnp.sum(jnp.take_along_axis(logp, a_t[..., None], axis=-1).squeeze(-1), axis=-1)
        
        return (jnp.where(m_mem, mem_new, mem_old), jnp.where(m_vec, ssum_new, ssum_old)), jnp.where(m_t > 0, lpw, 0.0)

    _, lpw_T = jax.lax.scan(scan_fn, carry0, tuple(in_T))
    return jnp.swapaxes(lpw_T, 0, 1)

@jax.jit
def ppo_update_step(meta_state, ref_params, worker_params, batch, adv_ep, kl_beta):
    obs, ssum, step, uprop, sw, decm, act, oldlp, mask = batch

    def loss_fn(p):
        mean_bt, swprob_bt, beta_bt, u_int_bt = meta_replay_scan(p["meta"], obs, ssum, step, uprop, sw, mask)
        mean_ref, swprob_ref, _, _ = meta_replay_scan(ref_params["meta"], obs, ssum, step, uprop, sw, mask)
        lpw_bt = worker_logprob_rescan(worker_params, p["interv"], obs, step, u_int_bt, act, mask)

        logstd     = jnp.clip(p["meta"]["u_logstd"], -5.0, 2.0)
        logstd_ref = jnp.clip(ref_params["meta"]["u_logstd"], -5.0, 2.0)
        
        lp_u  = gaussian_logprob_bt(uprop, mean_bt, logstd)
        lp_sw = bernoulli_logprob(sw, swprob_bt)
        
        # Masking, Зөвхөн valid data дээр log_prob тооцох
        lp_new = jnp.where(mask > 0, lp_u + lp_sw, 0.0)
        oldlp_safe = jnp.where(mask > 0, oldlp, 0.0)
        
        ratio     = jnp.exp(jnp.clip(lp_new - oldlp_safe, -10.0, 10.0))
        surr      = jnp.minimum(ratio * adv_ep[:, None], jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_ep[:, None])
        m_dec     = mask * decm
        surr_loss = jnp.sum(jnp.where(m_dec > 0, surr, 0.0)) / (jnp.sum(m_dec) + 1e-8)

        kl      = jnp.where(m_dec > 0, gaussian_kl_bt(mean_bt, logstd, mean_ref, logstd_ref) + bernoulli_kl(swprob_bt, swprob_ref), 0.0)
        kl_mean = jnp.sum(kl) / (jnp.sum(m_dec) + 1e-8)
        
        sw_pen   = ((jnp.sum(swprob_bt * mask) / (jnp.sum(mask) + 1e-8)) - TARGET_SW_RATE) ** 2
        beta_m   = jnp.sum(beta_bt * mask) / (jnp.sum(mask) + 1e-8)
        beta_pen = (BETA_MEAN_COEFF * (beta_m - BETA_MEAN_TARGET)**2) + (BETA_VAR_COEFF * (jnp.sum(((beta_bt - beta_m)**2) * mask) / (jnp.sum(mask) + 1e-8)))
        ent      = (U_ENTROPY_COEFF * gaussian_entropy(logstd)) + (SWITCH_ENTROPY_COEFF * (jnp.sum(bernoulli_entropy(swprob_bt) * mask) / (jnp.sum(mask) + 1e-8)))
        
        dec_pg = jnp.sum(jnp.where(m_dec > 0, lpw_bt * adv_ep[:, None], 0.0)) / (jnp.sum(m_dec) + 1e-8)
        
        loss = -surr_loss + (kl_beta * kl_mean) + (SW_RATE_COEFF * sw_pen) + beta_pen - ent - (DECODER_PG_COEFF * dec_pg)
        return loss, (kl_mean, sw_pen, beta_m, dec_pg)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(meta_state.params)
    
    # Gradient Sanitization (JAX 0.6+)
    grads = jtu.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    grads = jtu.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    return meta_state.apply_gradients(grads=grads), loss, aux

def ppo_epoch(trainer, ref_params, trajs, advs, kl_beta):
    idx = np.arange(len(trajs)); np.random.shuffle(idx); last_stats = None
    for i in range(0, len(trajs), MINI_BATCH_SIZE):
        mb_idx = idx[i:i + MINI_BATCH_SIZE]
        batch = pad_batch([trajs[j] for j in mb_idx])
        trainer.meta_state, loss, aux = ppo_update_step(
            trainer.meta_state, ref_params, trainer.worker_state.params, batch, jnp.asarray(advs[mb_idx], dtype=jnp.float32), kl_beta
        )
        last_stats = (loss, aux)
    return last_stats


# INFERENCE & RENDERING

@jax.jit
def inference_step_jit(worker_params, meta_params, interv_params, obs, carry, h, mem, u_int, step_in_macro):
    """
    Inference буюу Human Play дээр ашиглах JIT функц.
    """
    worker, meta, interv = TXLWorkerPolicy(), EpisodicMetaController(), ResidualIntervention()
    obs_j, step_j        = obs[None, :], jnp.array([step_in_macro], dtype=jnp.int32)
    
    h2, mem2, _, sw, u_int2, _, _ = meta.apply(
        {"params": meta_params}, h, mem, obs_j, carry[1], u_int, step_j,
        forced_action=None, force_macro_switch=True, rng=None
    )
    
    delta = interv.apply({"params": interv_params}, u_int2, carry[1])
    logits, carry2, _ = worker.apply({"params": worker_params}, obs_j, carry, step_j, delta, True)
    
    act_tokens = jnp.argmax(logits[0], axis=-1).astype(jnp.int32)
    next_step  = jnp.where(sw[0] > 0, 0, (step_in_macro + 1) % MACRO_STEP).astype(jnp.int32)
    
    return act_tokens, carry2, h2, mem2, u_int2, next_step


def play_policy_human(trainer, episodes=2):
    """
    Машин сургалтын үр дүнг тоглуулж харах функц.
    """
    try:
        env = make_single_env(999, render_mode="human")()
    except Exception as e:
        print(f"[PLAY] Error creating env: {e}"); return

    print(f"\n[PLAY] Rendering {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + 1000 + ep)
        
        # State Init
        carry    = trainer.agent.worker.init_carry (1)
        h        = trainer.agent.meta  .init_hidden(1)
        mem      = trainer.agent.meta  .init_memory(1)
        u_int    = jnp.zeros((1, U_DIM))
        step_val = jnp.array(0, dtype=jnp.int32)
        
        done, score = False, 0.0
        
        while not done:
            act_tok, carry, h, mem, u_int, step_val = inference_step_jit(
                trainer.worker_state.params,
                trainer.meta_state.params["meta"  ],
                trainer.meta_state.params["interv"],
                jnp.array(obs), carry, h, mem, u_int, step_val
            )
            
            obs, _, term, trunc, info = env.step(np.array(act_tok))
            
            if PLAY_SLEEP_SEC > 0:
                time.sleep(PLAY_SLEEP_SEC)
            
            if term or trunc:
                done = True
                score = (float(info.get("seq_goal_index", 0)) * 10.0) + (50.0 if info.get("is_success", 0) > 0.5 else 0.0)
        
        print(f"[PLAY] Ep {ep+1} Score: {score:.1f}")
    
    env.close()



# MAIN

def main():
    np.random.seed(SEED); random.seed(SEED)
    trainer = Trainer(seed=SEED)

    # Phase 1: SFT (Warm Start)
    if SFT_ENABLE and (not os.path.exists(SFT_FLAG)):
        print("[SFT] Generating Heuristic Data...")
        sft_env          = make_single_env(0)()
        obs_all, act_all = generate_sft_dataset(sft_env, episodes=SFT_EPISODES)
        sft_env.close()
        
        print(f"[SFT] Training on {len(obs_all)} samples...")
        sft_train(trainer, obs_all, act_all)
        
        #with open(SFT_FLAG, "w") as f: f.write("done")

    # Phase 2: Internal RL
    vec_env = create_vec_env(GROUP_SIZE)
    kl_beta = float(KL_BETA)

    print("\n" + "="*60)
    print("  INTERNAL RL - BULLETPROOF STABLE VERSION")
    print(f"  Updates: {UPDATES} | Batch: {GROUP_SIZE} | Memory Slots: {NUM_SLOTS}")
    print("="*60 + "\n")

    for upd in range(1, UPDATES + 1):
        # Collect Data (Rollout)
        ref_params = trainer.meta_state.params
        trajs, scores = collect_rollouts(vec_env, trainer)
        
        # Compute Advantage (Normalized)
        advs, mean_score = compute_grpo_advantages(scores)

        # PPO Update
        for _ in range(PPO_EPOCHS):
            last_stats = ppo_epoch(trainer, ref_params, trajs, advs, kl_beta)
        
        loss, (kl_v, sw_pen, beta_m, dec_pg) = last_stats

        # KL Adjustment
        if kl_v > TARGET_KL * 1.5: kl_beta *= KL_ALPHA
        elif kl_v < TARGET_KL / 1.5: kl_beta /= KL_ALPHA

        # Logging
        if upd % 5 == 0 or upd == 1:
            best = float(np.max(scores)) if len(scores) else 0.0
            print(f"[UPD {upd:4d}] Score: {mean_score:6.2f} | Best: {best:6.2f} | Loss: {loss:6.3f} | KL: {kl_v:.4f} | Beta: {beta_m:.2f} | DecPG: {dec_pg:.3f}")

        # Rendering
        if PLAY_ENABLE and (upd % PLAY_EVERY == 0):
            play_policy_human(trainer, episodes=PLAY_EPISODES)

    vec_env.close()
    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()