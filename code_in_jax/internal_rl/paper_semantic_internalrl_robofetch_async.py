#
# INTERNAL RL - ROBOTIC FETCH, META-CONTROLLER + RESIDUAL INTERVENTION (ASYNC VECTORIZED)
#
#
# ЛАВЛАГАА:
#   - Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning
#     https://arxiv.org/abs/2512.20605
#
#
# ЗОРИЛГО:
#   FetchPush орчинг Autoregressive Token Generation хэлбэрт оруулж, Internal RL буюу
#   дотоод хийсвэрлэл үүсгэх аргыг ашиглан шатлалтай (hierarchical) удирдлага хийх.
#   Энэ хувилбар нь AsyncVectorEnv ашиглан олон орчинг зэрэг ажиллуулж сургалтыг хурдасгана.
#
#
# АРХИТЕКТУРЫН БҮРЭЛДЭХҮҮН ХЭСГҮҮД:
#
#   1. BaseARPolicy (Worker - Autoregressive):
#      - Үндсэн үйлдлийг (action tokens) гүйцэтгэгч.
#      - RL шатанд энэ сүлжээний жингүүд царсан байна (FROZEN) байна.
#      - Гаднаас орж ирэх delta (Δ) вектороор дамжуулан "бодлоо" өөрчилнө.
#
#   2. MetaController (Manager):
#      - Env болон Worker-ийн дотоод төлөвийг ажиглаж, удирдах үүрэгтэй.
#      - Proposal Head : Шинэ команд буюу intent (u_prop) санал болгоно.
#      - Switch Head   : Шинэ команд өгөх эсэх (switch)-ийг шийднэ.
#      - Beta Config   : Хугацааны нэгтгэлийн коэффициент (β)-ийг тохируулна.
#      - Integration   : Өмнөх u_int болон шинэ u_prop-ийг β-аар жигнэн нэгтгэнэ.
#
#   3. ResidualIntervention (Decoder):
#      - Meta-гаас гарсан u_int кодыг Low-Rank матриц (A, B) ашиглан
#        Worker-ийн ойлгох хэл буюу delta (Δ) вектор руу хөрвүүлнэ.
#      - Энэ delta нь Worker-ийн анхаарлын (attention) механизмд шууд нөлөөлнө.
#
#
# СУРГАЛТЫН ҮЕ ШАТУУД:
#
#   PHASE 1: SFT (Supervised Fine-Tuning)
#     - Worker моделийг Heuristic өгөгдөл дээр сургаж, орчны тухай
#       суурь ойлголт, хөдөлгөөний эвсэл суулгана.
#
#   PHASE 2: Internal RL (PPO + GRPO) - Parallel
#     - 16 орчинг зэрэг ажиллуулж (AsyncVectorEnv) өгөгдөл цуглуулна.
#     - Worker-ийг царцааж, зөвхөн MetaController болон Intervention хэсгийг сургана.
#


import os
import math
import time
import random
import numpy              as np

import gymnasium          as gym
import gymnasium_robotics
from   gymnasium.vector   import AsyncVectorEnv

import jax
import jax.numpy          as jnp
import flax.linen         as nn
import optax

from flax.training        import train_state


# ЕРӨНХИЙ ТОХИРГОО

SEED                 = 42

ENV_ID               = "FetchPush-v4"
MAX_EPISODE_STEPS    = 200

ACTION_DIM           = 4
ACTION_BINS          = 256

K_SEQUENTIAL_GOALS   = 3
GOAL_DIST_THRESHOLD  = 0.05
SPARSE_FINAL_REWARD  = True

MACRO_STEP           = 10

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

EMBED_DIM            = 256
NUM_HEADS            = 4
MEMORY_LEN           = 16

U_DIM                = 128
GRU_H_DIM            = 256
LRANK                = 32

SWITCH_TAU           = 0.50

BETA_MIN             = 0.05
BETA_MAX             = 0.995

SWITCH_ENTROPY_COEFF = 0.01
U_ENTROPY_COEFF      = 0.001

TARGET_SW_RATE       = 0.10
SW_RATE_COEFF        = 1.0

BETA_MEAN_TARGET     = 0.85
BETA_MEAN_COEFF      = 0.2
BETA_VAR_COEFF       = 0.1

DECODER_PG_COEFF     = 0.10

SFT_ENABLE           = True
SFT_FLAG             = "sft_done_fetch_internal_rl_async.flag"
SFT_EPISODES         = 200
SFT_EPOCHS           = 3
SFT_BATCH_SIZE       = 256

# PERIODIC PLAY
PLAY_ENABLE          = True
PLAY_EVERY           = 3      # хэдэн update тутамд үзүүлэх вэ
PLAY_EPISODES        = 2      # хэдэн episode үзүүлэх вэ
PLAY_SLEEP_SEC       = 0.0    

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# ENV WRAPPERS

class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Observation Dict-ийг хавтгай вектор (flat vector) болгон хувиргана.
      - observation   : (25,) биеийн байрлал + объект
      - desired_goal  : (3,)  хүрэх ёстой цэг
    Гаралт: concat(observation, desired_goal) -> shape (28,)
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
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
    - Reset хийхэд    : K ширхэг зорилго үүснэ.
    - Episode дундуур : Агент эхний зорилгод хүрвэл дараагийнх нь идэвхжинэ.
    - Шагнал          : Зөвхөн эцсийн зорилгод хүрэхэд шагнал өгнө (Sparse Reward).
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

        for _ in range(self.k - 1):
            offset = np.random.uniform(-0.10, 0.10, size=3).astype(np.float32)
            g      = base_goal + offset
            g[2]   = max(0.42, float(g[2]))
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
            if self._goal_idx < self.k - 1:
                self._goal_idx         += 1
                info["seq_goal_index"]  = self._goal_idx
                info["is_success"    ]  = 0.0

                terminated = False
                reward     = 0.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info
            else:
                info["is_success"] = 1.0
                terminated         = True
                reward             = 1.0 if self.sparse_final_reward else float(base_reward)
                return self._update_obs(obs), reward, terminated, truncated, info

        info["is_success"] = 0.0
        reward             = 0.0 if self.sparse_final_reward else float(base_reward)
        return self._update_obs(obs), reward, terminated, truncated, info


# ASYNC VECTOR ENV SETUP

def make_single_env(rank, render_mode=None):
    """
    AsyncVectorEnv-д ашиглагдах үйлдвэрлэгч функц.
    Process бүр өөр seed-тэй байхын тулд rank ашиглана.
    """
    def _thunk():
        env = gym.make(
            ENV_ID,
            render_mode       = render_mode,
            max_episode_steps = MAX_EPISODE_STEPS,
        )
        env = SequentialGoalsWrapper(
            env,
            k                   = K_SEQUENTIAL_GOALS,
            dist_threshold      = GOAL_DIST_THRESHOLD,
            sparse_final_reward = SPARSE_FINAL_REWARD,
        )
        env = TokenActionWrapper(env, bins=ACTION_BINS)
        env = TransformerObservationWrapper(env)
        env.reset(seed=SEED + rank)
        return env
    return _thunk


def create_vec_env(n_envs, render_mode=None):
    """
    AsyncVectorEnv ашиглан олон орчинг параллель үүсгэх.
    """
    fns = [make_single_env(i, render_mode=render_mode) for i in range(n_envs)]
    return AsyncVectorEnv(fns)


# WORKER, BaseARPolicy (Recurrent Transformer)

class CWRRTCell(nn.Module):
    """
    Worker моделийн санах ой болон attention блок.
    Residual Intervention:
      - Meta-гаас ирсэн delta (Δ) векторыг оролт дээр нэмэх замаар моделийн үйлдэлд нөлөөлнө.
      - x_in = x + delta
    Carry States:
      - mem  : Сүүлийн хэдэн алхмын KV cache (context).
      - ssum : Урт хугацааны хураангуй (intent).
    """
    embed_dim : int
    num_heads : int
    mem_len   : int = MEMORY_LEN

    @nn.compact
    def __call__(self, carry, x, delta):
        mem, ssum = carry

        x_in = x + delta

        new_mem_entry = x_in[:, None, :]
        updated_mem   = jnp.concatenate([mem[:, 1:, :], new_mem_entry], axis=1)

        y = nn.LayerNorm()(x_in)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q  = y[:, None, :],
            inputs_kv = updated_mem
        )
        y = y.squeeze(1)
        x_mid = x_in + y

        y     = nn.LayerNorm()(x_mid)
        y     = nn.Dense(self.embed_dim * 4)(y)
        y     = nn.gelu(y)
        y     = nn.Dense(self.embed_dim)(y)
        x_out = x_mid + y

        lam      = 0.90
        new_ssum = (ssum * lam) + (x_out * (1.0 - lam))

        return (updated_mem, new_ssum), x_out


class BaseARPolicy(nn.Module):
    """
    BaseARPolicy (Worker).
    Autoregressive байдлаар токен үүсгэнэ. RL шатанд энэ модель сурахгүй (frozen), гэхдээ delta-гаар дамжин удирдагдана.
    """
    action_dim : int = ACTION_DIM
    action_bins: int = ACTION_BINS
    embed_dim  : int = EMBED_DIM
    mem_len    : int = MEMORY_LEN

    def setup(self):
        self.input_proj  = nn.Dense(self.embed_dim)
        self.time_embed  = nn.Embed(MACRO_STEP + 1, self.embed_dim)
        self.cell        = CWRRTCell(embed_dim=self.embed_dim, num_heads=NUM_HEADS, mem_len=self.mem_len)
        self.action_head = nn.Dense(self.action_dim * self.action_bins)

    def init_carry(self, batch_size):
        return (
            jnp.zeros((batch_size, self.mem_len, self.embed_dim)),
            jnp.zeros((batch_size, self.embed_dim)),
        )

    def __call__(self, obs, carry, step_in_macro, delta):
        x     = self.input_proj(obs)
        t_emb = self.time_embed(jnp.clip(step_in_macro, 0, MACRO_STEP))
        x     = x + t_emb

        next_carry, x_out = self.cell(carry, x, delta)

        logits = self.action_head(x_out)
        logits = logits.reshape(obs.shape[0], self.action_dim, self.action_bins)

        base_feat = x_out
        return logits, next_carry, base_feat


# META, Controller Components

class ControllerEncoderGRU(nn.Module):
    """
    Worker-ийн ssum болон орчны мэдээллийг базаж, хугацааны дарааллыг ойлгох GRU блок.
    """
    h_dim: int = GRU_H_DIM

    @nn.compact
    def __call__(self, h_prev, x_t):
        gru    = nn.GRUCell(features=self.h_dim)
        h_t, _ = gru(h_prev, x_t)
        return h_t

    def init_hidden(self, batch_size):
        return jnp.zeros((batch_size, self.h_dim), dtype=jnp.float32)


class ProposalHead(nn.Module):
    """
    Worker-д өгөх шинэ команд (intent) санал болгох хэсэг. Gaussian тархалтаас дээж авна.
    """
    u_dim: int = U_DIM

    @nn.compact
    def __call__(self, h):
        mean   = nn.Dense(self.u_dim)(h)
        logstd = self.param("u_logstd", nn.initializers.constant(-1.0), (self.u_dim,))
        return mean, logstd


class BetaConfigurator(nn.Module):
    """
    Хугацааны нэгтгэлийн коэффициент (β)-ийг тооцоолох хэсэг.
    """
    @nn.compact
    def __call__(self, h):
        logit = nn.Dense(1)(h).squeeze(-1)
        beta  = nn.sigmoid(logit)
        beta  = BETA_MIN + (BETA_MAX - BETA_MIN) * beta
        return beta, logit


class SwitchHead(nn.Module):
    """
    Шинэ команд өгөх эсэх (Termination) шийдвэрийг гаргах хэсэг. Bernoulli тархалт ашиглана.
    """
    @nn.compact
    def __call__(self, h):
        logit = nn.Dense(1)(h).squeeze(-1)
        prob  = nn.sigmoid(logit)
        return prob, logit


class TemporalIntegrationUnit(nn.Module):
    """
    u_int_t = beta_eff * u_int_{t-1} + (1-beta_eff) * u_prop_t
    """
    @nn.compact
    def __call__(self, u_int_prev, u_prop, beta, switch):
        sw       = switch[:, None].astype(jnp.float32)
        beta_eff = beta[:, None] * (1.0 - sw)
        u_int    = (beta_eff * u_int_prev) + ((1.0 - beta_eff) * u_prop)
        return u_int, beta_eff.squeeze(-1)


# DECODER & INTERVENTION

class ControllerDecoderHyperNet(nn.Module):
    """
    Low-Rank Adaptation (LoRA) матрицуудыг u_int кодноос үүсгэнэ.
    """
    embed_dim: int = EMBED_DIM
    rank     : int = LRANK
    u_dim    : int = U_DIM

    def setup(self):
        self.mlp1 = nn.Dense(self.embed_dim)
        self.mlp2 = nn.Dense(self.embed_dim)

        self.to_A = nn.Dense(self.embed_dim * self.rank)
        self.to_B = nn.Dense(self.rank * self.embed_dim)
        self.to_b = nn.Dense(self.embed_dim)

    def __call__(self, u_int):
        x = nn.tanh(self.mlp1(u_int))
        x = nn.tanh(self.mlp2(x))

        A = self.to_A(x).reshape(u_int.shape[0], self.embed_dim, self.rank)
        B = self.to_B(x).reshape(u_int.shape[0], self.rank, self.embed_dim)
        b = self.to_b(x)

        A = jnp.tanh(A) * 0.5
        B = jnp.tanh(B) * 0.5
        b = jnp.tanh(b) * 0.5
        return A, B, b


class ResidualIntervention(nn.Module):
    """
    Delta = (ssum @ A) @ B + b
    """
    embed_dim: int = EMBED_DIM
    rank     : int = LRANK

    def setup(self):
        self.decoder = ControllerDecoderHyperNet(embed_dim=self.embed_dim, rank=self.rank, u_dim=U_DIM)

    def __call__(self, u_int, ssum):
        A, B, b = self.decoder(u_int)

        y     = jnp.matmul(ssum[:, None, :], A).squeeze(1)
        delta = jnp.matmul(y[:, None, :], B).squeeze(1) + b
        return delta


# META CONTROLLER (MANAGER)

class MetaController(nn.Module):
    """
    Оролт  : obs, ssum, u_int_prev
    Гаралт : h_t, u_prop, switch, u_int
    """
    u_dim : int = U_DIM
    h_dim : int = GRU_H_DIM

    def setup(self):
        self.obs_proj   = nn.Dense(128)
        self.ssum_proj  = nn.Dense(128)

        self.encoder    = ControllerEncoderGRU(h_dim=self.h_dim)
        self.proposal   = ProposalHead(u_dim=self.u_dim)
        self.beta_cfg   = BetaConfigurator()
        self.switch_hd  = SwitchHead()
        self.integrator = TemporalIntegrationUnit()

    def init_hidden(self, batch_size):
        return jnp.zeros((batch_size, self.h_dim), dtype=jnp.float32)

    def __call__(self, h_prev, obs, ssum, u_int_prev, step_in_macro, force_macro_switch=False, rng=None):
        xo   = nn.tanh(self.obs_proj(obs))
        xs   = nn.tanh(self.ssum_proj(ssum))
        x_in = jnp.concatenate([xo, xs], axis=-1)

        h_t  = self.encoder(h_prev, x_in)

        mean, logstd      = self.proposal(h_t)
        beta, beta_logit  = self.beta_cfg(h_t)
        sw_prob, sw_logit = self.switch_hd(h_t)

        if rng is None:
            u_prop = mean
            switch = (sw_prob > SWITCH_TAU).astype(jnp.int32)
        else:
            rng, k1, k2 = jax.random.split(rng, 3)
            eps         = jax.random.normal(k1, shape=mean.shape)
            u_prop      = mean + jnp.exp(logstd)[None, :] * eps
            sw          = jax.random.bernoulli(k2, sw_prob)
            switch      = sw.astype(jnp.int32)

        if force_macro_switch:
            is_boundary = (step_in_macro == 0).astype(jnp.int32)
            switch      = jnp.maximum(switch, is_boundary)

        u_int, beta_eff = self.integrator(u_int_prev, u_prop, beta, switch)

        aux = {
            "mean"      : mean,
            "logstd"    : logstd,
            "beta"      : beta,
            "beta_logit": beta_logit,
            "sw_prob"   : sw_prob,
            "sw_logit"  : sw_logit,
            "beta_eff"  : beta_eff,
        }

        return h_t, u_prop, switch, u_int, aux, rng


# ТУСЛАХ ФУНКЦҮҮД

@jax.jit
def gaussian_logprob(x, mean, logstd):
    var = jnp.exp(2.0 * logstd)
    return -0.5 * jnp.sum(((x - mean) ** 2) / (var + 1e-8) + 2.0 * logstd + jnp.log(2.0 * jnp.pi), axis=-1)

@jax.jit
def gaussian_entropy(logstd):
    return jnp.sum(logstd + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))

@jax.jit
def gaussian_kl(mean_new, logstd_new, mean_ref, logstd_ref):
    var_new = jnp.exp(2.0 * logstd_new)
    var_ref = jnp.exp(2.0 * logstd_ref)
    term1   = (logstd_ref - logstd_new)
    term2   = (var_new + (mean_new - mean_ref) ** 2) / (2.0 * var_ref + 1e-8)
    kl      = jnp.sum(term1 + term2 - 0.5, axis=-1)
    return kl

@jax.jit
def bernoulli_logprob(action01, prob):
    a = action01.astype(jnp.float32)
    p = jnp.clip(prob, 1e-6, 1.0 - 1e-6)
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


# HIERARCHICAL AGENT

class HierarchicalAgent:
    """
    Бүх моделийг багцалсан класс.
        Base + Meta + Intervention.
    """
    def __init__(self):
        self.base_model   = BaseARPolicy()
        self.meta_model   = MetaController()
        self.intervention = ResidualIntervention()


# TRAINING STATE

class Trainer:
    """
    Сургалтын төлөвийг хадгалах класс.
        base_state : Worker моделийн параметрүүд (Сургалтын үед өөрчлөгдөхгүй).
        meta_state : Meta болон Intervention параметрүүд (Сургагдана).
    """
    def __init__(self, seed=SEED):
        self.rng   = jax.random.PRNGKey(seed)
        self.agent = HierarchicalAgent()

        self.rng, k1, k2, k3, k4 = jax.random.split(self.rng, 5)

        dummy_obs   = jnp.zeros((1, 28), dtype=jnp.float32)
        dummy_carry = self.agent.base_model.init_carry(1)
        dummy_step  = jnp.array([0], dtype=jnp.int32)
        dummy_delta = jnp.zeros((1, EMBED_DIM), dtype=jnp.float32)

        base_params = self.agent.base_model.init(k1, dummy_obs, dummy_carry, dummy_step, dummy_delta)["params"]

        dummy_h     = jnp.zeros((1, GRU_H_DIM), dtype=jnp.float32)
        dummy_u_int = jnp.zeros((1, U_DIM), dtype=jnp.float32)

        meta_params = self.agent.meta_model.init(
            k2,
            dummy_h,
            dummy_obs,
            dummy_carry[1],
            dummy_u_int,
            dummy_step,
            force_macro_switch=True,
            rng=k3
        )["params"]

        interv_params = self.agent.intervention.init(k4, dummy_u_int, dummy_carry[1])["params"]

        self.base_state = train_state.TrainState.create(
            apply_fn = self.agent.base_model.apply,
            params   = base_params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR_BASE_SFT),
            ),
        )

        meta_bundle = {"meta": meta_params, "interv": interv_params}

        self.meta_state = train_state.TrainState.create(
            apply_fn = None,
            params   = meta_bundle,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR_META_RL),
            ),
        )


# PHASE 1, SFT (Warm Start)

def generate_sft_dataset(env, episodes=SFT_EPISODES):
    """
    Heuristic буюу дүрэмд суурилсан аргаар сургалтын өгөгдөл цуглуулах.
    Worker-д анхан шатны мэдлэг олгоход хэрэглэнэ.
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
            if term or trunc:
                break

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int32)


def sft_train(trainer, obs_all, act_all):
    """
    Worker-ийг SFT горимоор сургах.
    Meta болон Intervention оролцохгүй, delta=0 гэж үзнэ.
    """
    N   = len(obs_all)
    idx = np.arange(N)

    @jax.jit
    def sft_step(base_state, obs_b, act_b):
        def loss_fn(p_base):
            B      = obs_b.shape[0]
            carry0 = trainer.agent.base_model.init_carry(B)
            step0  = jnp.zeros((B,), dtype=jnp.int32)
            delta0 = jnp.zeros((B, EMBED_DIM), dtype=jnp.float32)

            logits, _, _ = trainer.agent.base_model.apply(
                {"params": p_base},
                obs_b, carry0, step0, delta0
            )

            logp    = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(act_b, ACTION_BINS)
            loss    = -jnp.sum(one_hot * logp, axis=-1).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(base_state.params)
        base_state  = base_state.apply_gradients(grads=grads)
        return base_state, loss

    for ep in range(SFT_EPOCHS):
        np.random.shuffle(idx)
        losses = []

        for i in range(0, N - SFT_BATCH_SIZE + 1, SFT_BATCH_SIZE):
            b     = idx[i:i + SFT_BATCH_SIZE]
            obs_b = jnp.asarray(obs_all[b], dtype=jnp.float32)
            act_b = jnp.asarray(act_all[b], dtype=jnp.int32  )

            trainer.base_state, loss = sft_step(trainer.base_state, obs_b, act_b)
            losses.append(loss)

        print(f"[SFT] Epoch {ep} | Loss: {float(jnp.mean(jnp.array(losses))):.6f}")


# PHASE 2, INTERNAL RL (ASYNC ROLLOUTS)

@jax.jit
def policy_step_vec(base_params, meta_params, interv_params, obs_j, carry, h, u_int, step_j, rng):
    """
    Нэг алхам дээрх бүх моделийн урсгалыг (Meta->Interv->Worker) гүйцэтгэх JIT функц.
    Vectorized: B env нэг дор.
    """
    base   = BaseARPolicy()
    meta   = MetaController()
    interv = ResidualIntervention()

    rng, k_meta, k_act = jax.random.split(rng, 3)

    ssum_j = carry[1]

    h_next, u_prop, sw, u_int_next, aux, _ = meta.apply(
        {"params": meta_params},
        h, obs_j, ssum_j, u_int, step_j,
        force_macro_switch=True, rng=k_meta
    )

    mean    = aux["mean"   ]
    logstd  = aux["logstd" ]
    sw_prob = aux["sw_prob"]

    logp_meta = gaussian_logprob(u_prop, mean, logstd) + bernoulli_logprob(sw, sw_prob)

    delta = interv.apply({"params": interv_params}, u_int_next, ssum_j)

    logits, carry_next, _ = base.apply(
        {"params": base_params}, obs_j, carry, step_j, delta
    )

    # Worker Sampling
    logp_w = jax.nn.log_softmax(logits, axis=-1)  # (B, A, BINS)
    B      = obs_j.shape[0]

    keys = jax.random.split(k_act, B * ACTION_DIM).reshape(B, ACTION_DIM, 2)

    def sample_tok(k, lp_vec):
        return jax.random.categorical(k, lp_vec, axis=-1)

    acts = jax.vmap(
        lambda keys_b, logp_b: jax.vmap(sample_tok)(keys_b, logp_b)
    )(keys, logp_w).astype(jnp.int32)  # (B, A)

    return h_next, u_int_next, carry_next, acts, u_prop, sw, logp_meta, rng


def score_from_info_dict(info_d):
    """
    Нэг env-ийн info dict-ээс score гаргана.
    """
    if info_d is None:
        return 0.0
    seq_idx = float(info_d.get("seq_goal_index", 0.0))
    succ    = float(info_d.get("is_success", 0.0))
    return (seq_idx * 10.0) + (50.0 if succ > 0.5 else 0.0)


def extract_final_infos(infos, B):
    """
    Gymnasium VectorEnv infos-ийн final_info бүтэцтэй ажиллах.
    - AsyncVectorEnv ихэвчлэн infos["final_info"] дээр дууссан env-ийн info dict-үүдийг өгдөг.
    """
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
        else:
            # Зарим тохиолдолд шууд dict-of-arrays байж болно (хуучин хэв маяг)
            # Энэ тохиолдолд бид score-оо шууд авах боломжгүй тул None үлдээнэ.
            pass

    elif isinstance(infos, (list, tuple)):
        # Хэрэв infos нь list of dicts бол шууд ашиглана
        for i in range(min(B, len(infos))):
            final_info[i] = infos[i]

    return final_info


def reset_states_where_done(h, u_int, carry, done_mask):
    """
    Дууссан орчнуудын санах ойг (Hidden states) тэглэх.
    """
    dm    = jnp.asarray(done_mask)
    mem, ssum = carry

    h     = jnp.where(dm[:, None], jnp.zeros_like(h), h)
    u_int = jnp.where(dm[:, None], jnp.zeros_like(u_int), u_int)

    mem   = jnp.where(dm[:, None, None], jnp.zeros_like(mem), mem)
    ssum  = jnp.where(dm[:, None], jnp.zeros_like(ssum), ssum)

    return h, u_int, (mem, ssum)


def collect_group_rollouts_vec(vec_env, trainer):
    """
    AsyncVectorEnv ашиглан олон Episode-ийг зэрэг цуглуулах (Vectorized Rollout).
    """
    obs, infos = vec_env.reset(seed=SEED)
    obs        = np.asarray(obs, dtype=np.float32)
    B          = obs.shape[0]

    carry = trainer.agent.base_model.init_carry(B)
    h     = jnp.zeros((B, GRU_H_DIM), dtype=jnp.float32)
    u_int = jnp.zeros((B, U_DIM), dtype=jnp.float32)

    step_in_macro = np.zeros((B,), dtype=np.int32)

    obs_buf   = [[] for _ in range(B)]
    ssum_buf  = [[] for _ in range(B)]
    step_buf  = [[] for _ in range(B)]
    uprop_buf = [[] for _ in range(B)]
    sw_buf    = [[] for _ in range(B)]
    act_buf   = [[] for _ in range(B)]
    oldlp_buf = [[] for _ in range(B)]
    decm_buf  = [[] for _ in range(B)]

    done_mask    = np.zeros((B,), dtype=bool)
    final_scores = np.zeros((B,), dtype=np.float32)

    base_params   = trainer.base_state.params
    meta_params   = trainer.meta_state.params["meta"  ]
    interv_params = trainer.meta_state.params["interv"]

    steps = 0

    while (not np.all(done_mask)) and (steps < MAX_EPISODE_STEPS):
        obs_j  = jnp.asarray(obs, dtype=jnp.float32)
        step_j = jnp.asarray(step_in_macro, dtype=jnp.int32)
        ssum_j = carry[1]

        h, u_int, carry, acts, u_prop, sw, lp_m, trainer.rng = policy_step_vec(
            base_params, meta_params, interv_params,
            obs_j, carry, h, u_int, step_j, trainer.rng
        )

        acts_np = np.asarray(acts, dtype=np.int32)

        for i in range(B):
            if not done_mask[i]:
                obs_buf  [i].append(obs[i])
                ssum_buf [i].append(np.asarray(ssum_j[i], dtype=np.float32))
                step_buf [i].append(int(step_in_macro[i]))
                uprop_buf[i].append(np.asarray(u_prop[i], dtype=np.float32))
                sw_buf   [i].append(int(sw[i]))
                act_buf  [i].append(acts_np[i])

                # PPO ratio нь зөвхөн Meta logprob дээр суурилна
                oldlp_buf[i].append(float(lp_m[i]))
                decm_buf [i].append(float(sw[i]))

        obs, _, term, trunc, infos = vec_env.step(acts_np)
        obs = np.asarray(obs, dtype=np.float32)

        step_done = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)

        if np.any(step_done):
            # Gymnasium VectorEnv: дууссан env-ийн info нь ихэвчлэн final_info дээр ирнэ
            finfos = extract_final_infos(infos, B)

            just_finished = step_done & (~done_mask)
            for i in np.where(just_finished)[0]:
                final_scores[i] = float(score_from_info_dict(finfos[i]))

            # Дууссан env-үүдийн hidden state-ийг тэглэнэ
            h, u_int, carry = reset_states_where_done(h, u_int, carry, just_finished)

            # Дууссан env-үүдийн macro step-ийг цэвэрхэн 0 болгоно
            step_in_macro[just_finished] = 0

        done_mask = done_mask | step_done

        steps += 1
        step_in_macro = (step_in_macro + 1) % MACRO_STEP

    # Дуусч амжаагүй env үүд: хамгийн сүүлийн infos-оор score тооцох (fallback)
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
            "act"      : np.asarray(act_buf  [i], dtype=np.int32  ),
            "old_logp" : np.asarray(oldlp_buf[i], dtype=np.float32),
            "decmask"  : np.asarray(decm_buf [i], dtype=np.float32),
        })

    return trajs, final_scores


def compute_grpo_advantages(scores):
    """
    GRPO Advantage тооцоолол. Baseline сүлжээ ашиглахгүйгээр бүлэг доторх дунджаас хазайлтыг тооцно.
    """
    mean = float(np.mean(scores))
    adv  = (scores - mean).astype(np.float32)
    adv  = np.clip(adv, -5.0, 5.0)
    return adv, mean


def pad_batch(trajs):
    """
    Ялгаатай урттай episode-уудыг нэг batch болгож padding хийх.
    """
    B     = len(trajs)
    max_t = 0
    for tr in trajs:
        max_t = max(max_t, int(tr["obs"].shape[0]))

    obs   = np.zeros((B, max_t, 28        ), dtype=np.float32)
    ssum  = np.zeros((B, max_t, EMBED_DIM ), dtype=np.float32)
    step  = np.zeros((B, max_t)            , dtype=np.int32  )
    uprop = np.zeros((B, max_t, U_DIM     ), dtype=np.float32)
    sw    = np.zeros((B, max_t)            , dtype=np.int32  )
    act   = np.zeros((B, max_t, ACTION_DIM), dtype=np.int32  )

    oldlp = np.zeros((B, max_t)            , dtype=np.float32)
    decm  = np.zeros((B, max_t)            , dtype=np.float32)
    mask  = np.zeros((B, max_t)            , dtype=np.float32)

    for i, tr in enumerate(trajs):
        T = int(tr["obs"].shape[0])
        if T == 0: continue

        obs  [i, :T] = tr["obs"     ]
        ssum [i, :T] = tr["ssum"    ]
        step [i, :T] = tr["step"    ]
        uprop[i, :T] = tr["u_prop"  ]
        sw   [i, :T] = tr["sw"      ]
        act  [i, :T] = tr["act"     ]

        oldlp[i, :T] = tr["old_logp"]
        decm [i, :T] = tr["decmask" ]
        mask [i, :T] = 1.0

    return (
        jnp.asarray(obs  , dtype=jnp.float32),
        jnp.asarray(ssum , dtype=jnp.float32),
        jnp.asarray(step , dtype=jnp.int32  ),
        jnp.asarray(uprop, dtype=jnp.float32),
        jnp.asarray(sw   , dtype=jnp.int32  ),
        jnp.asarray(act  , dtype=jnp.int32  ),
        jnp.asarray(oldlp, dtype=jnp.float32),
        jnp.asarray(decm , dtype=jnp.float32),
        jnp.asarray(mask , dtype=jnp.float32),
    )


# PPO SCAN (RECONSTRUCT + WORKER LOGPROB)

@jax.jit
def meta_forward_scan_actions(params_meta, obs_bt, ssum_bt, step_bt, uprop_bt, sw_bt):
    """
    Rollout дээр хийгдсэн үйлдлүүдийг (uprop, sw) ашиглан u_int дарааллыг дахин сэргээх.
    """
    meta = MetaController()
    B, T, _ = obs_bt.shape

    h0      = jnp.zeros((B, GRU_H_DIM), dtype=jnp.float32)
    u_int0  = jnp.zeros((B, U_DIM), dtype=jnp.float32)

    obs_T   = jnp.swapaxes(obs_bt  , 0, 1)
    ssum_T  = jnp.swapaxes(ssum_bt , 0, 1)
    step_T  = jnp.swapaxes(step_bt , 0, 1)
    uprop_T = jnp.swapaxes(uprop_bt, 0, 1)
    sw_T    = jnp.swapaxes(sw_bt   , 0, 1)

    def scan_fn(carry, x):
        h_prev, u_int_prev = carry
        obs_t, ssum_t, step_t, uprop_t, sw_t = x

        h_t, _, _, _, aux, _ = meta.apply(
            {"params": params_meta},
            h_prev, obs_t, ssum_t, u_int_prev, step_t,
            force_macro_switch=True, rng=None
        )

        mean   = aux["mean"   ]
        swprob = aux["sw_prob"]
        beta   = aux["beta"   ]

        is_boundary = (step_t == 0).astype(jnp.int32)
        sw_used     = jnp.maximum(sw_t.astype(jnp.int32), is_boundary)

        sw_f     = sw_used.astype(jnp.float32)[:, None]
        beta_eff = beta.astype(jnp.float32)[:, None] * (1.0 - sw_f)
        u_int_t  = (beta_eff * u_int_prev) + ((1.0 - beta_eff) * uprop_t)

        carry2 = (h_t, u_int_t)
        out    = (mean, swprob, beta, u_int_t)
        return carry2, out

    (_, _), outs = jax.lax.scan(scan_fn, (h0, u_int0), (obs_T, ssum_T, step_T, uprop_T, sw_T))
    mean_T, swprob_T, beta_T, uint_T = outs

    return (
        jnp.swapaxes(mean_T  , 0, 1),
        jnp.swapaxes(swprob_T, 0, 1),
        jnp.swapaxes(beta_T  , 0, 1),
        jnp.swapaxes(uint_T  , 0, 1)
    )


@jax.jit
def worker_logprob_scan(params_base, params_interv, obs_bt, step_bt, u_int_bt, actions_bt):
    """
    Worker token logprob-ийг тооцоолно. (Decoder/Intervention gradient bridge)
    """
    base   = BaseARPolicy()
    interv = ResidualIntervention()

    B, T, _ = obs_bt.shape
    carry0  = base.init_carry(B)

    obs_T   = jnp.swapaxes(obs_bt    , 0, 1)
    step_T  = jnp.swapaxes(step_bt   , 0, 1)
    uint_T  = jnp.swapaxes(u_int_bt  , 0, 1)
    act_T   = jnp.swapaxes(actions_bt, 0, 1)

    def scan_fn(carry, x):
        obs_t, step_t, u_int_t, act_t = x
        mem, ssum = carry

        delta = interv.apply({"params": params_interv}, u_int_t, ssum)

        logits, carry2, _ = base.apply({"params": params_base}, obs_t, carry, step_t, delta)

        logp = jax.nn.log_softmax(logits, axis=-1)
        a    = act_t[..., None]
        sel  = jnp.take_along_axis(logp, a, axis=-1).squeeze(-1)
        lpw  = jnp.sum(sel, axis=-1)
        return carry2, lpw

    _, lpw_T = jax.lax.scan(scan_fn, carry0, (obs_T, step_T, uint_T, act_T))
    return jnp.swapaxes(lpw_T, 0, 1)


# PPO UPDATE (META + DECODER)

@jax.jit
def ppo_update_step(meta_state, ref_params_bundle, base_params, obs, ssum, step, uprop, sw, act, oldlp, adv_ep, decmask, mask, kl_beta):
    """
    MetaController болон Decoder-ийг сургах PPO Loss.
    PPO ratio нь зөвхөн Meta logprob дээр суурилна.
    Decoder gradient-ийг worker_logprob auxiliary loss-аар хадгална.
    """
    def loss_fn(p_bundle):
        p_meta   = p_bundle["meta"  ]
        p_interv = p_bundle["interv"]

        mean_bt, swprob_bt, beta_bt, u_int_bt = meta_forward_scan_actions(
            p_meta, obs, ssum, step, uprop, sw
        )
        mean_ref_bt, swprob_ref_bt, _, _ = meta_forward_scan_actions(
            ref_params_bundle["meta"], obs, ssum, step, uprop, sw
        )

        logstd_new = p_meta["proposal"]["u_logstd"]
        B, T, _    = uprop.shape

        u_flat     = uprop.reshape(B*T, U_DIM)
        mean_flat  = mean_bt.reshape(B*T, U_DIM)
        newlp_u    = gaussian_logprob(u_flat, mean_flat, logstd_new).reshape(B, T)

        sw_i       = sw.astype(jnp.int32)
        newlp_sw   = bernoulli_logprob(sw_i, swprob_bt)
        newlp_meta = newlp_u + newlp_sw

        ratio       = jnp.exp(newlp_meta - oldlp)

        adv_bt      = adv_ep[:, None]
        unclipped   = ratio * adv_bt
        clipped     = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_bt
        surr        = jnp.minimum(unclipped, clipped)

        m_all       = mask
        m_dec       = mask * decmask

        surr_mean   = jnp.sum(surr * m_dec) / (jnp.sum(m_dec) + 1e-8)

        logstd_ref  = ref_params_bundle["meta"]["proposal"]["u_logstd"]
        mean_ref_f  = mean_ref_bt.reshape(B*T, U_DIM)

        kl_u        = gaussian_kl(mean_flat, logstd_new, mean_ref_f, logstd_ref).reshape(B, T)
        kl_sw       = bernoulli_kl(swprob_bt, swprob_ref_bt)
        kl_bt       = kl_u + kl_sw
        kl_mean     = jnp.sum(kl_bt * m_dec) / (jnp.sum(m_dec) + 1e-8)

        sw_mean     = jnp.sum(swprob_bt * m_all) / (jnp.sum(m_all) + 1e-8)
        sw_rate_pen = (sw_mean - TARGET_SW_RATE) ** 2

        beta_mean   = jnp.sum(beta_bt * m_all) / (jnp.sum(m_all) + 1e-8)
        beta_var    = jnp.sum(((beta_bt - beta_mean) ** 2) * m_all) / (jnp.sum(m_all) + 1e-8)
        beta_pen    = (BETA_MEAN_COEFF * (beta_mean - BETA_MEAN_TARGET)**2) + \
                      (BETA_VAR_COEFF  * (beta_var))

        ent_u       = gaussian_entropy(logstd_new)
        ent_sw      = jnp.sum(bernoulli_entropy(swprob_bt) * m_all) / (jnp.sum(m_all) + 1e-8)
        ent_bonus   = (U_ENTROPY_COEFF * ent_u) + (SWITCH_ENTROPY_COEFF * ent_sw)

        logp_worker_bt = worker_logprob_scan(
            base_params, p_interv, obs, step, u_int_bt, act
        )
        dec_pg = jnp.sum((logp_worker_bt * adv_bt) * m_dec) / (jnp.sum(m_dec) + 1e-8)

        loss = -surr_mean + (kl_beta * kl_mean) + (SW_RATE_COEFF * sw_rate_pen) + beta_pen - ent_bonus - (DECODER_PG_COEFF * dec_pg)

        aux = (kl_mean, ent_u, ent_sw, sw_mean, beta_mean, beta_var, dec_pg)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(meta_state.params)
    meta_state = meta_state.apply_gradients(grads=grads)
    return meta_state, loss, aux


def ppo_epoch(trainer, ref_params_bundle, trajs, advs, kl_beta):
    """
    Цуглуулсан өгөгдөл дээр нэг удаагийн PPO давтамж гүйх.
    """
    idx = np.arange(len(trajs))
    np.random.shuffle(idx)

    stats = [0.0] * 8

    for i in range(0, len(trajs), MINI_BATCH_SIZE):
        mb_idx   = idx[i:i + MINI_BATCH_SIZE]
        mb_trajs = [trajs[j] for j in mb_idx]
        mb_adv   = jnp.asarray(advs[mb_idx], dtype=jnp.float32)

        obs, ssum, step, uprop, sw, act, oldlp, decm, mask = pad_batch(mb_trajs)

        trainer.meta_state, loss, aux = ppo_update_step(
            trainer.meta_state,
            ref_params_bundle,
            trainer.base_state.params,
            obs, ssum, step,
            uprop, sw, act, oldlp,
            mb_adv, decm, mask,
            kl_beta
        )

        stats[0] = float(loss)
        for k in range(7): stats[k+1] = float(aux[k])

    return stats


def play_policy_human(trainer, episodes=1):
    """
    Сургалтын явцад бодит env дээр сурсан policy-г үзүүлэх (render=human).
    - Meta   : deterministic (rng=None)
    - Worker : deterministic argmax
    """
    env = make_single_env(999, render_mode="human")()

    base_params   = trainer.base_state.params
    meta_params   = trainer.meta_state.params["meta"  ]
    interv_params = trainer.meta_state.params["interv"]

    base   = BaseARPolicy()
    meta   = MetaController()
    interv = ResidualIntervention()

    for ep in range(int(episodes)):
        obs, info = env.reset(seed=SEED + 999 + ep)
        obs       = np.asarray(obs, dtype=np.float32)

        carry = trainer.agent.base_model.init_carry(1)
        h     = jnp.zeros((1, GRU_H_DIM), dtype=jnp.float32)
        u_int = jnp.zeros((1, U_DIM), dtype=jnp.float32)

        step_in_macro = 0
        done          = False
        t             = 0

        while (not done) and (t < MAX_EPISODE_STEPS):
            obs_j  = jnp.asarray(obs[None, :], dtype=jnp.float32)
            step_j = jnp.asarray([step_in_macro], dtype=jnp.int32)
            ssum_j = carry[1]

            # Meta (Deterministic)
            h, u_prop, sw, u_int, aux, _ = meta.apply(
                {"params": meta_params},
                h, obs_j, ssum_j, u_int, step_j,
                force_macro_switch=True,
                rng=None
            )

            # Intervention
            delta = interv.apply({"params": interv_params}, u_int, ssum_j)

            # Worker (Deterministic argmax)
            logits, carry, _ = base.apply(
                {"params": base_params},
                obs_j, carry, step_j, delta
            )

            # logits, (1, ACTION_DIM, ACTION_BINS)
            act    = jnp.argmax(logits[0], axis=-1).astype(jnp.int32)  # (ACTION_DIM,)
            act_np = np.asarray(act, dtype=np.int32)

            obs, r, term, trunc, info = env.step(act_np)
            obs = np.asarray(obs, dtype=np.float32)

            t += 1
            step_in_macro = (step_in_macro + 1) % MACRO_STEP

            if PLAY_SLEEP_SEC > 0.0:
                time.sleep(float(PLAY_SLEEP_SEC))

            if term or trunc:
                done = True

    env.close()


# MAIN EXECUTION

def main():
    np.random.seed(SEED)
    random.seed(SEED)

    trainer = Trainer(seed=SEED)

    # PHASE 1, SFT Warm Start
    if SFT_ENABLE and (not os.path.exists(SFT_FLAG)):
        print("[SFT] Heuristic өгөгдөл үүсгэж байна...")

        sft_env          = make_single_env(0, render_mode=None)()
        obs_all, act_all = generate_sft_dataset(sft_env, episodes=SFT_EPISODES)
        sft_env.close()

        print(f"[SFT] Нийт алхам: {len(obs_all)} | Batch: {SFT_BATCH_SIZE}")
        sft_train(trainer, obs_all, act_all)

        with open(SFT_FLAG, "w") as f:
            f.write("done")

    vec_env = create_vec_env(GROUP_SIZE, render_mode=None)
    kl_beta = float(KL_BETA)

    print("\n" + "=" * 72)
    print("  INTERNAL RL - Base Frozen, Meta+Decoder PPO+GRPO (ASYNC VECTOR)")
    print(f"  Updates: {UPDATES} | Group: {GROUP_SIZE} | MacroMax: {MACRO_STEP}")
    print(f"  U_DIM: {U_DIM} | GRU_H: {GRU_H_DIM} | Embed: {EMBED_DIM} | Rank: {LRANK}")
    print("=" * 72 + "\n")

    for upd in range(1, UPDATES + 1):
        ref_params_bundle = trainer.meta_state.params

        trajs, scores    = collect_group_rollouts_vec(vec_env, trainer)
        advs, mean_score = compute_grpo_advantages(scores)

        stats = []
        for _ in range(PPO_EPOCHS):
            stats = ppo_epoch(trainer, ref_params_bundle, trajs, advs, kl_beta)

        loss_v, kl_v, ent_u, ent_sw, sw_m, beta_m, beta_v, dec_pg = stats

        if kl_v > TARGET_KL * 1.5:
            kl_beta *= KL_ALPHA
        elif kl_v < TARGET_KL / 1.5:
            kl_beta /= KL_ALPHA

        if upd % 5 == 0 or upd == 1:
            best = float(np.max(scores)) if len(scores) else 0.0
            print(
                f"[UPD {upd:4d}] "
                f"Score: {mean_score:6.2f} | Best: {best:6.2f} | "
                f"KL: {kl_v:.4f} | BetaKL: {kl_beta:.3f} | "
                f"SW%: {sw_m:.2f} | BetaAvg: {beta_m:.2f} | "
                f"DecPG: {dec_pg:.3f}"
            )
        
        if PLAY_ENABLE and (upd % PLAY_EVERY == 0):
            play_policy_human(trainer, episodes=PLAY_EPISODES)

    vec_env.close()


if __name__ == "__main__":
    main()
