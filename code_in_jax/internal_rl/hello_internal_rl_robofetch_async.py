#
# INTERNAL RL - ROBOTIC CWRRT + PPO(GRPO) TUNING (ASYNC VECTORIZED FETCH)
#
# RoboticCWRRT, Robotic Cross-Window Residual Recurrent Transformer, Sharavsambuu.G (2026/01/09)
#
# ЛАВЛАГАА:
#   - Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning
#     https://arxiv.org/abs/2512.20605
#
#
# ЗОРИЛГО:
#   FetchPush орчинг Autoregressive token generation хэлбэртэй болгож
#   Internal RL / Temporal Abstraction санааг шалгах туршилт.
#   Энэ удаад AsyncVectorEnv ашиглан олон орчинг зэрэг ажиллуулж (Parallel Rollout),
#   өгөгдөл цуглуулах хурдыг (Throughput) нэмэгдүүлнэ.
#
#   1) Action      : Continuous (physics) -> Discrete tokens (0..255) per action dim
#   2) Observation : Dict -> flat vector (28,)
#   3) Task        : K дараалсан sub-goal (pinpad) бүхий HRL даалгавар.
#
#
# АРХИТЕКТУР (RoboticCWRRT - Improved):
#   - Sliding Window Memory: Attention механизм нь зөвхөн одоогийн оролтыг бус
#     сүүлийн MEMORY_LEN алхмын мэдээллийг харж шийдвэр гаргана.
#   - ssum (Intent): Урт хугацааны зорилго. THINK үе дээр шинэчлэгдэж ACT үе дээр царцана (frozen).
#   - Time Embedding: Macro step доторх цаг хугацааны баримжааг модельд өгнө.
#
#
# СУРГАЛТЫН ҮЕ ШАТУУД:
#   PHASE 1 - SFT (Warm Start):
#     - Heuristic policy-оор цуглуулсан өгөгдөл дээр Supervised Learning хийнэ.
#
#   PHASE 2 - PPO + GRPO (Async):
#     - Group Rollout (GROUP_SIZE) цуглуулна. Орчин бүр тусдаа process дээр ажиллана.
#     - Episode дуусах үед тухайн орчны carry (mem, ssum) төлөвийг тэглэх (Reset)
#       логикийг reset_carry_where_done функцээр шийднэ.
#     - GRPO advantage: A_i = score_i - mean(score_group).
#
#


import os
import math
import random
import numpy as np
import time
import optax
import gymnasium          as gym
import gymnasium_robotics
import jax
import jax.numpy          as jnp
import flax.linen         as nn
from flax.training        import train_state


# GLOBAL CONFIGURATION

SEED                 = 42

# Environment Settings
ENV_ID               = "FetchPush-v4"
ACTION_BINS          = 256
ACTION_DIM           = 4

MAX_EPISODE_STEPS    = 200
K_SEQUENTIAL_GOALS   = 3
GOAL_DIST_THRESHOLD  = 0.05
SPARSE_FINAL_REWARD  = True

# Internal RL / Temporal Abstraction
MACRO_STEP           = 10     # Нэг intent барих хугацаа

# Vectorized Rollout Settings
GROUP_SIZE           = 16     # Зэрэг ажиллах орчны тоо (Parallel Envs)

# PPO + GRPO Hyperparameters
UPDATES              = 200
PPO_EPOCHS           = 3
MINI_BATCH_SIZE      = 16
CLIP_EPS             = 0.2
ENTROPY_COEFF        = 0.01

# KL Regularization
KL_BETA              = 0.04
TARGET_KL            = 0.05
KL_ALPHA             = 1.2

# Optimizer & Model Settings
LR_SFT               = 3e-4
LR_RL                = 1e-5
MAX_GRAD_NORM        = 1.0

EMBED_DIM            = 256
NUM_HEADS            = 4
MEMORY_LEN           = 16     # Sliding window memory size

# Warm Start
SFT_ENABLE           = True
SFT_FLAG             = "sft_done_fetch_async.flag"
SFT_EPISODES         = 200
SFT_EPOCHS           = 3
SFT_BATCH_SIZE       = 256

# JAX Configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# ENV WRAPPERS

class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Observation Dict -> Flat Float32 Vector.
    Бүтэц: [observation (25)] + [desired_goal (3)] = 28
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
    Continuous [-1,1] <-> Discrete Tokens [0, bins-1] хөрвүүлэгч.
    """
    def __init__(self, bins=256):
        self.bins = int(bins)

    def encode(self, continuous_action):
        clipped = np.clip(continuous_action, -1.0, 1.0)
        norm    = (clipped + 1.0) / 2.0
        return (norm * (self.bins - 1)).astype(np.int32)

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float32)
        norm = tokens / (self.bins - 1)
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
        return self.tokenizer.decode(np.array(token_action)).astype(np.float32)


class SequentialGoalsWrapper(gym.Wrapper):
    """
    Pinpad буюу дараалсан K зорилготой болгох wrapper.
    Агент эхний зорилгод хүрвэл дараагийн зорилго идэвхжинэ.
    """
    def __init__(self, env, k=3, dist_threshold=0.05, sparse_final_reward=True):
        super().__init__(env)
        self.k                   = int(k)
        self.dist_threshold      = float(dist_threshold)
        self.sparse_final_reward = bool(sparse_final_reward)
        self._goals              = []
        self._goal_idx           = 0

    def _update_obs(self, obs):
        obs = dict(obs)
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

        reached = dist < self.dist_threshold

        info = dict(info)
        info["seq_goal_index"    ] = self._goal_idx
        info["seq_reached"       ] = reached
        info["seq_dist"          ] = dist
        info["distance_threshold"] = self.dist_threshold

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
    Олон процессыг зэрэгцүүлэн ажиллуулж тооцооллыг хурдасгах AsyncVectorEnv үүсгэх.
    """
    fns = [make_single_env(i, render_mode=render_mode) for i in range(n_envs)]
    return gym.vector.AsyncVectorEnv(fns)


# MODEL ARCHITECTURE

class CWRRTCell(nn.Module):
    """
    CWRRT Cell with Sliding Window Memory.
    Attention нь одоо ганц токен дээр биш, санах ой (mem) дээр ажиллана.
    """
    embed_dim: int
    num_heads: int
    mem_len  : int = MEMORY_LEN

    @nn.compact
    def __call__(self, carry, x):
        mem, ssum = carry
        # mem shape : (Batch, MEM_LEN, Embed)
        # ssum shape: (Batch, Embed)

        # Intent Injection
        alpha = nn.sigmoid(self.param("alpha", nn.initializers.zeros, (self.embed_dim,)))
        x_in  = x + (ssum * alpha)

        # Sliding Window Update
        new_mem_entry = x_in[:, None, :] # (B, 1, E)
        updated_mem   = jnp.concatenate([mem[:, 1:, :], new_mem_entry], axis=1)

        # Memory Attention
        y = nn.LayerNorm()(x_in)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=y[:, None, :], 
            inputs_kv=updated_mem
        )
        y = y.squeeze(1)
        x_mid = x_in + y

        # MLP Block
        y = nn.LayerNorm()(x_mid)
        y = nn.Dense(self.embed_dim * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim)(y)
        x_out = x_mid + y

        # Intent Update
        lam      = nn.sigmoid(self.param("lambda", nn.initializers.zeros, (self.embed_dim,)))
        new_ssum = (ssum * lam) + (x_out * (1.0 - lam))

        return (updated_mem, new_ssum), x_out


class RoboticCWRRT(nn.Module):
    """
    Policy Network with Time Embeddings.
    Async ажиллагааг дэмжих зорилгоор time_idx-г гаднаас авна.
    """
    action_dim : int = ACTION_DIM
    action_bins: int = ACTION_BINS
    embed_dim  : int = EMBED_DIM
    mem_len    : int = MEMORY_LEN

    def setup(self):
        self.input_proj  = nn.Dense(self.embed_dim)
        self.time_embed  = nn.Embed(MACRO_STEP + 1, self.embed_dim) # Time awareness
        self.cell        = CWRRTCell(embed_dim=self.embed_dim, num_heads=NUM_HEADS, mem_len=self.mem_len)
        self.action_head = nn.Dense(self.action_dim * self.action_bins)

    def init_carry(self, batch_size):
        return (
            jnp.zeros((batch_size, self.mem_len, self.embed_dim)), # mem history
            jnp.zeros((batch_size, self.embed_dim)),               # ssum intent
        )

    def __call__(self, obs, carry, time_idx, freeze_thought=False):
        mem, ssum = carry

        x     = self.input_proj(obs)
        t_emb = self.time_embed(jnp.clip(time_idx, 0, MACRO_STEP))
        x     = x + t_emb

        (new_mem, new_ssum), x_out = self.cell((mem, ssum), x)

        is_frozen = jnp.asarray(freeze_thought)
        if is_frozen.ndim > 0:
            is_frozen = is_frozen[..., None]

        # Freeze Logic, Think үед ssum шинэчлэгдэнэ Act үед хуучнаараа үлдэнэ
        next_ssum  = jnp.where(is_frozen, ssum, new_ssum)
        # Mem (Context) үргэлж шинэчлэгдэнэ
        next_carry = (new_mem, next_ssum)

        logits = self.action_head(x_out)
        logits = logits.reshape(obs.shape[0], self.action_dim, self.action_bins)
        return logits, next_carry

    def forward_sequence(self, obs_btf, freeze_btf):
        """
        Training үед бүтэн дарааллыг боловсруулах (Scan).
        """
        B, T, _ = obs_btf.shape
        carry0  = self.init_carry(B)

        obs_T   = jnp.swapaxes(obs_btf, 0, 1)
        frz_T   = jnp.swapaxes(freeze_btf, 0, 1)
        
        # Time index generated on the fly for training
        time_T  = jnp.arange(T) % MACRO_STEP
        time_T  = jnp.tile(time_T[:, None], (1, B))

        def scan_fn(carry, inp):
            obs_t, frz_t, t_idx = inp
            logits_t, carry     = self.__call__(obs_t, carry, t_idx, freeze_thought=frz_t)
            return carry, logits_t

        _, logits_T = jax.lax.scan(scan_fn, carry0, (obs_T, frz_T, time_T))
        return jnp.swapaxes(logits_T, 0, 1)


# DISTRIBUTIONS & METRICS

@jax.jit
def logprob_bt_from_logits(logits_btav, actions_bta):
    logp  = jax.nn.log_softmax(logits_btav, axis=-1)
    taken = jnp.take_along_axis(logp, actions_bta[..., None], axis=-1).squeeze(-1)
    return jnp.sum(taken, axis=-1)

@jax.jit
def entropy_bt_from_logits(logits_btav):
    p   = jax.nn.softmax(logits_btav, axis=-1)
    lp  = jax.nn.log_softmax(logits_btav, axis=-1)
    ent = -jnp.sum(p * lp, axis=-1)
    return jnp.mean(ent, axis=-1)

@jax.jit
def kl_bt_from_logits(logits_new_btav, logits_ref_btav):
    p_new  = jax.nn.softmax(logits_new_btav, axis=-1)
    lp_new = jax.nn.log_softmax(logits_new_btav, axis=-1)
    lp_ref = jax.nn.log_softmax(logits_ref_btav, axis=-1)
    kl     = jnp.sum(p_new * (lp_new - lp_ref), axis=-1)
    return jnp.mean(kl, axis=-1)


# TRAINING STATE

class Trainer:
    def __init__(self, seed=SEED):
        self.model  = RoboticCWRRT()
        self.rng    = jax.random.PRNGKey(seed)

        dummy_obs   = jnp.zeros((1, 28), dtype=jnp.float32)
        dummy_carry = self.model.init_carry(1)
        dummy_t     = jnp.array([0], dtype=jnp.int32)
        params      = self.model.init(self.rng, dummy_obs, dummy_carry, dummy_t, freeze_thought=False)["params"]

        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR_SFT),
            ),
        )

    def set_lr(self, lr):
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            params   = self.state.params,
            tx       = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(lr),
            ),
        )


# ASYNC VECTORIZED ROLLOUTS

@jax.jit
def step_think(params, obs_bf, carry):
    """
    THINK Phase: Intent шинэчлэх (freeze=False), time_idx=0 гэж үзнэ.
    """
    model = RoboticCWRRT()
    t_idx = jnp.zeros((obs_bf.shape[0],), dtype=jnp.int32)
    return model.apply({"params": params}, obs_bf, carry, t_idx, freeze_thought=False)

@jax.jit
def step_act(params, obs_bf, carry, time_idx_scalar):
    """
    ACT Phase: Intent царцсан (freeze=True), time_idx-ийг алхам бүрт нэмэгдүүлж дамжуулна.
    """
    model = RoboticCWRRT()
    B     = obs_bf.shape[0]
    t_idx = jnp.full((B,), time_idx_scalar, dtype=jnp.int32)
    return model.apply({"params": params}, obs_bf, carry, t_idx, freeze_thought=True)

@jax.jit
def sample_actions_and_oldlp(rng, logits_bav):
    """
    Logits-оос үйлдэл санамсаргүй сонгох (Sampling).
    """
    logp_bav = jax.nn.log_softmax(logits_bav, axis=-1)
    B, A, V  = logits_bav.shape

    rngs = jax.random.split(rng, B * A).reshape(B, A, 2)

    def cat_one(r, lp):
        return jax.random.categorical(r, lp, axis=-1)

    acts_ba  = jax.vmap(lambda rr, lp: jax.vmap(cat_one)(rr, lp))(rngs, logp_bav).astype(jnp.int32)

    taken_lp = jnp.take_along_axis(logp_bav, acts_ba[..., None], axis=-1).squeeze(-1)
    oldlp_b  = jnp.sum(taken_lp, axis=-1)
    return acts_ba, oldlp_b


def calculate_scores_batch(infos, group_size):
    """
    AsyncVectorEnv-ийн info dict-ээс оноог задалж авах.
    """
    if "seq_goal_index" in infos:
        seq_idx = np.asarray(infos["seq_goal_index"], dtype=np.float32)
    else:
        seq_idx = np.zeros(group_size, dtype=np.float32)

    if "is_success" in infos:
        is_succ = np.asarray(infos["is_success"], dtype=np.float32)
    else:
        is_succ = np.zeros(group_size, dtype=np.float32)

    return (seq_idx * 10.0) + ((is_succ > 0.5).astype(np.float32) * 50.0)


def reset_carry_where_done(carry, done_mask):
    """
    Дууссан (Done) орчны санах ойг (Carry) тэглэх.
    Бусад үргэлжилж буй орчны санах ойг хэвээр үлдээнэ.
    """
    mem, ssum = carry
    dm        = jnp.asarray(done_mask) # (B,)

    mem0  = jnp.zeros_like(mem)
    ssum0 = jnp.zeros_like(ssum)

    # Done=True бол 0-р солино, Done=False бол хуучнаар нь
    mem  = jnp.where(dm[:, None, None], mem0,  mem)
    ssum = jnp.where(dm[:, None],      ssum0, ssum)

    return (mem, ssum)


def collect_group_rollouts_vec(vec_env, trainer):
    """
    AsyncVectorEnv ашиглан олон Episode-ийг зэрэг цуглуулах.
    Анхаарах:
      - Орчин бүр өөр цагт дуусаж болно.
      - Дууссан орчны Carry-г reset хийх ёстой.
      - Macro Step дундуур орчин дуусвал шинэ episode эхэлнэ.
    """
    obs, infos = vec_env.reset(seed=SEED)
    obs        = np.asarray(obs, dtype=np.float32)

    B     = obs.shape[0]
    carry = trainer.model.init_carry(B)

    done_mask    = np.zeros((B,), dtype=bool      )
    final_scores = np.zeros((B,), dtype=np.float32)

    obs_buf   = [[] for _ in range(B)]
    act_buf   = [[] for _ in range(B)]
    oldlp_buf = [[] for _ in range(B)]

    steps = 0
    # Бүх орчин дуустал эсвэл хамгийн их алхам хүртэл давтана
    while (not np.all(done_mask)) and (steps < MAX_EPISODE_STEPS):
        
        # THINK PHASE
        # Эхний алхам дээр бодно (Intent Update).
        obs_j    = jnp.asarray(obs, dtype=jnp.float32)
        _, carry = step_think(trainer.state.params, obs_j, carry)

        # ACT PHASE (Macro Loop)
        for m_step in range(MACRO_STEP):
            if np.all(done_mask) or steps >= MAX_EPISODE_STEPS:
                break

            obs_j = jnp.asarray(obs, dtype=jnp.float32)
            # m_step-ийг time embedding болгож өгнө
            logits, carry = step_act(trainer.state.params, obs_j, carry, m_step)

            trainer.rng, subkey = jax.random.split(trainer.rng)
            acts_j, oldlp_j     = sample_actions_and_oldlp(subkey, logits)

            acts  = np.asarray(acts_j , dtype=np.int32  )
            oldlp = np.asarray(oldlp_j, dtype=np.float32)

            # Buffer руу хадгалах (Зөвхөн дуусаагүй орчнуудыг)
            for i in range(B):
                if not done_mask[i]:
                    obs_buf  [i].append(obs [i].copy())
                    act_buf  [i].append(acts[i].copy())
                    oldlp_buf[i].append(float(oldlp[i]))

            # Environment Step (Async)
            obs, _, term, trunc, infos = vec_env.step(acts)
            obs = np.asarray(obs, dtype=np.float32)

            step_done = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)

            # Дууссан мөчид оноог хадгалах
            current_scores              = calculate_scores_batch(infos, B)
            just_finished               = step_done & (~done_mask)
            final_scores[just_finished] = current_scores[just_finished]

            done_mask = done_mask | step_done
            steps    += 1

            # Дууссан орчнуудын Carry-г шинэчлэх (Reset logic)
            if np.any(just_finished):
                carry = reset_carry_where_done(carry, just_finished)

    # Хэрэв Max Step хүрээд дуусаагүй орчин байвал оноог нь авна
    if np.any(~done_mask):
        current_scores           = calculate_scores_batch(infos, B)
        final_scores[~done_mask] = current_scores[~done_mask]

    trajs = []
    for i in range(B):
        o  = np.asarray(obs_buf  [i], dtype=np.float32)
        a  = np.asarray(act_buf  [i], dtype=np.int32  )
        lp = np.asarray(oldlp_buf[i], dtype=np.float32)
        trajs.append({"obs": o, "actions": a, "old_logp": lp})

    return trajs, final_scores


def compute_grpo_advantages(scores):
    """
    GRPO Advantage Calculation.
    """
    mean = float(np.mean(scores))
    adv  = (scores - mean).astype(np.float32)
    adv  = np.clip(adv, -5.0, 5.0)
    return adv, mean


# PPO + GRPO UPDATE (BATCHED)

def build_freeze_schedule(B, T, mask):
    """
    Training-д зориулж Freeze mask үүсгэх.
    t % MACRO_STEP == 0 үед False (Think), бусад үед True (Act).
    """
    t      = np.arange(T)
    freeze = (t % MACRO_STEP != 0)[None, :].repeat(B, axis=0)
    freeze = freeze & (mask > 0.0) # Padding хэсэгт хамаагүй
    return freeze.astype(np.bool_)


def pad_minibatch(trajs):
    """
    Variable length trajectories -> Padded Batch.
    """
    B       = len(trajs)
    max_len = max(int(t["obs"].shape[0]) for t in trajs) if B > 0 else 0

    obs   = np.zeros((B, max_len, 28), dtype=np.float32)
    acts  = np.zeros((B, max_len, 4) , dtype=np.int32  )
    oldlp = np.zeros((B, max_len)    , dtype=np.float32)
    mask  = np.zeros((B, max_len)    , dtype=np.float32)

    for i, t in enumerate(trajs):
        T = int(t["obs"].shape[0])
        if T == 0: continue
        obs  [i, :T] = t["obs"     ]
        acts [i, :T] = t["actions" ]
        oldlp[i, :T] = t["old_logp"]
        mask [i, :T] = 1.0

    freeze = build_freeze_schedule(B, max_len, mask)

    return (
        jnp.asarray(obs   , dtype=jnp.float32),
        jnp.asarray(acts  , dtype=jnp.int32  ),
        jnp.asarray(oldlp , dtype=jnp.float32),
        jnp.asarray(mask  , dtype=jnp.float32),
        jnp.asarray(freeze, dtype=jnp.bool_  ),
    )


@jax.jit
def ppo_grpo_update_step(state, ref_params, obs, acts, oldlp, adv, mask, freeze, kl_beta):
    """
    PPO Loss Step
    """
    def loss_fn(p):
        model = RoboticCWRRT()

        # Forward Pass with Time Embeddings logic inside forward_sequence
        logits_new = model.apply({"params": p         }, obs, freeze, method=RoboticCWRRT.forward_sequence)
        logits_ref = model.apply({"params": ref_params}, obs, freeze, method=RoboticCWRRT.forward_sequence)

        newlp_bt = logprob_bt_from_logits(logits_new, acts)
        ratio    = jnp.exp(newlp_bt - oldlp)

        adv_bt    = adv[:, None]
        unclipped = ratio * adv_bt
        clipped   = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_bt
        surr      = jnp.minimum(unclipped, clipped)

        kl_bt  = kl_bt_from_logits(logits_new, logits_ref)
        ent_bt = entropy_bt_from_logits(logits_new)

        m         = mask
        surr_mean = jnp.sum(surr   * m) / (jnp.sum(m) + 1e-8)
        kl_mean   = jnp.sum(kl_bt  * m) / (jnp.sum(m) + 1e-8)
        ent_mean  = jnp.sum(ent_bt * m) / (jnp.sum(m) + 1e-8)

        loss = -surr_mean + (kl_beta * kl_mean) - (ENTROPY_COEFF * ent_mean)
        return loss, (kl_mean, ent_mean)

    (loss, (kl_mean, ent_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, kl_mean, ent_mean


def ppo_grpo_epochs(trainer, ref_params, trajs, advs, kl_beta):
    idx = np.arange(len(trajs))
    last_loss, last_kl, last_ent = 0.0, 0.0, 0.0

    for _ep in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(trajs), MINI_BATCH_SIZE):
            mb_idx   = idx[i:i + MINI_BATCH_SIZE]
            mb_trajs = [trajs[j] for j in mb_idx]
            mb_adv   = jnp.asarray(advs[mb_idx], dtype=jnp.float32)

            obs, acts, oldlp, mask, freeze = pad_minibatch(mb_trajs)

            trainer.state, loss, kl_m, ent_m = ppo_grpo_update_step(
                trainer.state, ref_params, obs, acts, oldlp, mb_adv, mask, freeze, kl_beta
            )

            last_loss = float(loss)
            last_kl   = float(kl_m)
            last_ent  = float(ent_m)

    return last_loss, last_kl, last_ent


# WARM START (SFT)

def generate_sft_dataset(env, episodes=SFT_EPISODES):
    """
    SFT Dataset үүсгэх (Heuristic).
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

            delta       = (target - grip_pos) * 5.0
            delta       = np.clip(delta, -1.0, 1.0)
            cont_action = np.append(delta, -1.0).astype(np.float32)

            tokens = env.tokenizer.encode(cont_action).astype(np.int32)

            obs_list.append(obs.astype(np.float32))
            act_list.append(tokens.astype(np.int32))

            obs, _, term, trunc, info = env.step(tokens)
            if term or trunc:
                break

    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int32)


def sft_train(trainer, obs_all, act_all):
    N   = len(obs_all)
    idx = np.arange(N)

    @jax.jit
    def sft_step(state, obs_b, act_b):
        def loss_fn(p):
            model     = RoboticCWRRT()
            carry0    = model.init_carry(obs_b.shape[0])
            t_idx0    = jnp.zeros((obs_b.shape[0],), dtype=jnp.int32)
            logits, _ = model.apply({"params": p}, obs_b, carry0, t_idx0, freeze_thought=False)
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
            b     = idx[i:i + SFT_BATCH_SIZE]
            obs_b = jnp.asarray(obs_all[b], dtype=jnp.float32)
            act_b = jnp.asarray(act_all[b], dtype=jnp.int32)
            trainer.state, loss = sft_step(trainer.state, obs_b, act_b)
            losses.append(loss)
        print(f"[SFT] Epoch {ep} | Loss: {float(jnp.mean(jnp.array(losses))):.6f}")


# MAIN EXECUTION

def main():
    np.random.seed(SEED)
    random.seed(SEED)

    trainer = Trainer(seed=SEED)

    # Warm Start
    if SFT_ENABLE and (not os.path.exists(SFT_FLAG)):
        sft_env = make_single_env(0, render_mode=None)()
        obs_all, act_all = generate_sft_dataset(sft_env, episodes=SFT_EPISODES)
        sft_env.close()

        print(f"[SFT] Steps: {len(obs_all)} | Batch: {SFT_BATCH_SIZE}")
        sft_train(trainer, obs_all, act_all)

        with open(SFT_FLAG, "w") as f:
            f.write("done")

    trainer.set_lr(LR_RL)

    # Initialize Async Environment
    vec_env = create_vec_env(GROUP_SIZE, render_mode=None)

    frozen_ref = trainer.state.params
    kl_beta    = float(KL_BETA)

    print("\n" + "=" * 78)
    print("  INTERNAL RL (AsyncVectorEnv + Batched JAX) - PPO+GRPO")
    print(f"  Env: {ENV_ID} | Group: {GROUP_SIZE} | Macro: {MACRO_STEP} | MaxT: {MAX_EPISODE_STEPS}")
    print("=" * 78 + "\n")

    # Training Loop
    for upd in range(1, UPDATES + 1):
        trajs, scores     = collect_group_rollouts_vec(vec_env, trainer)
        advs, mean_score  = compute_grpo_advantages(scores)

        loss, kl_v, ent_v = ppo_grpo_epochs(trainer, frozen_ref, trajs, advs, kl_beta)

        if kl_v > TARGET_KL * 1.5:
            kl_beta *= KL_ALPHA
        elif kl_v < TARGET_KL / 1.5:
            kl_beta /= KL_ALPHA

        if upd % 5 == 0 or upd == 1:
            print(
                f"[UPD {upd:4d}] "
                f"MeanScore: {mean_score:7.2f} | Best: {float(np.max(scores)):7.2f} | "
                f"Loss: {loss:8.4f} | KL: {kl_v:.4f} | Ent: {ent_v:.4f} | Beta: {kl_beta:.4f}"
            )

        if upd % 20 == 0:
            viz       = make_single_env(999, render_mode="human")()
            obs, info = viz.reset()
            carry     = trainer.model.init_carry(1)

            done = False
            t = 0

            while (not done) and (t < MAX_EPISODE_STEPS):
                # Visualize (Single Thread)
                obs_j = jnp.asarray(obs[None, :], dtype=jnp.float32)
                _, carry = step_think(trainer.state.params, obs_j, carry)

                for _k in range(MACRO_STEP):
                    if t >= MAX_EPISODE_STEPS: break

                    obs_j         = jnp.asarray(obs[None, :], dtype=jnp.float32)
                    logits, carry = step_act(trainer.state.params, obs_j, carry, _k) # time_idx байдлаар k-г дамжуулах

                    trainer.rng, subkey = jax.random.split(trainer.rng)
                    act_j, _            = sample_actions_and_oldlp(subkey, logits)
                    act                 = np.asarray(act_j[0], dtype=np.int32)

                    obs, _, term, trunc, info = viz.step(act)
                    t += 1

                    if term or trunc:
                        done = True
                        break
            viz.close()

    vec_env.close()


if __name__ == "__main__":
    main()