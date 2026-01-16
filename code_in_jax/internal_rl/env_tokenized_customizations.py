#
#  TOKENIZED ENV CUSTOMIZATION: FETCHPUSH-V4 (SEQUENTIAL SUBGOALS + TOKEN ACTIONS + DIRECTIONAL PUSHISH PRIOR)
#
#  ЗОРИЛГО:
#   Энэхүү код нь FetchPush-v4 робот орчныг токен үйлдэлтэй (MultiDiscrete)
#   болон дараалсан subgoal даалгавартай болгож тохируулсан туршилтын орчин юм.
#
#   1) Токен үйлдэл (TokenActionWrapper):
#      - Үйлдлийг тасралтгүй Box [-1,1] орон зайгаас дискрет токен [0..bins-1] руу хувиргана.
#      - Ингэснээр policy нь categorical action tokens хэлбэрээр ажиллана.
#
#   2) Observations-г хавтгайруулах (TransformerObservationWrapper):
#      - Dict observation-ыг нэг урт float32 вектор болгон хувиргана:
#          [observation (25)] + [desired_goal (3)] = 28
#      - NaN/Inf хамгаалалт + хэт том утга таслах (stability invariant) хийж өгнө.
#
#   3) Sequential Subgoals (SequentialGoalsWrapper):
#      - Random exploration үед шууд object->goal нь ихэнхдээ "dead" болох тул
#        learnable subgoal бүтэц хэрэглэнэ:
#          Subgoal-0      : gripper -> block
#          Subgoal-1..k-2 : gripper -> push_point (object-ийн ард, goal чиглэлтэй)
#          Final          : object(achieved_goal) -> goal
#      - Reward нь dist-ийн өөрчлөлт (progress) + subgoal бонус + final sparse reward.
#
#   4) Pushish Token Prior:
#      - Цэвэр random token нь block-ийг хөдөлгөх магадлал бага тул,
#        "түр зуурын" push-тэй төстэй хөдөлгөөний token дээжлэгч нэмнэ.
#      - Directional prior нь dbg_target чиглэл рүү XY хөдөлгөөнийг bias хийнэ.
#
#
#  ЕРӨНХИЙ УРСГАЛ
#   - Env specs хэвлэнэ
#   - epsilon-greedy болон macro-step rollouts ажиллуулж:
#     (a) dist өөрчлөгдөж байна уу?          (control/physics амьд уу)
#     (b) subgoal index ахиж байна уу?       (task learnable уу      )
#     (c) reward жижиг, тогтвортой байна уу? (numerical stability    )
#
#




import gymnasium as gym
import gymnasium_robotics  # registers Fetch* envs
import numpy as np


class TransformerObservationWrapper(gym.ObservationWrapper):
    """
    Dict observation -> flat float32 vector.
    Structure:
        [observation (25)] + [desired_goal (3)] = 28

    Invariant:
      - NaN/Inf утга орж ирвэл 0 болгоно
      - Хэт том утгыг clip хийж тогтвортой болгоно
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

    def observation(self, obs_dict):
        x = np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"]],
            axis=0,
        ).astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.clip_abs > 0:
            x = np.clip(x, -self.clip_abs, self.clip_abs).astype(np.float32)

        return x


class ActionTokenizer:
    """
    Continuous [-1,1] <-> Discrete tokens [0, bins-1]
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
    Token action space (MultiDiscrete) -> underlying env continuous Box action.

    Жич:
      - dtype-г int32 байлгаж, дараах pipeline (JAX) талд нэг мөр болгоно.
    """
    def __init__(self, env, bins=256):
        super().__init__(env)
        self.tokenizer    = ActionTokenizer(bins=bins)
        n                 = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete([bins] * n)

    def action(self, token_action):
        token_action = np.asarray(token_action, dtype=np.int32)
        token_action = np.clip(token_action, 0, self.tokenizer.bins - 1)

        a = self.tokenizer.decode(token_action).astype(np.float32)
        return a


class SequentialGoalsWrapper(gym.Wrapper):
    """
    FetchPush дээр learnable sequential subgoal task үүсгэх.

    Subgoal бүтэц:
      - goal_idx=0           : gripper               -> block distance
      - goal_idx=1..k-2      : gripper               -> push_point distance
      - goal_idx=k-1 (final) : object(achieved_goal) -> goal distance

    Reward:
      - Алхам бүрт dist-ийн өөрчлөлтөөр (prev_dist - dist) shaping өгнө (clip хийнэ).
      - Subgoal хүрэх үед нэмэлт бонус өгнө.
      - Final goal хүрэх үед reward=+1 (sparse).

    Жич:
      - Base env-ийн terminated-г ашиглахгүй (wrapper өөрөө termination удирдана).
    """
    def __init__(
        self,
        env,
        k                    = 3,
        goal_threshold0      = 0.10,
        goal_threshold1      = 0.10,
        goal_threshold_final = 0.12,
        sparse_final_reward  = True,
        goal_jitter          = 0.02,
        z_floor              = 0.42,
        z_ceil               = 0.80,
        xy_clip              = 0.12,
        shaping_enable       = True,
        shaping_scale        = 1.0,
        shaping_clip         = 0.05,
        subgoal_bonus        = 0.20,
        push_offset          = 0.06,
    ):
        super().__init__(env)
        self.k                    = int(k)
        self.goal_threshold0      = float(goal_threshold0     )
        self.goal_threshold1      = float(goal_threshold1     )
        self.goal_threshold_final = float(goal_threshold_final)
        self.sparse_final_reward  = bool (sparse_final_reward )

        self.goal_jitter          = float(goal_jitter         )
        self.z_floor              = float(z_floor             )
        self.z_ceil               = float(z_ceil              )
        self.xy_clip              = float(xy_clip             )

        self.shaping_enable       = bool (shaping_enable      )
        self.shaping_scale        = float(shaping_scale       )
        self.shaping_clip         = float(shaping_clip        )
        self.subgoal_bonus        = float(subgoal_bonus       )

        self.push_offset          = float(push_offset         )

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
            base = self._base_goal
            g[0] = np.clip(g[0], base[0] - self.xy_clip, base[0] + self.xy_clip)
            g[1] = np.clip(g[1], base[1] - self.xy_clip, base[1] + self.xy_clip)

        g[2] = float(np.clip(g[2], self.z_floor, self.z_ceil))
        g    = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return g

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        base            = np.asarray(obs["desired_goal"], dtype=np.float32)
        self._base_goal = self._safe_goal(base)

        goals = [None] * self.k

        goals[0] = self._base_goal.copy()
        for i in range(1, self.k - 1):
            off = np.random.uniform(-self.goal_jitter, self.goal_jitter, size=3).astype(np.float32)
            g   = self._safe_goal(self._base_goal + off)
            goals[i] = g

        goals[self.k - 1] = self._base_goal.copy()

        self._goals     = goals
        self._goal_idx  = 0
        self._prev_dist = None

        obs = self._set_current_goal_in_obs(obs)

        info = dict(info)
        info["seq_goal_index"] = self._goal_idx
        info["seq_goals"     ] = np.stack(self._goals, axis=0)
        info["seq_reached"   ] = False
        info["seq_dist"      ] = None
        info["thr0"          ] = self.goal_threshold0
        info["thr1"          ] = self.goal_threshold1
        info["thr_final"     ] = self.goal_threshold_final

        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        terminated = False

        obs_vec   = np.asarray(obs["observation"], dtype=np.float32)
        grip_pos  = obs_vec[0:3]
        block_pos = obs_vec[3:6]

        dbg_target     = None
        info_push_point = None

        if self._goal_idx == 0:
            target    = block_pos.copy()
            target[2] = max(target[2], self.z_floor)

            dist       = float(np.linalg.norm(grip_pos - target))
            thresh     = self.goal_threshold0
            dbg_target = target.copy()

        elif self._goal_idx < (self.k - 1):
            achieved   = np.asarray(obs["achieved_goal"], dtype=np.float32)
            final_goal = self._goals[self.k - 1]

            v  = (final_goal - achieved).astype(np.float32)
            vn = v / (np.linalg.norm(v) + 1e-6)

            push_point    = achieved - vn * self.push_offset
            push_point[2] = max(push_point[2], self.z_floor)

            dist            = float(np.linalg.norm(grip_pos - push_point))
            thresh          = self.goal_threshold1
            dbg_target      = push_point.copy()
            info_push_point = push_point.copy()

        else:
            achieved     = np.asarray(obs["achieved_goal"], dtype=np.float32)
            current_goal = self._goals[self._goal_idx]

            dist       = float(np.linalg.norm(achieved - current_goal))
            thresh     = self.goal_threshold_final
            dbg_target = current_goal.copy()

        reached = dist <= thresh

        info = dict(info)
        info["seq_goal_index"] = self._goal_idx
        info["seq_reached"   ] = reached
        info["seq_dist"      ] = dist
        info["seq_thresh"    ] = thresh

        info["dbg_grip_pos"  ] = grip_pos.copy()
        info["dbg_block_pos" ] = block_pos.copy()
        info["dbg_goal"      ] = self._goals[self._goal_idx].copy()
        info["dbg_achieved"  ] = np.asarray(obs["achieved_goal"], dtype=np.float32).copy()
        info["dbg_target"    ] = dbg_target.copy() if dbg_target is not None else None
        info["dbg_push_point"] = info_push_point.copy() if info_push_point is not None else None

        shaping_r = 0.0
        if self.shaping_enable:
            if self._prev_dist is None:
                shaping_r = 0.0
            else:
                shaping_r = (self._prev_dist - dist) * self.shaping_scale

            shaping_r = float(np.clip(shaping_r, -self.shaping_clip, self.shaping_clip))

        self._prev_dist = dist

        if reached:
            if self._goal_idx < self.k - 1:
                self._goal_idx += 1
                obs             = self._set_current_goal_in_obs(obs)

                info["seq_goal_index"] = self._goal_idx
                info["is_success"    ] = 0.0

                reward = shaping_r + self.subgoal_bonus

                self._prev_dist = None
                return obs, float(reward), False, truncated, info

            info["is_success"] = 1.0
            reward             = 1.0 if self.sparse_final_reward else float(base_reward)

            obs = self._set_current_goal_in_obs(obs)
            return obs, float(reward), True, truncated, info

        obs = self._set_current_goal_in_obs(obs)

        info["is_success"] = 0.0
        reward = shaping_r

        return obs, float(reward), False, truncated, info


def print_env_specs(env):
    print("\n" + "=" * 60)
    print(f"ENV: {env.unwrapped.spec.id if env.unwrapped.spec else 'Unknown'}")
    print("=" * 60)
    print(f"Action Space (tokenized): {env.action_space}")
    o, info = env.reset(seed=0)
    if isinstance(o, dict):
        print("Observation is dict (unexpected after wrapper).")
    else:
        print(f"Observation (flat) shape: {o.shape}, dtype={o.dtype}")
        print(f"Sample obs[:5]: {o[:5]}")
        print(f"Finite obs? {np.isfinite(o).all()}")
    print("=" * 60 + "\n")


def sample_pushish_tokens(env, bins=256):
    """
    Push-тэй төстэй token action дээжлэгч.

    Таамаглал (FetchPush-v4 action=4):
      - [dx, dy, dz, gripper]
      - dx/dy: дунджаас хазайлттай random
      - dz    : ихэвчлэн 0 орчим, бага зэрэг доош bias
      - gripper: ихэвчлэн хаалттай тал руу bias

    Жич:
      - Энэ нь зөвхөн diagnostic rollout-д зориулсан "prior".
    """
    nvec = env.action_space.nvec
    if int(nvec.shape[0]) != 4:
        return env.action_space.sample().astype(np.int32)

    mid = (nvec // 2).astype(np.int32)

    def clip_tok(x):
        return int(np.clip(x, 0, bins - 1))

    dx = clip_tok(np.random.normal(loc=mid[0], scale=48.0))
    dy = clip_tok(np.random.normal(loc=mid[1], scale=48.0))

    dz = clip_tok(np.random.normal(loc=mid[2] - 12.0, scale=18.0))

    gr = clip_tok(np.random.normal(loc=64.0, scale=24.0))

    return np.asarray([dx, dy, dz, gr], dtype=np.int32)



def sample_pushish_tokens_directional(env, info, bins=256):
    """
    Directional pushish prior:
      - dbg_target чиглэл рүү gripper-ийг зөөлөн "татах" токен гаргана.
      - idx>=1 үед dbg_push_point-ийг давуу ашиглаж, "contact" барих боломжийг өсгөнө.
      - Z-г ихэвчлэн 0 орчим / бага зэрэг доош байлгаж, ширээний ойролцоо ажиллуулна.
      - Gripper-г ихэвчлэн хаалттай тал руу bias хийнэ.
    """
    nvec = env.action_space.nvec
    if int(nvec.shape[0]) != 4:
        return env.action_space.sample().astype(np.int32)

    idx = info.get("seq_goal_index", -1)

    gp = info.get("dbg_grip_pos", None)
    if gp is None:
        return sample_pushish_tokens(env, bins=bins).astype(np.int32)

    if idx >= 1:
        tg = info.get("dbg_push_point", None)
        if tg is None:
            tg = info.get("dbg_target", None)
    else:
        tg = info.get("dbg_target", None)

    if tg is None:
        return sample_pushish_tokens(env, bins=bins).astype(np.int32)

    gp = np.asarray(gp, dtype=np.float32)
    tg = np.asarray(tg, dtype=np.float32)

    d  = (tg - gp).astype(np.float32)
    dn = float(np.linalg.norm(d[0:2]) + 1e-6)

    ux = float(d[0] / dn)
    uy = float(d[1] / dn)

    mid = (nvec // 2).astype(np.int32)

    def clip_tok(x):
        return int(np.clip(x, 0, bins - 1))

    amp_xy = float(np.random.uniform(18.0, 44.0))

    dx = clip_tok(mid[0] + ux * amp_xy + np.random.normal(0.0, 10.0))
    dy = clip_tok(mid[1] + uy * amp_xy + np.random.normal(0.0, 10.0))

    dz = clip_tok(np.random.normal(loc=mid[2] - 10.0, scale=12.0))

    gr = clip_tok(np.random.normal(loc=56.0, scale=20.0))

    return np.asarray([dx, dy, dz, gr], dtype=np.int32)




def run_rollout_eps_greedy(env, eps=0.2, seed=None, print_every=20, prior_prob=0.85, directional_prob=0.75):
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps        = 0

    while True:
        if np.random.rand() < eps:
            if np.random.rand() < float(prior_prob):
                if np.random.rand() < float(directional_prob):
                    action_tokens = sample_pushish_tokens_directional(env, info).astype(np.int32)
                else:
                    action_tokens = sample_pushish_tokens(env).astype(np.int32)
            else:
                action_tokens = env.action_space.sample().astype(np.int32)
        else:
            if np.random.rand() < float(directional_prob):
                action_tokens = sample_pushish_tokens_directional(env, info).astype(np.int32)
            else:
                action_tokens = sample_pushish_tokens(env).astype(np.int32)

        obs, reward, terminated, truncated, info = env.step(action_tokens)

        total_reward += float(reward)
        steps += 1

        if (print_every is not None) and (steps % int(print_every) == 0):
            ag = info.get("dbg_achieved", None)
            tg = info.get("dbg_target"  , None)
            print(
                f"[EPS] t={steps:3d} idx={info.get('seq_goal_index', -1)} "
                f"dist={info.get('seq_dist', None):.4f} thr={info.get('seq_thresh', None)} "
                f"r={float(reward):+.3f} "
                f"ach=({ag[0]:.3f},{ag[1]:.3f},{ag[2]:.3f}) "
                f"tgt=({tg[0]:.3f},{tg[1]:.3f},{tg[2]:.3f})"
            )

        if terminated or truncated:
            print(
                f"steps={steps} total_reward={total_reward:.3f} "
                f"is_success={info.get('is_success', 0.0)} "
                f"seq_goal_index={info.get('seq_goal_index', -1)} "
                f"final_dist={info.get('seq_dist', None)}"
            )
            return


def run_rollout_macro(env, macro_step_size=5, seed=None, print_every=20, prior_prob=0.90, directional_prob=0.85):
    """
    Macro-step exploration: нэг token-action сонгоод K алхам барина.
    K-г бага байлгавал (жишээ нь 5) gripper хэт хол "савлах" нь багасна.
    """
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps        = 0

    while True:
        if np.random.rand() < float(prior_prob):
            if np.random.rand() < float(directional_prob):
                action_tokens = sample_pushish_tokens_directional(env, info).astype(np.int32)
            else:
                action_tokens = sample_pushish_tokens(env).astype(np.int32)
        else:
            action_tokens = env.action_space.sample().astype(np.int32)

        for _ in range(int(macro_step_size)):
            obs, reward, terminated, truncated, info = env.step(action_tokens)

            total_reward += float(reward)
            steps += 1

            if (print_every is not None) and (steps % int(print_every) == 0):
                gp = info.get("dbg_grip_pos" , None)
                bp = info.get("dbg_block_pos", None)
                ag = info.get("dbg_achieved" , None)
                tg = info.get("dbg_target"   , None)
                print(
                    f"[MACRO] t={steps:3d} idx={info.get('seq_goal_index', -1)} "
                    f"dist={info.get('seq_dist', None):.4f} thr={info.get('seq_thresh', None)} "
                    f"r={float(reward):+.3f} "
                    f"grip=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                    f"block=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}) "
                    f"ach=({ag[0]:.3f},{ag[1]:.3f},{ag[2]:.3f}) "
                    f"tgt=({tg[0]:.3f},{tg[1]:.3f},{tg[2]:.3f})"
                )

            if terminated or truncated:
                print(
                    f"[MACRO] steps={steps} total_reward={total_reward:.3f} "
                    f"is_success={info.get('is_success', 0.0)} "
                    f"seq_goal_index={info.get('seq_goal_index', -1)} "
                    f"final_dist={info.get('seq_dist', None)}"
                )
                return


def run_summary(env, episodes=100, seed0=0, max_steps=200, eps=0.2, prior_prob=0.85, directional_prob=0.75):
    """
    Rollout summary runner:
      - success rate
      - idx ахилтын статистик (idx=1/2 хүрсэн эсэх)
      - idx=2 дээр achieved_goal (block) хөдөлсөн эсэх
      - дуусах үеийн idx болон dist
    """
    succ_count       = 0
    reach1_count     = 0
    reach2_count     = 0
    moved2_count     = 0

    steps_sum        = 0
    steps_succ_sum   = 0
    steps_to2_sum    = 0
    steps_to2_count  = 0

    final_idx_sum    = 0
    final_dist_sum   = 0.0

    moved_thr        = 0.02

    for ep in range(int(episodes)):
        obs, info = env.reset(seed=int(seed0) + ep)

        reached1 = False
        reached2 = False

        t_to2    = None

        ach2_init = None
        ach2_last = None

        total_r = 0.0

        for t in range(int(max_steps)):
            if np.random.rand() < float(eps):
                if np.random.rand() < float(prior_prob):
                    if np.random.rand() < float(directional_prob):
                        action_tokens = sample_pushish_tokens_directional(env, info).astype(np.int32)
                    else:
                        action_tokens = sample_pushish_tokens(env).astype(np.int32)
                else:
                    action_tokens = env.action_space.sample().astype(np.int32)
            else:
                if np.random.rand() < float(directional_prob):
                    action_tokens = sample_pushish_tokens_directional(env, info).astype(np.int32)
                else:
                    action_tokens = sample_pushish_tokens(env).astype(np.int32)

            obs, reward, terminated, truncated, info = env.step(action_tokens)
            total_r += float(reward)

            idx = int(info.get("seq_goal_index", -1))

            if (not reached1) and (idx >= 1):
                reached1 = True

            if (not reached2) and (idx >= 2):
                reached2 = True
                t_to2    = int(t + 1)

                ach2_init = info.get("dbg_achieved", None)
                if ach2_init is not None:
                    ach2_init = np.asarray(ach2_init, dtype=np.float32).copy()

            if idx >= 2:
                ach2_last = info.get("dbg_achieved", None)
                if ach2_last is not None:
                    ach2_last = np.asarray(ach2_last, dtype=np.float32).copy()

            if terminated or truncated:
                break

        steps_ep = int(t + 1)
        steps_sum += steps_ep

        is_succ = float(info.get("is_success", 0.0)) >= 0.5
        if is_succ:
            succ_count     += 1
            steps_succ_sum += steps_ep

        if reached1:
            reach1_count += 1
        if reached2:
            reach2_count += 1

        if t_to2 is not None:
            steps_to2_sum   += int(t_to2)
            steps_to2_count += 1

        if (ach2_init is not None) and (ach2_last is not None):
            dmove = float(np.linalg.norm(ach2_last[0:2] - ach2_init[0:2]))
            if dmove >= float(moved_thr):
                moved2_count += 1

        final_idx  = int(info.get("seq_goal_index", -1))
        final_dist = float(info.get("seq_dist", 0.0)) if info.get("seq_dist", None) is not None else 0.0

        final_idx_sum  += final_idx
        final_dist_sum += final_dist

    n = float(max(1, int(episodes)))

    succ_rate   = 100.0 * (float(succ_count  ) / n)
    reach1_rate = 100.0 * (float(reach1_count) / n)
    reach2_rate = 100.0 * (float(reach2_count) / n)
    moved2_rate = 100.0 * (float(moved2_count) / n)

    avg_steps   = float(steps_sum) / n
    avg_steps_s = float(steps_succ_sum) / float(max(1, succ_count))

    avg_to2     = float(steps_to2_sum) / float(max(1, steps_to2_count))

    avg_fidx    = float(final_idx_sum) / n
    avg_fdist   = float(final_dist_sum) / n

    print("\n" + "=" * 60)
    print("SUMMARY RUNNER")
    print("=" * 60)
    print(f"Episodes            : {int(episodes)}")
    print(f"MaxSteps            : {int(max_steps)}")
    print(f"eps                 : {float(eps):.3f}")
    print(f"prior_prob          : {float(prior_prob):.3f}")
    print(f"directional_prob    : {float(directional_prob):.3f}")
    print(f"moved_thr (XY)      : {float(moved_thr):.3f} m")
    print("-" * 60)
    print(f"Success rate        : {succ_rate:.1f}% ({succ_count}/{int(episodes)})")
    print(f"Reached idx>=1      : {reach1_rate:.1f}% ({reach1_count}/{int(episodes)})")
    print(f"Reached idx>=2      : {reach2_rate:.1f}% ({reach2_count}/{int(episodes)})")
    print(f"Moved at idx>=2     : {moved2_rate:.1f}% ({moved2_count}/{int(episodes)})")
    print("-" * 60)
    print(f"Avg steps (all)     : {avg_steps:.1f}")
    print(f"Avg steps (success) : {avg_steps_s:.1f}")
    print(f"Avg steps to idx=2  : {avg_to2:.1f}")
    print("-" * 60)
    print(f"Avg final idx       : {avg_fidx:.2f}")
    print(f"Avg final dist      : {avg_fdist:.4f}")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    raw_env = gym.make("FetchPush-v4", render_mode="human", max_episode_steps=200)

    raw_env = SequentialGoalsWrapper(
        raw_env,
        k                    = 3,
        goal_threshold0      = 0.10,
        goal_threshold1      = 0.10,
        goal_threshold_final = 0.12,
        sparse_final_reward  = True,
        goal_jitter          = 0.02,
        xy_clip              = 0.12,
        shaping_enable       = True,
        shaping_scale        = 1.0,
        shaping_clip         = 0.05,
        subgoal_bonus        = 0.20,
        push_offset          = 0.06,
    )

    env = TokenActionWrapper(raw_env, bins=256)
    env = TransformerObservationWrapper(env, clip_abs=10.0)

    print_env_specs(env)

    print("Running epsilon-greedy rollouts (directional pushish prior + random tokens with prob eps)...")
    for i in range(10):
        run_rollout_eps_greedy(env, eps=0.2, seed=i, print_every=20, prior_prob=0.85, directional_prob=0.75)

    print("Running macro-step rollouts (hold directional/prior token-action for K steps)...")
    for i in range(5):
        run_rollout_macro(env, macro_step_size=5, seed=100 + i, print_every=20, prior_prob=0.90, directional_prob=0.85)

    env.close()


    sum_raw = gym.make("FetchPush-v4", render_mode=None, max_episode_steps=200)
    sum_raw = SequentialGoalsWrapper(
        sum_raw, 
        k                    = 3, 
        goal_threshold0      = 0.10, 
        goal_threshold1      = 0.10, 
        goal_threshold_final = 0.12,
        sparse_final_reward  = True, 
        goal_jitter          = 0.02, 
        xy_clip              = 0.12,
        shaping_enable       = True, 
        shaping_scale        = 1.0, 
        shaping_clip         = 0.05,
        subgoal_bonus        = 0.20, 
        push_offset          = 0.06
        )
    sum_env = TokenActionWrapper(sum_raw, bins=256)
    sum_env = TransformerObservationWrapper(sum_env, clip_abs=10.0)

    print("Running summary runner (directional prior)...")
    run_summary(sum_env, episodes=100, seed0=1000, max_steps=200, eps=0.2, prior_prob=0.85, directional_prob=0.75)

    sum_env.close()
