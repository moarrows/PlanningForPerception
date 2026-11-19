import copy
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecNormalize,
    VecTransposeImage,
)
from multi_drone_mujoco.envs.custom_aviary import CustomAviary
from multi_drone_mujoco.utils.custom_feature_extractor import DroneCombinedExtractor

# ---- filenames: edit these ----
OLD_MODEL = "quadcopter_V7"
OLD_STATS = "vecnormalize_V7.pkl"
NEW_MODEL = "quadcopter_V7_new"
NEW_STATS = "vecnormalize_V7_new.pkl"

N_ENVS = 4
N_STEPS = 1024  # N_ENVS * N_STEPS = 4096 per rollout
ROUTE_RADIUS = 1.5  # last curriculum radius
HIT_RADIUS = 0.25  # current hit radius


def make_env():
    def _init():
        return Monitor(CustomAviary(record=False, gui=False))

    return _init


def key_dims(policy):
    """Output width of each key's branch, in the order the extractor concatenates them."""
    dims = {}
    for key, ext in policy.features_extractor.extractors.items():
        if hasattr(ext, "features_dim"):  # CNN branch
            dims[key] = ext.features_dim
        else:  # flatten branch
            dims[key] = int(np.prod(policy.observation_space.spaces[key].shape))
    return dims


def offsets(dims):
    out, o = {}, 0
    for k, d in dims.items():
        out[k] = (o, o + d)
        o += d
    return out


def build_vec(n, training):
    venv = (
        SubprocVecEnv([make_env() for _ in range(n)])
        if training
        else DummyVecEnv([make_env()])
    )
    venv.env_method("set_curriculum_radius", ROUTE_RADIUS)
    venv.env_method("set_waypoint_radius", HIT_RADIUS)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_obs_keys=[
            key
            for key, space in venv.observation_space.spaces.items()
            if space.dtype == np.float32 and len(space.shape) == 1
        ],
        norm_reward=training,
        training=training,
        clip_obs=10.0,
    )
    return venv


def describe_observations(env):
    obs = env.reset()
    for key, value in obs.items():
        print(
            f"{key}: shape={value.shape}, dtype={value.dtype}, "
            f"range=[{value.min():.4g}, {value.max():.4g}], "
            f"mean={value.mean():.4g}, std={value.std():.4g}, "
            f"finite={np.isfinite(value).all()}"
        )
    if "depth" not in obs:
        raise RuntimeError("The conversion environment does not expose depth")
    if not np.isfinite(obs["depth"]).all():
        raise RuntimeError("Depth contains non-finite values")
    if obs["depth"].min() < 0.0 or obs["depth"].max() > 1.0:
        raise RuntimeError("Depth is outside the declared [0, 1] range")
    return obs


def main():
    # 1. old model and old normalization stats
    old = PPO.load(OLD_MODEL, device="cuda")
    with open(OLD_STATS, "rb") as f:
        old_vn = pickle.load(f)

    # 2. new camera envs, with the old stats copied in
    train_norm = build_vec(N_ENVS, training=True)
    eval_norm = build_vec(1, training=False)
    norm_keys = [
        key
        for key, space in train_norm.observation_space.spaces.items()
        if key in old_vn.obs_rms
    ]
    for vn in (train_norm, eval_norm):
        for k in norm_keys:
            vn.obs_rms[k] = copy.deepcopy(old_vn.obs_rms[k])
        vn.clip_obs = old_vn.clip_obs
    train_norm.ret_rms = copy.deepcopy(old_vn.ret_rms)

    env = VecTransposeImage(train_norm)
    eval_env = VecTransposeImage(eval_norm)

    new_policy_kwargs = {
        **old.policy_kwargs,
        "features_extractor_class": DroneCombinedExtractor,
    }

    new = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=new_policy_kwargs,
        n_steps=N_STEPS,
        batch_size=256,
        learning_rate=1e-4,
        n_epochs=old.n_epochs,
        clip_range=old.clip_range,
        clip_range_vf=old.clip_range_vf,
        ent_coef=old.ent_coef,
        target_kl=old.target_kl,
        gamma=old.gamma,
        gae_lambda=old.gae_lambda,
        tensorboard_log="./tb_logs",
        device="cuda",
    )

    # 4. work out column layout from the real extractor order
    old_dims, new_dims = key_dims(old.policy), key_dims(new.policy)
    old_off, new_off = offsets(old_dims), offsets(new_dims)
    print("old branches:", old_dims)
    print("new branches:", new_dims)
    assert set(old_dims) <= set(new_dims), "old obs keys missing from the new env"
    for k in old_dims:
        if old_dims[k] != new_dims[k]:
            print(
                f"Reinitializing branch mapping for {k}: "
                f"{old_dims[k]} -> {new_dims[k]}"
            )

    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()

    # 5. copy every tensor whose shape still matches
    copied = 0
    for name, tensor in old_sd.items():
        if name in new_sd and new_sd[name].shape == tensor.shape:
            new_sd[name] = tensor.clone()
            copied += 1
    print("copied same-shape tensors:", copied)

    # 6. first MLP layers: place old columns by key, new image columns start at zero
    for name in [
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    ]:
        w = torch.zeros_like(new_sd[name])

        for k in old_dims:

            # Cannot directly transfer flattened depth features
            # into learned CNN features.
            if old_dims[k] != new_dims[k]:
                continue

            a, b = old_off[k]
            c, d = new_off[k]

            w[:, c:d] = old_sd[name][:, a:b]

        new_sd[name] = w
    new.policy.load_state_dict(new_sd)

    # 7. Check the added input and verify it initially has no policy influence.
    obs = describe_observations(env)
    old_obs = {k: obs[k] for k in old_dims}
    a_old, _ = old.predict(old_obs, deterministic=True)
    a_new, _ = new.predict(obs, deterministic=True)
    diff = float(np.abs(a_old - a_new).max())
    print(f"max action difference old vs new: {diff:.2e}  (should be ~1e-6)")
    assert diff < 1e-3, "transfer failed: check the column layout printed above"

    # 8. quick behaviour check with the camera env
    mean_r, std_r = evaluate_policy(
        new, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(
        f"camera-env eval: {mean_r:.1f} +/- {std_r:.1f}  (old state-only policy scored ~500)"
    )

    for key, extractor in new.policy.features_extractor.extractors.items():
        print(key, type(extractor).__name__,
            getattr(extractor, "features_dim", "N/A"))

    # 9. save
    new.save(NEW_MODEL)
    train_norm.save(NEW_STATS)
    print("saved", NEW_MODEL, "and", NEW_STATS)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
