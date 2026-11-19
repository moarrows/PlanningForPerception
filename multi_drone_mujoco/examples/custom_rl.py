import time
import mujoco
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecTransposeImage,
    SubprocVecEnv,
)

from multi_drone_mujoco.envs.custom_aviary import CustomAviary
from multi_drone_mujoco.utils.scenario_loader import (
    load_scenarios,
    inject_obstacles,
)

SCENARIO_PATH = "scenario_data/train_10.npz"
EVAL_SCENARIO_PATH = "scenario_data/eval_10.npz"

from stable_baselines3 import PPO
import torch

class DiagnosticPPO(PPO):
    def train(self):
        # Check the first minibatch before any optimizer updates
        rollout_data = next(
            self.rollout_buffer.get(self.batch_size)
        )

        with torch.no_grad():
            values, log_prob, entropy = (
                self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions
                )
            )

            log_ratio = log_prob - rollout_data.old_log_prob
            ratio = torch.exp(log_ratio)

            kl = ((ratio - 1) - log_ratio).mean()

            print(f"\nPRE-UPDATE KL: {kl.item():.6f}\n")

        super().train()

class WaypointCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env=None,
        start_radius=0.4,
        max_radius=1.5,
        step=0.2,
        window=20,
        threshold=1.5,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.radius = start_radius
        self.max_radius = max_radius
        self.step = step
        self.window = window
        self.threshold = threshold
        self.recent = []

    def _on_training_start(self):
        self.training_env.env_method("set_curriculum_radius", self.radius)
        if self.eval_env is not None:
            self.eval_env.env_method("set_curriculum_radius", self.radius)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            reached = info["waypoints_reached"][0]
            self.logger.record_mean("custom/waypoints_reached", reached)
            self.recent.append(reached)

            dbg = info.get("dbg", {})
            for name, value in dbg.items():
                self.logger.record_mean(f"custom/{name}", value)

            if len(self.recent) >= self.window:
                mean_wp = np.mean(self.recent[-self.window :])
                if mean_wp >= self.threshold and self.radius < self.max_radius:
                    self.radius = min(self.radius + self.step, self.max_radius)
                    self.training_env.env_method("set_curriculum_radius", self.radius)
                    if self.eval_env is not None:
                        self.eval_env.env_method("set_curriculum_radius", self.radius)
                    self.recent.clear()  # re-measure at the new difficulty
                    if self.verbose:
                        print(f"Curriculum: radius -> {self.radius:.2f}")
        self.logger.record("custom/curriculum_radius", self.radius)
        return True


class WaypointRadiusCallback(BaseCallback):
    def __init__(
        self,
        eval_env=None,
        start_radius=0.25,
        min_radius=0.1,
        step=0.05,
        window=20,
        threshold=4.5,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.radius = start_radius
        self.min_radius = min_radius
        self.step = step
        self.window = window
        self.threshold = threshold
        self.recent = []

    def _on_training_start(self):
        self.training_env.env_method("set_waypoint_radius", self.radius)
        if self.eval_env is not None:
            self.eval_env.env_method("set_waypoint_radius", self.radius)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.recent.append(info["waypoints_reached"][0])

            if len(self.recent) >= self.window:
                mean_wp = np.mean(self.recent[-self.window :])
                if mean_wp >= self.threshold and self.radius > self.min_radius:
                    self.radius = max(self.radius - self.step, self.min_radius)
                    self.training_env.env_method("set_waypoint_radius", self.radius)
                    if self.eval_env is not None:
                        self.eval_env.env_method("set_waypoint_radius", self.radius)
                    self.recent.clear()
                    if self.verbose:
                        print(f"Hit radius -> {self.radius:.2f}")
        self.logger.record("custom/hit_radius", self.radius)
        return True

def load_model():
    n_envs = 8
    env = SubprocVecEnv(
        [make_env(SCENARIO_PATH, sequential=False) for _ in range(n_envs)]
    )
    env = VecNormalize.load("vecnormalize_V8.pkl", env)  # restores the running stats
    env.training = True  # keep updating stats while training
    env.norm_reward = True
    env = VecTransposeImage(env)

    eval_env = DummyVecEnv(
        [make_env(EVAL_SCENARIO_PATH, sequential=True)]
    )
    eval_env = VecNormalize.load("vecnormalize_V8.pkl", eval_env)
    eval_env.training = False  # freeze stats
    eval_env.norm_reward = False
    eval_env = VecTransposeImage(eval_env)

    model = PPO.load(
        "quadcopter_V8",
        env=env,
        device="cuda",
        custom_objects={"learning_rate": 1e-4, "lr_schedule": lambda _: 1e-4, "n_epochs": 5, "ent_coef": 0.01, "target_kl": 0.05}, 
    )

    return env, eval_env, model


def make_model():
    n_envs = 8  # no more than your CPU core count
    env = SubprocVecEnv(
        [make_env(SCENARIO_PATH, sequential=False) for _ in range(n_envs)]
    )
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_obs_keys=["kin", "waypoint_rel"],  # normalize only these
        norm_reward=True,
        clip_obs=10.0,
    )

    eval_env = DummyVecEnv(
        [make_env(EVAL_SCENARIO_PATH, sequential=True)]
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_obs_keys=["kin", "waypoint_rel"],
        norm_reward=False,
        training=False,  # don't update stats during eval
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=512,  # x 8 envs = 4096 per rollout
        batch_size=256,  # divides 4096
        learning_rate=1e-4,
        policy_kwargs=dict(log_std_init=-1.0),
        ent_coef=0.001,
        target_kl=0.05,
        tensorboard_log="./tb_logs",
        device="cuda",
    )

    return env, eval_env, model


def make_env(scenario_path=SCENARIO_PATH, sequential=False):
    def _init():
        scenarios = load_scenarios(scenario_path, sequential=sequential)
        e = CustomAviary(
            record=False,
            gui=False,
            scenario_dataset=scenarios,
        )
        inject_obstacles(e, scenarios.obstacle_count)
        return Monitor(e)

    return _init


def main():
    print("=" * 60)
    print("Training PPO on CustomAviary")
    print("=" * 60)

    env, eval_env, model = load_model()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    waypoint_curriculum_callback = WaypointCurriculumCallback(
        eval_env=eval_env, start_radius=3.0
    )
    waypoint_radius_callback = WaypointRadiusCallback(
        eval_env=eval_env, start_radius=0.10, window=50
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=5_000_000,
        callback=[
            eval_callback,
            waypoint_curriculum_callback,
            waypoint_radius_callback,
            checkpoint_callback,
        ],
        reset_num_timesteps=False,
    )

    print("Training finished. Evaluating final policy...")

    model.save("quadcopter_V9")
    env.venv.save("vecnormalize_V9.pkl")

    # Evaluate a few episodes
    obs = eval_env.reset()
    total_reward = 0.0
    episodes = 0
    while episodes < 5:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = eval_env.step(action)
        total_reward += float(reward[0])
        if dones[0]:
            print(f"Episode {episodes + 1} finished with reward: {total_reward:.2f}")
            total_reward = 0.0
            episodes += 1

    env.close()
    eval_env.close()


def view_env():
    print("=" * 60)
    print("Viewing CustomAviary Environment")
    print("=" * 60)

    scenarios = load_scenarios(EVAL_SCENARIO_PATH, sequential=True)
    env = CustomAviary(record=False, gui=False, scenario_dataset=scenarios)
    inject_obstacles(env, scenarios.obstacle_count)

    env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            viewer.sync()
            env.step(action)

            time.sleep(0.01)

    env.close()


def test_model():
    env, eval_env, model = load_model()
    base_env = eval_env.venv.venv.envs[0].unwrapped

    obs = eval_env.reset()
    total_reward = 0.0
    episodes = 0
    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        while episodes < 5:
            action, _states = model.predict(obs, deterministic=True)
            viewer.sync()
            obs, reward, dones, infos = eval_env.step(action)
            total_reward += float(reward[0])
            if dones[0]:
                total_reward = 0.0
                episodes += 1

    env.close()
    eval_env.close()


if __name__ == "__main__":
    # test_env()  # Test the environment for stability
    main()  # Train the PPO agent
    # view_env()  # View the environment with obstacles
    # test_model() # Test the trained model visually
