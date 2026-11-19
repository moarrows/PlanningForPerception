"""Cycle through offline scenarios in a MuJoCo viewer."""

import argparse
import time

import mujoco

from multi_drone_mujoco.envs.custom_aviary import CustomAviary
from multi_drone_mujoco.utils.scenario_loader import (
    apply_scenario,
    inject_obstacles,
    load_scenarios,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to a scenario .npz file")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    scenarios = load_scenarios(args.dataset, sequential=True)
    env = CustomAviary(record=False, gui=False, scenario_dataset=scenarios)
    inject_obstacles(env, scenarios.obstacle_count)
    env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            apply_scenario(env, scenarios.next())
            mujoco.mj_forward(env.model, env.data)
            deadline = time.time() + args.seconds
            while viewer.is_running() and time.time() < deadline:
                viewer.sync()
                time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
