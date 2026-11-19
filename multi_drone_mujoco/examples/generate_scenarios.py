"""Generate offline training and evaluation waypoint/obstacle scenarios."""

import argparse
from pathlib import Path

from multi_drone_mujoco.utils.scenario_loader import generate_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scenario_data")
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for obstacle_count in (5, 10, 15, 20):
        scenarios = generate_dataset(
            args.train_count,
            obstacle_count,
            seed=1000 + obstacle_count,
        )
        save_dataset(
            str(output / f"train_{obstacle_count}.npz"),
            scenarios,
        )

    for obstacle_count in (5, 10, 15, 20):
        scenarios = generate_dataset(
            args.eval_count,
            obstacle_count,
            seed=1000 + obstacle_count,
        )
        save_dataset(
            str(output / f"eval_{obstacle_count}.npz"),
            scenarios,
        )
    print(f"Wrote scenario datasets to {output.resolve()}")


if __name__ == "__main__":
    main()
