"""Offline waypoint and obstacle scenario generation and loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True)
class Scenario:
    waypoints: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_radii: np.ndarray
    obstacle_half_heights: np.ndarray


class ScenarioDataset:
    """Loads a fixed scenario dataset and selects scenarios for environments."""

    def __init__(self, path: str, sequential: bool = True):
        data = np.load(path)
        self.waypoints = np.asarray(data["waypoints"], dtype=np.float32)
        self.obstacle_positions = np.asarray(data["obstacle_positions"], dtype=np.float32)
        self.obstacle_radii = np.asarray(data["obstacle_radii"], dtype=np.float32)
        self.obstacle_half_heights = np.asarray(
            data["obstacle_half_heights"], dtype=np.float32
        )
        if self.waypoints.ndim != 3 or self.waypoints.shape[1:] != (5, 3):
            raise ValueError("waypoints must have shape (scenarios, 5, 3)")
        if self.obstacle_positions.ndim != 3 or self.obstacle_positions.shape[2] != 3:
            raise ValueError("obstacle_positions must have shape (scenarios, obstacles, 3)")
        if self.obstacle_positions.shape[:2] != self.obstacle_radii.shape:
            raise ValueError("obstacle_radii shape does not match obstacle_positions")
        if self.obstacle_positions.shape[:2] != self.obstacle_half_heights.shape:
            raise ValueError(
                "obstacle_half_heights shape does not match obstacle_positions"
            )
        self.sequential = sequential
        self._next_index = 0

    @property
    def size(self) -> int:
        return self.waypoints.shape[0]

    @property
    def obstacle_count(self) -> int:
        return self.obstacle_positions.shape[1]

    def get(self, index: int) -> Scenario:
        index %= self.size
        return Scenario(
            self.waypoints[index].copy(),
            self.obstacle_positions[index].copy(),
            self.obstacle_radii[index].copy(),
            self.obstacle_half_heights[index].copy(),
        )

    def next(self, rng: Optional[np.random.Generator] = None) -> Scenario:
        if self.sequential:
            scenario = self.get(self._next_index)
            self._next_index = (self._next_index + 1) % self.size
            return scenario
        if rng is None:
            rng = np.random.default_rng()
        return self.get(int(rng.integers(self.size)))


def load_scenarios(path: str, sequential: bool = True) -> ScenarioDataset:
    return ScenarioDataset(path, sequential=sequential)


def inject_obstacles(env, count: int) -> None:
    """Add fixed-size obstacle bodies to an already-created aviary."""
    obstacle_xml = []
    colors = (
        "0.9 0.2 0.2 0.9",
        "0.2 0.8 0.2 0.9",
        "0.2 0.3 0.9 0.9",
        "0.9 0.6 0.1 0.9",
        "0.7 0.2 0.9 0.9",
        "0.1 0.8 0.8 0.9",
    )
    for i in range(count):
        radius = 0.05
        half_height = 0.6
        obstacle_xml.append(
            f"""
        <body name="obs_{i}" pos="0 0 {half_height:.3f}">
          <geom type="cylinder" size="{radius:.3f} {half_height:.3f}"
                rgba="{colors[i % len(colors)]}"/>
          <inertial pos="0 0 0" mass="0.01"
                    diaginertia="0.0001 0.0001 0.0001"/>
        </body>
        """
        )

    xml = env.xml_str.replace("</worldbody>", "".join(obstacle_xml) + "</worldbody>")
    env.model = mujoco.MjModel.from_xml_string(xml)
    env.data = mujoco.MjData(env.model)
    env._renderer = None
    env._obstacle_body_ids = tuple(
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
        for i in range(count)
    )
    mujoco.mj_forward(env.model, env.data)
    env._updateAndStoreKinematicInformation()


def apply_scenario(env, scenario: Scenario) -> None:
    if len(env._obstacle_body_ids) != len(scenario.obstacle_positions):
        raise ValueError("Scenario obstacle count does not match the environment")
    env.WAYPOINTS = scenario.waypoints.copy()
    for i, body_id in enumerate(env._obstacle_body_ids):
        env.model.body_pos[body_id] = scenario.obstacle_positions[i]
        geom_ids = np.flatnonzero(env.model.geom_bodyid == body_id)
        for geom_id in geom_ids:
            if env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                env.model.geom_size[geom_id, 0] = scenario.obstacle_radii[i]
                env.model.geom_size[geom_id, 1] = scenario.obstacle_half_heights[i]
    mujoco.mj_forward(env.model, env.data)


def _clear(candidate, radius, points, placed, margin, drone_radius):
    if any(
        np.linalg.norm(candidate[:2] - point[:2])
        <= margin + radius + drone_radius
        for point in points
    ):
        return False

    return all(
        np.linalg.norm(candidate[:2] - point[:2])
        > margin + radius + other_radius
        for point, other_radius in placed
    )


def generate_scenario(
    obstacle_count: int,
    seed: int,
    initial_position: Optional[np.ndarray] = None,
) -> Scenario:
    rng = np.random.default_rng(seed)
    if initial_position is None:
        initial_position = np.array([0.0, 0.0, 0.5])
    drone_radius = float(np.hypot(0.06, 0.015))
    margin = 0.25
    waypoints = []
    previous = initial_position
    for _ in range(5):
        for _ in range(500):
            candidate = np.array(
                [rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(0.1, 0.5)]
            )
            if np.linalg.norm(candidate - previous) > 0.9:
                break
        else:
            raise RuntimeError("Could not generate a waypoint route")
        waypoints.append(candidate)
        previous = candidate
    waypoints = np.asarray(waypoints, dtype=np.float32)
    route = [initial_position] + list(waypoints)
    segments = list(zip(route[:-1], route[1:]))
    radii = rng.uniform(0.035, 0.065, obstacle_count).astype(np.float32)
    half_heights = rng.uniform(0.75, 1.5, obstacle_count).astype(np.float32)
    positions = [None] * obstacle_count
    keep_away = [initial_position] + list(waypoints)

    gate_count = (round(obstacle_count * 0.25) // 2) * 2
    blocker_count = round(obstacle_count * 0.25)
    route_count = round(obstacle_count * 0.25)

    def try_place(candidate, index):
        candidate = candidate.copy()
        candidate[2] = float(half_heights[index])
        placed = [position for position in positions if position is not None]

        if not _clear(
            candidate,
            float(radii[index]),
            keep_away,
            placed,
            margin,
            drone_radius,
        ):
            return False

        positions[index] = (candidate, float(radii[index]))
        return True

    # Place centered blockers first so gates cannot consume the route corridor.
    for index in range(blocker_count):
        placed = False
        candidates = []
        for segment_index in rng.permutation(len(segments)):
            start, end = segments[segment_index]
            direction = end[:2] - start[:2]
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            normal = np.array([-direction[1], direction[0]]) / length
            endpoint_clearance = margin + float(radii[index]) + drone_radius
            t_min = min(0.45, endpoint_clearance / length)
            if t_min >= 0.5:
                continue

            for t in rng.uniform(t_min, 1.0 - t_min, 12):
                center = start + t * (end - start)
                for offset in (-0.12, -0.06, 0.0, 0.06, 0.12):
                    candidate = center.copy()
                    candidate[:2] += normal * offset
                    candidates.append(candidate)

        rng.shuffle(candidates)
        for candidate in candidates:
            if try_place(candidate, index):
                placed = True
                break

        if not placed:
            # Keep the obstacle route-relevant if the exact route corridor is
            # crowded, but allow a wider near-route fallback.
            for _ in range(5000):
                segment_index = int(rng.integers(len(segments)))
                start, end = segments[segment_index]
                direction = end[:2] - start[:2]
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                normal = np.array([-direction[1], direction[0]]) / length
                candidate = start + rng.uniform(0.2, 0.8) * (end - start)
                candidate = candidate.copy()
                candidate[:2] += normal * rng.uniform(-0.35, 0.35)
                if try_place(candidate, index):
                    placed = True
                    break

        if not placed:
            raise RuntimeError(f"Could not place route blocker {index}")

    next_index = blocker_count

    # Place gate pairs transactionally.
    for pair_start in range(0, gate_count, 2):
        first_index = next_index + pair_start
        second_index = first_index + 1
        placed = False

        for _ in range(200000):
            start, end = segments[int(rng.integers(len(segments)))]
            direction = end[:2] - start[:2]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            normal = np.array([-direction[1], direction[0]]) / norm
            t = rng.uniform(0.25, 0.75)
            center = start + t * (end - start)

            opening = 2.0 * drone_radius + 0.12
            separation = (
                float(radii[first_index])
                + float(radii[second_index])
                + margin
                + opening
            )

            first = center.copy()
            second = center.copy()
            first[:2] -= normal * separation / 2.0
            second[:2] += normal * separation / 2.0
            first[2] = float(half_heights[first_index])
            second[2] = float(half_heights[second_index])

            if not _clear(
                first,
                float(radii[first_index]),
                keep_away,
                [position for position in positions if position is not None],
                margin,
                drone_radius,
            ):
                continue

            if not _clear(
                second,
                float(radii[second_index]),
                keep_away,
                [
                    position for position in positions if position is not None
                ]
                + [(first, float(radii[first_index]))],
                margin,
                drone_radius,
            ):
                continue

            positions[first_index] = (first, float(radii[first_index]))
            positions[second_index] = (second, float(radii[second_index]))
            placed = True
            break

        if not placed:
            raise RuntimeError(f"Could not place gate pair {pair_start // 2}")

    next_index += gate_count

    # Side obstacles encourage steering around the route.
    for route_index, index in enumerate(
        range(next_index, next_index + route_count)
    ):
        placed = False

        for _ in range(200000):
            start, end = segments[int(rng.integers(len(segments)))]
            direction = end[:2] - start[:2]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            normal = np.array([-direction[1], direction[0]]) / norm
            candidate = start + rng.uniform(0.2, 0.8) * (end - start)
            candidate = candidate.copy()

            side = 1.0 if route_index % 2 == 0 else -1.0
            offset = margin + float(radii[index]) + drone_radius
            offset += rng.uniform(0.1, 0.35)
            candidate[:2] += side * normal * offset

            if try_place(candidate, index):
                placed = True
                break

        if not placed:
            raise RuntimeError(f"Could not place route obstacle {index}")

    next_index += route_count

    # Remaining obstacles are ordinary random placements.
    for index in range(next_index, obstacle_count):
        placed = False

        for _ in range(200000):
            candidate = np.array(
                [
                    rng.uniform(-4.0, 4.0),
                    rng.uniform(-4.0, 4.0),
                    float(half_heights[index]),
                ]
            )

            if try_place(candidate, index):
                placed = True
                break

        if not placed:
            raise RuntimeError(f"Could not place random obstacle {index}")

    return Scenario(
        waypoints=waypoints,
        obstacle_positions=np.asarray(
            [positions[index][0] for index in range(obstacle_count)],
            dtype=np.float32,
        ),
        obstacle_radii=radii,
        obstacle_half_heights=half_heights,
    )


def generate_dataset(
    count: int,
    obstacle_count: int,
    seed: int,
) -> Tuple[Scenario, ...]:
    scenarios = []
    for index in range(count):
        scenarios.append(
            generate_scenario(
                obstacle_count=obstacle_count,
                seed=seed + index,
            )
        )
    return tuple(scenarios)


def save_dataset(path: str, scenarios: Tuple[Scenario, ...]) -> None:
    np.savez_compressed(
        path,
        waypoints=np.stack([scenario.waypoints for scenario in scenarios]),
        obstacle_positions=np.stack([scenario.obstacle_positions for scenario in scenarios]),
        obstacle_radii=np.stack([scenario.obstacle_radii for scenario in scenarios]),
        obstacle_half_heights=np.stack(
            [scenario.obstacle_half_heights for scenario in scenarios]
        ),
    )
