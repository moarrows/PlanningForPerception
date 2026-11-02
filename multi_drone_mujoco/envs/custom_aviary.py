"""Fly-Through Aviary: navigate through waypoints/gates.

Task: fly through a sequence of waypoints as quickly as possible.
"""

import numpy as np
import mujoco

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)
from gymnasium import spaces


class CustomAviary(BaseAviary):
    """Fly through waypoints task."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        waypoints=None,
        waypoint_radius: float = 0.25,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 30
        self.WAYPOINT_RADIUS = waypoint_radius
        self._dbg = getattr(
            self, "_dbg", dict(prog=0, time=0, angv=0, bonus=0, crash=0)
        )

        if waypoints is None:
            self.WAYPOINTS = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.5],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.5],
                ]
            )
        else:
            self.WAYPOINTS = np.array(waypoints)

        self.current_waypoint_idx = np.zeros(
            num_drones if num_drones > 1 else 1, dtype=int
        )

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]])

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs_type=ObservationType.KIN_RGB,
            act_type=ActionType.VEL,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
            vision_attributes=True,
            gimbal_control=True,
        )

        self._drone_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}"): i
            for i in range(self.NUM_DRONES)
        }

        self._prev_dist = None
        self.curriculum_radius = 0.4
        self._last_contact_step = -10_000

    def set_curriculum_radius(self, r):
        self.curriculum_radius = r

    def set_waypoint_radius(self, r):
        self.WAYPOINT_RADIUS = r

    def _randomize_obstacles(
        self, n_obs=5, xy_range=1.5, z_range=(0.3, 1.8), clearance=0.4, tries=50
    ):
        keep_away = [self.INIT_XYZS[0]] + list(self.WAYPOINTS)
        placed = []
        for i in range(n_obs):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}"
            )
            if body_id < 0:
                continue
            for _ in range(tries):
                cand = np.array(
                    [
                        self._rng.uniform(-xy_range, xy_range),
                        self._rng.uniform(-xy_range, xy_range),
                        self._rng.uniform(*z_range),
                    ]
                )
                ok = all(
                    np.linalg.norm(cand - p) > clearance for p in keep_away + placed
                )
                if ok:
                    break
            placed.append(cand)
            self.model.body_pos[body_id] = cand

        # per-episode motion so timing can't be memorized from the clock
        self._obs_phase = self._rng.uniform(0, 2 * np.pi)
        self._obs_amp = self._rng.uniform(0.2, 0.5)

    def reset(self, seed=None, options=None):
        self._filtered_action = None
        self._prev_dist = None
        self.current_waypoint_idx[:] = 0
        self._dbg = dict(prog=0.0, time=0.0, bonus=0.0, crash=0.0)
        self._last_contact_step = -10_000
        R = self.curriculum_radius  # start 0.4, grow to 1.5
        if seed is not None or not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(seed)
        min_gap = 2 * self.WAYPOINT_RADIUS + 0.2
        wps = []  # collects the 5 waypoints
        prev = self.INIT_XYZS[0].copy()  # start_pos: where the drone begins

        for k in range(5):
            for _ in range(20):  # up to 20 tries to find a far-enough point
                cand = np.array(
                    [
                        self._rng.uniform(-R, R),
                        self._rng.uniform(-R, R),
                        self._rng.uniform(0.3, 0.3 + R),
                    ]
                )
                if np.linalg.norm(cand - prev) > min_gap:
                    break
            wps.append(cand)
            prev = cand

        self.WAYPOINTS = np.array(wps)
        self._randomize_obstacles()
        return super().reset(seed=seed, options=options)

    def _obstacleCollisions(self):
        """Returns array of contact counts with obstacles, one per drone."""
        hits = np.zeros(self.NUM_DRONES)
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            # rootid maps child bodies (e.g. gimbal) back to the drone body
            r1 = self.model.body_rootid[b1]
            r2 = self.model.body_rootid[b2]
            for drone_root, other in ((r1, r2), (r2, r1)):
                if (
                    drone_root in self._drone_body_ids
                    and other in self._obstacle_body_ids
                ):
                    hits[self._drone_body_ids[drone_root]] += 1
        return hits

    def _computeReward(self):
        total = 0.0
        collisions = self._obstacleCollisions()

        for i in range(self.NUM_DRONES):
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            dist = np.linalg.norm(self.pos[i] - wp)
            total -= min(collisions[i], 1) * 10.0

            # Check waypoint reached
            if dist < self.WAYPOINT_RADIUS and self.current_waypoint_idx[i] < len(
                self.WAYPOINTS
            ):
                self.current_waypoint_idx[i] += 1
                total += 100.0  # Big bonus for reaching waypoint
                self._dbg["bonus"] += 100
                self._prev_dist = None
                continue

            if self._prev_dist is None:
                self._prev_dist = dist
            total += 5.0 * (self._prev_dist - dist)
            self._dbg["prog"] += 5.0 * (self._prev_dist - dist)
            self._prev_dist = dist

            total -= 0.005
            self._dbg["time"] -= 0.005

        if collisions.sum() > 0:
            self._last_contact_step = self.step_counter

        if self._computeTerminated():
            total -= 100
            self._dbg["crash"] -= 100

            print(self._term_reason)

        return float(total)

    def _computeTerminated(self):
        self._term_reason = None
        far_limit = 3.0 * self.curriculum_radius + 1.5  # 2.7 at R=0.4, 6.0 at R=1.5
        for i in range(self.NUM_DRONES):
            wp = self.WAYPOINTS[
                min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            ]
            if self.pos[i, 2] < 0.03:
                self._term_reason = "floor"
                return True
            if abs(self.rpy[i, 0]) > np.pi / 2 or abs(self.rpy[i, 1]) > np.pi / 2:
                recent = (
                    self.step_counter - self._last_contact_step
                ) < self.SIM_FREQ  # ~1 s
                self._term_reason = "tilt_after_contact" if recent else "tilt"
                return True
            if np.linalg.norm(self.pos[i] - wp) > far_limit:
                self._term_reason = "far"
                return True
        return False

    def _computeTruncated(self):
        # Done when all waypoints reached or time limit
        all_done = all(idx >= len(self.WAYPOINTS) for idx in self.current_waypoint_idx)
        time_up = self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC
        return all_done or time_up

    def _computeInfo(self):
        return {
            "waypoints_reached": [int(idx) for idx in self.current_waypoint_idx],
            "total_waypoints": len(self.WAYPOINTS),
        }

    def _observationSpace(self):
        obs = spaces.Dict(
            {
                "kin": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(20 * self.NUM_DRONES,),
                    dtype=np.float32,
                ),
                "rgb": spaces.Box(
                    low=0,
                    high=255,
                    shape=(48, 64, 3),
                    dtype=np.uint8,
                ),
                "waypoint_rel": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )

        if self.GIMBAL_CONTROL:
            obs.spaces["gimbal_rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(48, 64, 3),
                dtype=np.uint8,
            )

        return obs

    def _computeObs(self):
        kin = np.hstack([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        kin[16:20] = (kin[16:20] / self.HOVER_RPM - 1.0) / 0.05

        images = [self._getDroneImages(i) for i in range(self.NUM_DRONES)]
        imgs = images[0][0]  # was np.stack(...) -> now (48, 64, 3)

        n = len(self.WAYPOINTS)
        rel = (
            self.WAYPOINTS[min(self.current_waypoint_idx[0], n - 1)] - self.pos[0]
        ).astype(np.float32)  # shape (3,)

        obs = {
            "kin": kin.astype(np.float32),
            "rgb": np.ascontiguousarray(imgs, dtype=np.uint8),
            "waypoint_rel": rel,
        }

        if self.GIMBAL_CONTROL:
            obs["gimbal_rgb"] = np.zeros((48, 64, 3), dtype=np.uint8)

        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action[..., 3:6] = 0
        z = self.pos[0, 2]
        if z < 0.15:
            action[..., 2] = max(action[..., 2], 0.0)  # no descending near the floor

        # t = self.data.time
        # for i, actuator_id in enumerate(self.obstacle_actuator_ids):
        #     self.data.ctrl[actuator_id] = 0.5 * np.sin(2 * np.pi * 0.2 * t)

        return super().step(action)
