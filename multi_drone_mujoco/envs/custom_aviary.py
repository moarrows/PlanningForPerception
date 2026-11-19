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
from multi_drone_mujoco.utils.scenario_loader import ScenarioDataset, apply_scenario


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
        scenario_dataset: ScenarioDataset = None,
    ):
        self.EPISODE_LEN_SEC = 60
        self.WAYPOINT_RADIUS = waypoint_radius
        self.scenario_dataset = scenario_dataset
        self._dbg = getattr(
            self, "_dbg", dict(prog=0.0, time=0.0, bonus=0.0, crash=0.0, smooth=0.0, obstacleCollision=0.0, obstacleRadius=0.0)
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

        self.SAFE = 0.175

        self._prev_dist = None
        self.curriculum_radius = 1.5
        self._last_contact_step = -10_000
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._action_delta = 0.0
        self._obstacle_body_ids = ()

    def set_curriculum_radius(self, r):
        self.curriculum_radius = r

    def set_waypoint_radius(self, r):
        self.WAYPOINT_RADIUS = r

    def reset(self, seed=None, options=None):
        if seed is not None or not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(seed)

        self._filtered_action = None
        self._prev_dist = None
        self.current_waypoint_idx[:] = 0
        self._dbg = dict(prog=0.0, time=0.0, bonus=0.0, crash=0.0, smooth=0.0, obstacleCollision=0.0, obstacleRadius=0.0)
        self._last_contact_step = -10_000
        if self.scenario_dataset is not None:
            scenario = self.scenario_dataset.next(self._rng)
            apply_scenario(self, scenario)
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

    def _nearest_obstacle_dist(self, i=0):
        return self._nearest_obstacle_dist_from_point(self.pos[i])

    def _nearest_obstacle_dist_given_point(self, point):
        return self._nearest_obstacle_dist_from_point(point)

    def _nearest_obstacle_dist_from_point(self, point):
        """Return conservative clearance from a point to the obstacle bodies."""
        ids = sorted(b for b in self._obstacle_body_ids if b >= 0)
        if not ids:
            return 10.0

        point = np.asarray(point, dtype=np.float64)
        obstacle_positions = self.data.xpos[ids]
        obstacle_radii = []

        for body_id in ids:
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            obstacle_radii.append(
                max(
                    float(self.model.geom_size[geom_id, 0])
                    for geom_id in geom_ids
                    if self.model.geom_type[geom_id]
                    == mujoco.mjtGeom.mjGEOM_CYLINDER
                )
            )

        drone_radius = float(self.COLLISION_R)

        center_distances = np.linalg.norm(
            obstacle_positions[:, :2] - point[:2],
            axis=1,
        )

        clearances = center_distances - np.asarray(obstacle_radii) - drone_radius
        return float(clearances.min())

    def _computeOrientationReward(self, i):
        vxy = self.vel[i][:2]
        speed = np.linalg.norm(vxy)
        if speed < 0.2:
            return 0.0

        velocity_direction = vxy / speed
        forward_direction = np.array(
            [np.cos(self.rpy[i, 2]), np.sin(self.rpy[i, 2])]
        )
        alignment = float(np.clip(np.dot(forward_direction, velocity_direction), -1.0, 1.0))

        # The body-mounted camera looks along the drone's +X axis. Reward
        # horizontal motion in that same direction and penalize sideways or
        # backward flight.
        return -0.03 * min(speed, 1.0) * (1.0 - alignment)

    def _computeReward(self):
        total = 0.0
        collisions = self._obstacleCollisions()

        for i in range(self.NUM_DRONES):
            wp_idx = min(self.current_waypoint_idx[i], len(self.WAYPOINTS) - 1)
            wp = self.WAYPOINTS[wp_idx]
            dist = np.linalg.norm(self.pos[i] - wp)
            total -= min(collisions[i], 1) * 10.0
            self._dbg["obstacleCollision"] -= min(collisions[i], 1) * 50.0

            if self._prev_dist is None:
                self._prev_dist = dist
            total += 5.0 * (self._prev_dist - dist)
            self._dbg["prog"] += 5.0 * (self._prev_dist - dist)
            self._prev_dist = dist

            smooth = 0.005 * self._action_delta
            total -= smooth
            self._dbg["smooth"] -= smooth

            gap = self._nearest_obstacle_dist(i)
            if gap < self.SAFE:
                prox = 0.15 * ((self.SAFE - gap) / self.SAFE) ** 2     # 0 at SAFE, 2.0 at contact
                total -= prox
                self._dbg["obstacleRadius"] -= prox

            total -= 0.005
            self._dbg["time"] -= 0.005

            if dist < self.WAYPOINT_RADIUS and self.current_waypoint_idx[i] < len(
                self.WAYPOINTS
            ):
                self.current_waypoint_idx[i] += 1
                total += 100.0
                self._dbg["bonus"] += 100.0
                self._prev_dist = None
            
            if self.current_waypoint_idx[i] >= len(self.WAYPOINTS):
                total += 300
                self._dbg["bonus"] += 300

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
            if self._obstacleCollisions() >= 1:
                self._term_reason = "collision"
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
        info = {
            "waypoints_reached": [int(idx) for idx in self.current_waypoint_idx],
            "total_waypoints": len(self.WAYPOINTS),
        }

        if self._computeTerminated() or self._computeTruncated():
            info["dbg"] = dict(self._dbg)
            info["termination_reason"] = self._term_reason

        return info

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
                "depth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(48, 64, 1),
                    dtype=np.float32,
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
            # obs.spaces["gimbal_depth"] = spaces.Box(
            #     low=0.0,
            #     high=1.0,
            #     shape=(48, 64, 1),
            #     dtype=np.float32,
            # )

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
        depth = np.nan_to_num(
            np.asarray(images[0][1], dtype=np.float32),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        obs["depth"] = np.ascontiguousarray(
            np.clip(depth, 0.0, 1.0)[..., None],
            dtype=np.float32,
        )

        if self.GIMBAL_CONTROL:
            obs["gimbal_rgb"] = np.zeros(
                (48, 64, 3),
                dtype=np.uint8,
            )
            # obs["gimbal_depth"] = np.zeros(
            #     (48, 64, 1),
            #     dtype=np.float32,
            # )
            # obs["gimbal_rgb"] = np.ascontiguousarray(images[0][3], dtype=np.uint8)
            # obs["gimbal_depth"] = np.ascontiguousarray(
            #     images[0][4][..., None], dtype=np.float32
            # )

        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action[..., 4:6] = 0

        z = self.pos[0, 2]
        if z < 0.15:
            action[..., 2] = max(action[..., 2], 0.0)  # no descending near the floor

        v = action.reshape(-1)[:3]                       # only vx, vy, vz are active
        self._action_delta = float(np.linalg.norm(v - self._prev_action))
        self._prev_action = v.copy()

        return super().step(action)
