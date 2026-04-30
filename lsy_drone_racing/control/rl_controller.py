"""Controller that follows RL policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.constants import GRAVITY, MASS
from lsy_drone_racing.crazyflow.sim.physics import ang_vel2rpy_rates
#from lsy_drone_racing.utils.utils import draw_line
try:
    from crazyflow.sim.visualize import draw_line
    HAS_VIZ = True
except ImportError:
    draw_line = None
    HAS_VIZ = False
if TYPE_CHECKING:
    from numpy.typing import NDArray


class RLController(Controller):
    """Trajectory controller following RL policy."""

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        super().__init__(obs, info, config)

        log_dir = Path(__file__).parent.parent / "reinforcement_learning/log5"
        model_path = log_dir / "ppo_final_model_stablest_5s.zip"

        print(f"[RLController] Loaded model: {model_path.name}")
        self.model = PPO.load(model_path, device="cpu")

        self.act_bias = np.array(
            [MASS * GRAVITY, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        self.action = np.zeros(4, dtype=np.float32)
        self.d_safe = 1.0

        self._tick = 0
        self._freq = config.env.freq
        self._finished = False

        self.trajectory_record = []
        self.gate_corner_lines = []
        self.obstacle_line = None

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        state = self._obs_to_state(obs, self.action)

        input_obs = state[None, :]
        action, _ = self.model.predict(input_obs, deterministic=True)

        self.action = action.squeeze().astype(np.float32) + self.act_bias

        self.trajectory_record.append(obs["pos"].copy())

        return self.action.astype(np.float32)

    def _obs_to_state(
        self,
        obs: dict[str, NDArray[np.floating]],
        action: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        pos = obs["pos"].squeeze()
        vel = obs["vel"].squeeze()
        quat = obs["quat"].squeeze()
        ang_vel = obs["ang_vel"].squeeze()

        curr_gate = int(obs["target_gate"])

        gate_size = np.array([0.4, 0.4])
        half_w, half_h = gate_size[0] / 2.0, gate_size[1] / 2.0

        gate_rot_mat = R.from_quat(obs["gates_quat"][curr_gate]).as_matrix()

        corners_local = np.array(
            [
                [-half_w, 0.0, half_h],
                [half_w, 0.0, half_h],
                [-half_w, 0.0, -half_h],
                [half_w, 0.0, -half_h],
            ],
            dtype=np.float64,
        )

        gate_corners_pos = (
            gate_rot_mat @ corners_local.T
        ).T + obs["gates_pos"][curr_gate]

        rel_pos_gate = gate_corners_pos - pos[None, :]

        self.gate_corner_lines = [
            np.stack([gate_corners_pos[i], pos]) for i in range(4)
        ]

        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)

        if obstacles_pos.size == 0:
            rel_xy_obst_gaus = np.zeros(2, dtype=np.float64)
            self.obstacle_line = None
        else:
            obstacles_pos = obstacles_pos.reshape(-1, 3)
            obst_rel_xy_list = obstacles_pos[:, :2] - pos[:2]
            obst_dists = np.linalg.norm(obst_rel_xy_list, axis=-1)

            closest_obst_idx = int(np.argmin(obst_dists))
            rel_xy_obst = obst_rel_xy_list[closest_obst_idx]
            dist = float(obst_dists[closest_obst_idx])

            rel_xy_obst_gaus = (
                rel_xy_obst
                * np.exp(-((dist / (0.5 * self.d_safe)) ** 2))
                / (dist + 1e-6)
            )

            obstacle_vec_3d = np.concatenate([rel_xy_obst_gaus, np.array([0.0])])
            self.obstacle_line = np.stack([pos, pos + obstacle_vec_3d])

        rot_mat = R.from_quat(quat).as_matrix().reshape(-1)
        rpy_rates = ang_vel2rpy_rates(ang_vel, quat)

        state = np.concatenate(
            [
                pos,
                vel,
                rot_mat,
                rpy_rates,
                rel_pos_gate.reshape(-1),
                rel_xy_obst_gaus,
                action,
            ]
        )

        return state.astype(np.float32)

    def render_callback(self, sim) -> None:
        if not HAS_VIZ:
            return

        if len(self.trajectory_record) >= 2:
            draw_line(
                sim,
                np.array(self.trajectory_record),
                rgba=self.hex2rgba("#2BFF00F9"),
            )

        for line in self.gate_corner_lines:
            draw_line(
                sim,
                line,
                rgba=np.array([1.0, 1.0, 1.0, 0.2]),
            )

        if self.obstacle_line is not None:
            draw_line(
                sim,
                self.obstacle_line,
                rgba=np.array([1.0, 0.0, 1.0, 0.5]),
            )
    @staticmethod
    def get_latest_model_path(
        log_dir: str,
        lesson: int,
        idx: int | None = None,
    ) -> tuple[Path, int]:
        log_path = Path(log_dir)
        pattern = re.compile(rf"ppo_final_model_{lesson}_(\d+)\.zip")

        model_files = [
            (file, int(match.group(1)))
            for file in log_path.glob(f"ppo_final_model_{lesson}_*.zip")
            if (match := pattern.match(file.name))
        ]

        if not model_files:
            raise FileNotFoundError(
                f"No model found for lesson {lesson} in {log_path}"
            )

        latest_file, max_idx = max(model_files, key=lambda item: item[1])

        if idx is not None:
            for file, number in model_files:
                if number == idx:
                    return file, number

            raise FileNotFoundError(
                f"Model ppo_final_model_{lesson}_{idx}.zip not found in {log_path}"
            )

        return latest_file, max_idx

    @staticmethod
    def hex2rgba(hex_color: str = "#FFFFFFFF") -> NDArray[np.float64]:
        hex_color = hex_color.lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16)

        return np.array([r / 255, g / 255, b / 255, a / 255])

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self) -> None:
        self._tick = 0
        self._finished = False
        self.action = np.zeros(4, dtype=np.float32)
        self.trajectory_record = []
        self.gate_corner_lines = []
        self.obstacle_line = None