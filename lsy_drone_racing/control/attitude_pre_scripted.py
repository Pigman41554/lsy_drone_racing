"""Teacher Policy for Reinforcement Learning
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from lsy_drone_racing.control.constants import MASS
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.ext_tools import TrajectoryTool
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(FresssackController):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env=None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.env = env
        self.freq = config.env.freq
        self.drone_mass = MASS
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81
        self._tick = 0

        # Demo waypoints
        waypoints = np.array(
            [
                [1.0, 1.5, 0.05],
                [0.8, 1.0, 0.2],
                [0.55, -0.3, 0.5],
                [0.2, -1.3, 0.65],
                [1.1, -0.85, 1.1],
                [0.2, 0.5, 0.55],
                [0.0, 0.9, 0.45],
                [0.0, 1.2, 0.45],
                [0.0, 1.2, 0.55],
                [0.0, 1.0, 0.6],
                [0.0, 0.7, 0.75],
                [-0.5, 0.0, 1.1],
                [-0.5, -0.5, 1.1],
            ]
        )
        self.waypoints = waypoints
        t = np.linspace(0,10,len(waypoints))
        trajectory = CubicSpline(t, waypoints)

        # # pre-planned trajectory
        # t, pos, vel = FresssackController.read_trajectory(r"lsy_drone_racing/planned_trajectories/param_a_5_sec_offsets.csv")     
        # trajectory = CubicSpline(t, pos)
        # trajectory reparameterization
        self.traj_tool = TrajectoryTool()
        trajectory = self.traj_tool.extend_trajectory(trajectory)
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory)
        self.theta = 0.0 # traveled distance
        self.prev_pos = obs['pos']

        if len(obs['gates_pos'].shape) > 2: # vector env
            self.gates_theta, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, obs['gates_pos'][0])
        else:
            self.gates_theta, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, obs['gates_pos'])

        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # i = min(self._tick, len(self.x_des) - 1)
        # if i == len(self.x_des) - 1:  # Maximum duration reached
        #     self._finished = True

        try:
            draw_line(self.env, self.arc_trajectory(self.arc_trajectory.x), rgba=np.array([1.0, 0.0, 0.0, 0.2]))
            # draw_line(self.env, self.waypoints, rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except:
            pass

        # fetch waypoint from trajectory
        curr_theta, curr_wp = self.traj_tool.find_nearest_waypoint(self.arc_trajectory, obs['pos'], self.gates_theta[obs['target_gate']])
        self.theta += np.linalg.norm(obs['pos'] - self.prev_pos) # update theta
        alpha = 1.0
        self.theta = (1-alpha)*self.theta + alpha*curr_theta
        self.prev_pos = obs['pos']
        des_pos = self.arc_trajectory(self.theta+0.25)
        des_vel = np.zeros(3)
        # vel = self.arc_trajectory.derivative(1)(self.theta+0.15)
        # des_vel = 0.5 * vel/np.linalg.norm(vel)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self.freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)
        thrust_desired = max(thrust_desired, 0.3 * self.drone_mass * self.g)
        thrust_desired = min(thrust_desired, 1.8 * self.drone_mass * self.g)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
        thrust_desired, euler_desired
        return np.concatenate([[thrust_desired], euler_desired], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self.i_error[:] = 0
        self._tick = 0
