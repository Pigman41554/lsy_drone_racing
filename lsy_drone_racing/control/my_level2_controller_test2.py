from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import DM, MX, cos, dot, floor, if_else, norm_2, sin, vertcat
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.path_planner import PathConfig, PathPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MPCCConfig:
    N_horizon: int = 40
    T_horizon: float = 0.7

    model_arc_step: float = 0.05
    model_traj_length: float = 15.0

    q_lag: float = 90.0
    q_lag_peak: float = 700.0
    q_contour: float = 150.0
    q_contour_peak: float = 1000.0
    q_attitude: float = 1.0

    r_thrust: float = 0.25
    r_roll: float = 0.45
    r_pitch: float = 0.45
    r_yaw: float = 0.70

    mu_speed: float = 7.5
    w_speed_gate: float = 14.0

    pos_bounds: tuple = (
        (-2.6, 2.6),
        (-2.0, 1.8),
        (-0.1, 2.0),
    )
    vel_bounds: tuple = (0.0, 5.0)

    planned_duration: float = 30.0
    log_interval: int = 100


class MPCCController(Controller):
    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        mpcc_config: Optional[MPCCConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        super().__init__(obs, info, config)

        self.mpcc_cfg = mpcc_config or MPCCConfig()
        self.path_cfg = path_config or PathConfig()

        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self.finished = False

        self._dyn_params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(self._dyn_params["mass"])
        self._gravity = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = self._mass * self._gravity

        self.thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        self.thrust_max = float(self._dyn_params["thrust_max"]) * 4.0

        self.path_planner = PathPlanner(self.path_cfg)
        self._initial_pos = obs["pos"].copy()

        self._last_gate_flags = None
        self._last_obst_flags = None

        num_gates = len(obs["gates_pos"])
        self._gate_detected_flags = np.zeros(num_gates, dtype=bool)
        self._gate_real_positions = np.full((num_gates, 3), np.nan)

        self._plan_trajectory(obs)

        self.N = self.mpcc_cfg.N_horizon
        self.T = self.mpcc_cfg.T_horizon
        self.dt = self.T / self.N
        self.model_arc_step = self.mpcc_cfg.model_arc_step
        self.model_traj_length = self.mpcc_cfg.model_traj_length

        self._build_solver()

        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)

        self._current_pos = obs["pos"].copy()

        print(f"[MPCC] Initialized. Horizon: N={self.N}, T={self.T:.2f}s")
        print(f"[MPCC] Arc trajectory length: {self.arc_trajectory.x[-1]:.2f}")

    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        print(f"[MPCC] Planning trajectory at T={self._step_count / self._ctrl_freq:.2f}s")

        obs_planning = obs.copy()
        obs_planning["pos"] = self._initial_pos.copy()

        result = self.path_planner.plan_trajectory(
            obs_planning,
            trajectory_duration=self.mpcc_cfg.planned_duration,
            sampling_freq=self._ctrl_freq,
            for_mpcc=True,
            mpcc_extension_length=self.mpcc_cfg.model_traj_length,
        )

        self._trajectory_result = result
        self.trajectory = result.spline
        self.arc_trajectory = result.arc_spline
        self.waypoints = result.waypoints
        self.total_arc_length = result.total_length

        self._cached_gate_centers = obs["gates_pos"].copy()
        self._cached_obstacles = obs["obstacles_pos"].copy()

    def _build_solver(self) -> None:
        model = self._build_dynamics_model()

        ocp = AcadosOcp()
        ocp.model = model

        self.nx = model.x.rows()
        self.nu = model.u.rows()

        ocp.solver_options.N_horizon = self.N

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = self._build_cost_expression()

        ocp.constraints.lbx = np.array(
            [self.thrust_min, self.thrust_min, -1.57, -1.57, -1.57]
        )
        ocp.constraints.ubx = np.array(
            [self.thrust_max, self.thrust_max, 1.57, 1.57, 1.57]
        )
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

        ocp.constraints.lbu = np.array([-10.0, -8.0, -8.0, -8.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 8.0, 8.0, 8.0, 1.30])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.parameter_values = self._encode_trajectory_params()

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = self.T

        self.solver = AcadosOcpSolver(
            ocp,
            json_file="mpcc_racing.json",
            verbose=False,
        )
        self.ocp = ocp

    def _build_dynamics_model(self) -> AcadosModel:
        model_name = "mpcc_drone_racing"

        mass = self._mass
        gravity = self._gravity

        k = np.array(self._dyn_params["rpy_coef"], dtype=float)
        d = np.array(self._dyn_params["rpy_rates_coef"], dtype=float)
        b = np.array(self._dyn_params["cmd_rpy_coef"], dtype=float)

        eps = 1e-9
        a = -k / (d + eps)
        beta = -b / (d + eps)

        self.px = MX.sym("px")
        self.py = MX.sym("py")
        self.pz = MX.sym("pz")

        self.vx = MX.sym("vx")
        self.vy = MX.sym("vy")
        self.vz = MX.sym("vz")

        self.roll = MX.sym("roll")
        self.pitch = MX.sym("pitch")
        self.yaw = MX.sym("yaw")

        self.f_collective = MX.sym("f_collective")
        self.f_cmd = MX.sym("f_cmd")

        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")

        self.theta = MX.sym("theta")

        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(
            self.px,
            self.py,
            self.pz,
            self.vx,
            self.vy,
            self.vz,
            self.roll,
            self.pitch,
            self.yaw,
            self.f_collective,
            self.f_cmd,
            self.r_cmd,
            self.p_cmd,
            self.y_cmd,
            self.theta,
        )

        inputs = vertcat(
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        thrust = self.f_collective
        inv_mass = 1.0 / mass

        ax = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * cos(self.yaw)
            + sin(self.roll) * sin(self.yaw)
        )
        ay = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * sin(self.yaw)
            - sin(self.roll) * cos(self.yaw)
        )
        az = inv_mass * thrust * cos(self.roll) * cos(self.pitch) - gravity

        f_dyn = vertcat(
            self.vx,
            self.vy,
            self.vz,
            ax,
            ay,
            az,
            float(a[0]) * self.roll + float(beta[0]) * self.r_cmd,
            float(a[1]) * self.pitch + float(beta[1]) * self.p_cmd,
            float(a[2]) * self.yaw + float(beta[2]) * self.y_cmd,
            10.0 * (self.f_cmd - self.f_collective),
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        n_samples = int(self.model_traj_length / self.model_arc_step)

        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)
        self.qc_dyn = MX.sym("qc_dyn", n_samples)

        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params

        return model

    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        m = len(theta_vec)

        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (m - 1)
        idx_low = floor(idx_float)
        idx_high = idx_low + 1
        alpha = idx_float - idx_low

        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_high >= m, m - 1, idx_high)

        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])

        return (1.0 - alpha) * p_low + alpha * p_high

    def _encode_trajectory_params(self) -> np.ndarray:
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_step)

        pd_vals = self.arc_trajectory(theta_samples)
        tp_vals = self.arc_trajectory.derivative(1)(theta_samples)

        qc_dyn = np.zeros_like(theta_samples)

        for gate_center in self._cached_gate_centers:
            d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
            qc_gate = 0.4 * np.exp(-8.0 * d_gate**2)
            qc_dyn = np.maximum(qc_dyn, qc_gate)

        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            qc_obs = 0.4 * np.exp(-8.0 * d_obs_xy**2)
            qc_dyn = np.maximum(qc_dyn, qc_obs)

        return np.concatenate(
            [
                pd_vals.reshape(-1),
                tp_vals.reshape(-1),
                qc_dyn.reshape(-1),
            ]
        )

    def _build_cost_expression(self):
        cfg = self.mpcc_cfg

        position = vertcat(self.px, self.py, self.pz)
        attitude = vertcat(self.roll, self.pitch, self.yaw)
        control = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)

        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_step)

        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_theta = self._piecewise_linear_interp(
            self.theta,
            theta_grid,
            self.qc_dyn,
            dim=1,
        )

        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag

        q_att = cfg.q_attitude * DM(np.eye(3))

        track_cost = (
            (cfg.q_lag + cfg.q_lag_peak * qc_theta) * dot(e_lag, e_lag)
            + (cfg.q_contour + cfg.q_contour_peak * qc_theta)
            * dot(e_contour, e_contour)
            + attitude.T @ q_att @ attitude
        )

        r_control = DM(np.diag([cfg.r_thrust, cfg.r_roll, cfg.r_pitch, cfg.r_yaw]))
        smooth_cost = control.T @ r_control @ control

        speed_cost = (
            -cfg.mu_speed * self.v_theta_cmd
            + cfg.w_speed_gate * qc_theta * self.v_theta_cmd**2
        )

        return track_cost + smooth_cost + speed_cost

    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False

        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)

        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False

        if curr_obst.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr_obst
            return False

        gate_trigger = np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = np.any((~self._last_obst_flags) & curr_obst)

        for i, is_visited in enumerate(curr_gates):
            if is_visited and not self._gate_detected_flags[i]:
                self._gate_detected_flags[i] = True
                self._gate_real_positions[i] = obs["gates_pos"][i]
                print(
                    f"[GATE DETECTED] Gate {i + 1} at real position: "
                    f"[{obs['gates_pos'][i][0]:.3f}, "
                    f"{obs['gates_pos'][i][1]:.3f}, "
                    f"{obs['gates_pos'][i][2]:.3f}]"
                )

        self._last_gate_flags = curr_gates.copy()
        self._last_obst_flags = curr_obst.copy()

        return bool(gate_trigger or obst_trigger)

    def _check_position_bounds(self, pos: NDArray[np.floating]) -> bool:
        for i, (low, high) in enumerate(self.mpcc_cfg.pos_bounds):
            if pos[i] < low or pos[i] > high:
                return False
        return True

    def _check_velocity_bounds(self, vel: NDArray[np.floating]) -> bool:
        speed = float(np.linalg.norm(vel))
        low, high = self.mpcc_cfg.vel_bounds
        return low <= speed <= high

    def _fallback_hover(self) -> NDArray[np.floating]:
        cmd = np.array(
            [0.0, 0.0, 0.0, self.hover_thrust],
            dtype=np.float32,
        )
        cmd[3] = np.clip(cmd[3], self.thrust_min, self.thrust_max)
        return cmd

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        self._current_pos = obs["pos"].copy()

        if self._detect_environment_change(obs):
            print("[MPCC] Environment change detected, replanning...")
            self._plan_trajectory(obs)

            try:
                theta_proj, _ = self.path_planner.find_closest_point(
                    self.arc_trajectory,
                    obs["pos"],
                )
                self.last_theta = max(self.last_theta, float(theta_proj))
                self.last_theta = float(
                    np.clip(self.last_theta, 0.0, self.arc_trajectory.x[-1])
                )
            except Exception as exc:
                print(f"[MPCC] Warning: could not project theta after replanning: {exc}")

            param_vec = self._encode_trajectory_params()
            for k in range(self.N + 1):
                self.solver.set(k, "p", param_vec)

        roll, pitch, yaw = Rotation.from_quat(obs["quat"]).as_euler("xyz")

        x_now = np.concatenate(
            [
                obs["pos"],
                obs["vel"],
                np.array([roll, pitch, yaw]),
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
                np.array([self.last_theta]),
            ]
        )

        if not hasattr(self, "_x_warm"):
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]

        for i in range(self.N):
            self.solver.set(i, "x", self._x_warm[i])
            self.solver.set(i, "u", self._u_warm[i])

        self.solver.set(self.N, "x", self._x_warm[self.N])
        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)

        if self.last_theta >= float(self.arc_trajectory.x[-1]):
            self.finished = True
            print("[MPCC] Finished: reached end of path")

        if not self._check_position_bounds(obs["pos"]):
            self.finished = True
            print("[MPCC] Finished: position out of bounds")

        if not self._check_velocity_bounds(obs["vel"]):
            print("[MPCC] Warning: velocity out of bounds")

        status = self.solver.solve()

        if status != 0:
            print(f"[MPCC] Solver failed with status {status}, using fallback hover")
            self._step_count += 1
            return self._fallback_hover()

        self._x_warm = [self.solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.solver.get(i, "u") for i in range(self.N)]

        x_next = self.solver.get(1, "x")

        self.last_f_collective = float(x_next[9])
        self.last_f_cmd = float(x_next[10])
        self.last_rpy_cmd = np.array(x_next[11:14])
        self.last_theta = float(x_next[14])
        self.last_theta = float(np.clip(self.last_theta, 0.0, self.arc_trajectory.x[-1]))

        cmd = np.array(
            [
                self.last_rpy_cmd[0],
                self.last_rpy_cmd[1],
                self.last_rpy_cmd[2],
                self.last_f_cmd,
            ],
            dtype=np.float32,
        )

        cmd[0:3] = np.clip(cmd[0:3], -0.6, 0.6)
        cmd[3] = np.clip(cmd[3], self.thrust_min, self.thrust_max)

        if not np.all(np.isfinite(cmd)):
            print("[MPCC] Warning: non-finite command, using fallback hover")
            cmd = self._fallback_hover()

        if self._step_count % self.mpcc_cfg.log_interval == 0:
            print(
                f"[MPCC] T={self._step_count / self._ctrl_freq:.2f}s, "
                f"theta={self.last_theta:.2f}/{self.arc_trajectory.x[-1]:.2f}, "
                f"cmd=[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}, {cmd[3]:.2f}]"
            )

        self._step_count += 1
        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return self.finished

    def episode_callback(self) -> None:
        print("[MPCC] Episode reset")

        self._step_count = 0
        self.finished = False

        for attr in ["_last_gate_flags", "_last_obst_flags", "_x_warm", "_u_warm"]:
            if hasattr(self, attr):
                delattr(self, attr)

        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)

    def get_debug_lines(self):
        debug_lines = []

        if hasattr(self, "arc_trajectory"):
            try:
                full_path = self.arc_trajectory(self.arc_trajectory.x)
                debug_lines.append(
                    (full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0)
                )
            except Exception:
                pass

        if hasattr(self, "_x_warm"):
            try:
                pred_states = np.array([x[:3] for x in self._x_warm])
                debug_lines.append(
                    (pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0)
                )
            except Exception:
                pass

        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory"):
            try:
                target = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_pos, target])
                debug_lines.append(
                    (segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0)
                )
            except Exception:
                pass

        return debug_lines

    def get_trajectory(self) -> CubicSpline:
        return self.trajectory

    def get_arc_trajectory(self) -> CubicSpline:
        return self.arc_trajectory

    def get_progress(self) -> float:
        if hasattr(self, "arc_trajectory"):
            return self.last_theta / self.arc_trajectory.x[-1]
        return 0.0