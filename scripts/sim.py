"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations  #让类型注解延迟解析（避免循环引用问题）

import logging  #"""打印日志"""
from pathlib import Path  #"""跨平台处理文件路径"""
from typing import TYPE_CHECKING  #"""类型检查时才导入（避免运行时开销）"""

import fire  #"""自动把函数变成命令行接口 就可以用python sim.py调用"""
import gymnasium  #"""强化学习框架"""
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  #"""数据转换工具"""

from lsy_drone_racing.utils import load_config, load_controller
#"""loadconfig读取.toml配置  loadcontroller动态加载控制器类"""

if TYPE_CHECKING:
    from ml_collections import ConfigDict  #"""存配置"""

    from lsy_drone_racing.control.controller import Controller  #"""控制器基类"""
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv  #"""仿真环境"""


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.
       评估无人机控制器在多个回合中的表现。
    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
                配置文件的路径
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
            lsy_drone_racing/control/中控制器文件的名称或 None。
            如果为 None 则使用配置文件中指定的控制器
        n_runs: The number of episodes.回合数
        render: Enable/disable rendering the simulation.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
      #"""加载配置中 在当前文件项目根目录里面找config"""
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render  #"""如果命令行没说就用配置文件，说了就覆盖配置"""
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
      #"""命令行说了就用命令行的controller 没说就用config的"""
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)  #"""把env输出转换成numpy"""

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        #"""创建了控制器实例"""
        i = 0  #"""step计数"""
        fps = 60  #"""渲染帧率"""

        while True:
            curr_time = i / config.env.freq  #"""时间=step/频率"""

            action = controller.compute_control(obs, info)
            #"""核心接口 控制器根据compute_control这个函数输出动作"""

            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            #"""obs是新状态 reward是rl里要用的奖励 terminated是成功失败标志符"""
            #"""这一步是在action更新后更新控制器内部状态"""
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )#"""更新内部状态 判断是否提前结束"""
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    controller.render_callback(env.unwrapped.sim)
                    env.render()
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        #"""用于训练或者统计"""
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
