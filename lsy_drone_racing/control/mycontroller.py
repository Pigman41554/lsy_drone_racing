import numpy as np
from lsy_drone_racing.control.controller import Controller


class MyController(Controller):
    def __init__(self, obs, info, config):
        print("🔥 MY CONTROLLER LOADED")

        self.mass = 0.03  # 默认值，够用了
        self.g = 9.81

    def compute_control(self, obs, info=None):
        # hover thrust
        thrust = self.mass * self.g

        action = np.array([
            0.0,   # roll
            0.0,   # pitch
            0.0,   # yaw
            thrust
        ], dtype=np.float32)

        print("action shape:", action.shape)  # 🔍 调试

        return action