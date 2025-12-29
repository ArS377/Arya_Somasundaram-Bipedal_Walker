import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import numpy as np

class ForwardBonusWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        forward_vel = obs[2]
        reward += 10.0 * max(0, forward_vel) #rewarding more forward movement

        if abs(forward_vel) < 0.05:
            reward -= 1.0

        return obs, reward, terminated, truncated, info

def make_regular_env():
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
    env = ForwardBonusWrapper(env)
    env = Monitor(env)
    return env

def make_env():
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    env = ForwardBonusWrapper(env)
    env = Monitor(env)
    return env

def make_eval_env():
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    env = ForwardBonusWrapper(env)
    return env
