import gymnasium as gym
import imageio
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

def make_eval_env():
    return gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")

class GifRecorderCallback(BaseCallback):
    def __init__(self, gif_path="walker_basic.gif", record_every=100_000, fps=30):
        super().__init__()
        self.gif_path = gif_path
        self.record_every = record_every
        self.fps = fps
        self.eval_env = make_eval_env()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0:
            self.record_gif()
        return True

    def record_gif(self, episode_length=1600):
        frames = []
        obs, _ = self.eval_env.reset()

        for _ in range(episode_length):
            frames.append(self.eval_env.render())
            # Predict action using the current model state
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            if terminated or truncated:
                break

        imageio.mimsave(self.gif_path, frames, fps=self.fps)
        print(f"Saved GIF to {self.gif_path} at step {self.num_timesteps}")

log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("BipedalWalker-v3", hardcore=True)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir
)

gif_callback = GifRecorderCallback(
    gif_path="basic_bipedal_walker.gif",
    record_every=100_000
)

model.learn(
    total_timesteps=1_000_000,
    callback=gif_callback,
    tb_log_name="PPO_Basic_Walker"
)

model.save("ppo_basic_bipedal_walker")