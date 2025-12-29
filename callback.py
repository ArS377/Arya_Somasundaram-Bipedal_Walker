import imageio
import torch
from stable_baselines3.common.callbacks import BaseCallback

from envs import make_eval_env


class GifRecorderCallback(BaseCallback):
    def __init__(self, gif_path="walker.gif", record_every=100_000, fps=30):
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
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            if terminated or truncated:
                break

        imageio.mimsave(self.gif_path, frames, fps=self.fps)
        print(f"Saved GIF to {self.gif_path}")


class WeightNormCallback(BaseCallback):
    """Periodically renormalize policy weights to keep training stable."""
    def __init__(self, every=200_000):
        super().__init__()
        self.every = every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            with torch.no_grad():
                for p in self.model.policy.parameters():
                    if p.dim() > 1:
                        norm = p.norm(dim=0, keepdim=True).clamp(min=1e-6)
                        p.mul_(0.9 + 0.1 / norm)  # Gentler adjustment
            print(f"[WeightNorm] Renormalized at {self.num_timesteps}")
        return True
