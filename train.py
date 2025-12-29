import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
import torch

from envs import make_env
from envs import make_regular_env
from callback import GifRecorderCallback, WeightNormCallback
from config import (
    LOG_DIR,
    POLICY_KWARGS,
    TIMESTEPS,
    GIF_PATH,
    RECORD_EVERY,
    NAME,
    REGULAR_MODEL_SAVE_PATH,
    HARDCORE_MODEL_SAVE_PATH
)

num_envs = 4
env = DummyVecEnv([make_regular_env] * num_envs)

os.makedirs(LOG_DIR, exist_ok=True)

model = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=POLICY_KWARGS,

    use_sde=True,
    sde_sample_freq=16,
    ent_coef=0.05,

    n_steps=8192,
    batch_size=256,
    learning_rate=3e-4,
    clip_range=0.2,

    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=0.5,

    verbose=1,
    tensorboard_log=LOG_DIR,
)

gif_callback = GifRecorderCallback(gif_path=GIF_PATH, record_every=RECORD_EVERY)
callback = CallbackList([WeightNormCallback()])

model.learn(total_timesteps=TIMESTEPS, callback=callback , tb_log_name=NAME)
model.save(REGULAR_MODEL_SAVE_PATH)

env = DummyVecEnv([make_env] * num_envs)
model.set_env(env)

model.learning_rate = 1e-4
model.ent_coef = 0.03

gif_callback = GifRecorderCallback(gif_path=GIF_PATH, record_every=RECORD_EVERY)
callback = CallbackList([gif_callback, WeightNormCallback()])

model.learn(total_timesteps=TIMESTEPS, callback=callback, tb_log_name=NAME)
model.save(HARDCORE_MODEL_SAVE_PATH)
