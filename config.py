import torch.nn as nn

LOG_DIR = "./tensorboard_logs/"
GIF_PATH = "bipedal_walker.gif"
RECORD_EVERY = 200000
REGULAR_MODEL_SAVE_PATH = "ppo_bipedal_walker_regular"
HARDCORE_MODEL_SAVE_PATH = "ppo_bipedal_walker_hardcore"
NAME = "PPO_Bipedal_ForwardBonus"

TIMESTEPS = 5000000

POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.ReLU,
    ortho_init=True,
)
