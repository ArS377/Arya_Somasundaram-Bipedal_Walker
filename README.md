# Arya_Somasundaram-Bipedal_Walker

This repository contains code and configuration for training a reinforcement learning agent on a Bipedal Walker environment (customized reward / forward-bonus experiments). The project is organized into a small set of Python modules — each file has a single responsibility described below.

## Files and purpose

- `baseline_model.py`

  - This is where the baseline PPO model is defined and trained.

- `callback.py`

  - Contains custom gifrecording callbacks and weight normalization callbacks.

- `config.py`

  - Constains static variables that store specific constant values and hyperparameters.

- `envs.py`

  - The environment is defined here, and the custom reward wrapper is shaped here.

- `train.py`

  - The PPO model is defined here and trained.
