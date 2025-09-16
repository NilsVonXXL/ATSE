import gymnasium as gym
from rl.env import DeepThought42
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
import torch
import numpy as np
import random
from stable_baselines3.common.callbacks import CheckpointCallback

seed = 42
# Register environment
gym.register(
    id="DeepThought42-v0",
    entry_point="rl.env:DeepThought42",
)

def mask_fn(env):
    # Unwrap until we reach the base env with get_action_mask
    while hasattr(env, "env"):
        env = env.env
    return env.get_action_mask()

env = gym.make(
    "DeepThought42-v0",
    models_dir="c:/ATSE/ATSE/models",
    dataset_dir="c:/ATSE/ATSE/newDataset",
)
# Wrap the environment for action masking
env = ActionMasker(env, mask_fn)

env.reset(seed=seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    n_steps=32  # Log more frequently (default is 2048)
)

checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # Save every 1000 steps
    save_path='./checkpoints/',
    name_prefix='ppo_deepthought42'
)

model.learn(
    total_timesteps=10_000,
    tb_log_name="first_run",
    progress_bar=True,
    callback=checkpoint_callback
)

obs, info = env.reset(seed=seed)
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)  # <-- Ensure action is an integer
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset(seed=seed)

model.save("ppo_deepthought42")

env.close()