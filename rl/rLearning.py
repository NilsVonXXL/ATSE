
import gymnasium as gym
from rl.env import DeepThought42
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
import torch
import numpy as np
import random
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_DIR = None
DATASET_DIR = os.path.join(BASE_DIR, "randomDataset(-5,5)")


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

# Factory for parallel envs
def make_env():
    def _init():
        env = gym.make(
            "DeepThought42-v0",
            models_dir=MODELS_DIR,
            dataset_dir=DATASET_DIR,
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO with variable timesteps.")
    parser.add_argument('--timesteps', type=int, default=50_000, help='Total timesteps for training (default: 100000)')
    args = parser.parse_args()

    total_timesteps = args.timesteps
    # Set names based on timesteps
    checkpoint_dir = f'./checkpoints/{total_timesteps}steps/'
    tb_log_name = f"{total_timesteps}_steps_run"
    model_save_name = f"{total_timesteps}_ppo_deepthought42_random_-5_5"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Ensure single-threaded operation for each subprocess
    torch.set_num_threads(1)
    # Number of parallel environments
    num_envs = 10
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    env.reset()
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
        save_freq=10000,  # Save every 10000 steps
        save_path=checkpoint_dir,
        name_prefix='ppo_deepthought42'
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name,
        progress_bar=True,
        callback=checkpoint_callback
    )

    model.save(model_save_name)

    env.close()
    