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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "newDataset")


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
    #Ensure single-threaded operation for each subprocess
    torch.set_num_threads(1)
    # Number of parallel environments
    num_envs = 2
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
        save_freq=1000,  # Save every 1000 steps
        save_path='./checkpoints/',
        name_prefix='ppo_deepthought42'
    )

    model.learn(
        #debug
        total_timesteps=1_000,
        #total_timesteps=30_000,
        tb_log_name="first_run",
        progress_bar=True,
        callback=checkpoint_callback
    )

    # For evaluation, use a single env or handle vectorized obs/actions
    #obs = env.reset()
    #for i in range(1000):
    #    actions, _states = model.predict(obs, deterministic=True)
    #    obs, rewards, dones, truncateds, infos = env.step(actions)
    #    if dones.any():
    #        # Only reset the environments that are done
    #        obs = env.reset_done()

    model.save("ppo_deepthought42")

    env.close()