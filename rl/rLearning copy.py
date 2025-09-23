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

def make_env(rank):
    def _init():
        env = gym.make(
            "DeepThought42-v0",
            models_dir=MODELS_DIR,
            dataset_dir=DATASET_DIR,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init

num_envs = 4  
env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
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
    total_timesteps=50_000,
    tb_log_name="first_run",
    progress_bar=True,
    callback=checkpoint_callback
)


# --- Evaluation Loop (single env for demo) ---
eval_env = gym.make(
    "DeepThought42-v0",
    models_dir=MODELS_DIR,
    dataset_dir=DATASET_DIR,
)
eval_env = ActionMasker(eval_env, mask_fn)
obs, info = eval_env.reset(seed=seed)
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    action = int(action)  # <-- Ensure action is an integer
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    if done:
        obs, info = eval_env.reset(seed=seed)

model.save("ppo_deepthought42")

eval_env.close()