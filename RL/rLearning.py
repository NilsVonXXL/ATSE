import gymnasium as gym
from rl.env import DeepThought42
from stable_baselines3 import PPO

env = DeepThought42(models_dir="c:/ATSE/ATSE/models", dataset_dir="c:/ATSE/ATSE/newDataset")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()