import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=10000, tb_log_name="test_run")