from sb3_contrib import MaskablePPO
from rl.env import DeepThought42 
import pickle
import os
from tqdm import tqdm

# Specify PPO model path here (relative or absolute)
PPO_MODEL_PATH = "rl/trained on newDataset/20000_ppo_deepthought42.zip"  # Change to your model file path
ppo_model = MaskablePPO.load(PPO_MODEL_PATH)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "oldDataset")

env = DeepThought42(models_dir=MODELS_DIR, dataset_dir=DATASET_DIR)
stepdepth = []
num_episodes = len(env.initial_states)

for _ in tqdm(range(num_episodes), desc="Evaluating PPO on instances"):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action_mask = env.get_action_mask()
        action, _ = ppo_model.predict(obs, deterministic=True, action_masks=action_mask)
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        #print("Step:", steps, "Action:", action, "Reward:", reward, "Done:", done)
    
    #print(f"Episode completed in {steps} steps.")
    stepdepth.append(steps)

with open("stepdepth_0802_nD.pkl", "wb") as f:
    pickle.dump(stepdepth, f)