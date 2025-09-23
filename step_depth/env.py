import re
import gymnasium as gym
import numpy as np
import random
import pickle
import os
import json
from step_depth.bab_rl import bab_step
from micrograd.ibp import Interval, ibp
from micrograd.engine import Value

class DeepThought42(gym.Env):
    def __init__(self, models_dir, initial_states):
        super().__init__()
        self.models_dir = models_dir
        self.relu_dim = 32
        self.weights_dim = 337 
        self.inputs_dim = 3

        self.action_space = gym.spaces.Discrete(self.relu_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.weights_dim + self.inputs_dim + self.relu_dim,),
            dtype=np.float32
        )

        # Step counter for reward scaling
        self._step_count = 0
        self._reward = -1

        # Gather all (model_path, input_folder) pairs
        self.initial_states = initial_states
        self.model = None
        self.inputs = None
        self.splits = None
        self.in_bounds = None
        self.score = None
        self.state = None
        self.relu_status = None

    def reset(self, seed=None, options=None):
        weights_path, input_path, input_folder = self.initial_states
        self._step_count = 0  # Reset step counter at the start of each episode

        weights_path = os.path.normpath(weights_path)
        parts = weights_path.split(os.sep)
        domain = parts[-4]      # 'blobs'
        net_num = parts[-2]     # '2'
        model_name = f"model_{domain}_{net_num}.pkl"
        model_path = os.path.join(self.models_dir, model_name)

        # Load the model from models/
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        weights = [p for p in self.model.parameters()]
        self.weights_flat = np.array([w.data for w in weights], dtype=np.float32)

        # Parse inputs from folder name
        match = re.match(r'input-x-([-\d.]+)-y-([-\d.]+)-eps-([-\d.]+)', input_folder)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            eps = float(match.group(3))
            self.inputs = [x, y, eps]
        else:
            self.inputs = [0.0, 0.0, 0.0]
        self.inputs_vec = np.array(self.inputs, dtype=np.float32)

        # Set initial bounds
        tmp = [Value(x), Value(y)]
        self.in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in tmp}

        # Load initial relu nodes from step_0 folder
        step0_path = os.path.join(input_path, 'step_0')
        relu_json_path = os.path.join(step0_path, 'relu_nodes.json')
        with open(relu_json_path, 'r') as f:
            relu_dic = json.load(f)
        relu_nodes = np.array(list(relu_dic.values()), dtype=np.float32)
        self.relu_status = relu_nodes
        #print(self.relu_status)

        # Initial splits
        self.splits = dict()
        self.score = self.model(self.in_bounds)
        self.node_bounds = ibp(self.score, self.in_bounds, return_all=True)

        # Build initial state
        self.state = np.concatenate([self.weights_flat, self.inputs_vec, relu_nodes])
        
        info = {
            "score": self.score.data,
            "relu_nodes": self.relu_status
        }
        #print(weights_path, input_path,input_folder)
        #print(info)
        return self.state, info

    def step(self, action):
        # Call bab_step with node_bounds as input
        next_splits, next_relu_status, done, info = bab_step(self.score, self.in_bounds, self.node_bounds, self.splits, action)
        self.relu_status = np.array(list(next_relu_status.values()), dtype=np.float32)
        self.splits = next_splits
        self.inputs = self.inputs_vec

        # Build next state
        weights_flat = self.weights_flat
        next_state = np.concatenate([weights_flat, self.inputs_vec, self.relu_status])
        self.state = next_state

        if self._step_count == 0:
            pass
        else:
            # Multiply reward by 1.3 for each step in the episode
            self._reward = self._reward ** self._step_count

        self._step_count += 1
        reward = -1
        
        info = {
            "splits": self.splits.copy(),
            "relu_status": self.relu_status.copy(),
            "action": int(action),
            "reward": reward,
            "done": done
        }
        #print("x"*20)
        #print(info["action"], info["reward"], info["done"])

        terminated = done
        truncated = False
        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
    
    def get_action_mask(self):
    # Returns a boolean mask: True for valid actions, False for invalid
        return (self.relu_status == 1)