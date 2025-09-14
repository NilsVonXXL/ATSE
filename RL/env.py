import re
import gymnasium as gym
import numpy as np
import random
import pickle
import os
import json
from rl.bab_rl import bab_step
from micrograd.ibp import Interval
from micrograd.engine import Value

class BranchingEnv(gym.Env):
    def __init__(self, models_dir, dataset_dir):
        super().__init__()
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir
        self.relu_dim = 32
        self.weights_dim = 337 
        self.inputs_dim = 3

        self.action_space = gym.spaces.Discrete(self.relu_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.weights_dim + self.inputs_dim + self.relu_dim,),
            dtype=np.float32
        )

        # Gather all (model_path, input_folder) pairs
        self.initial_states = []
        for domain in os.listdir(self.dataset_dir):
            domain_path = os.path.join(self.dataset_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            for data_subdir in os.listdir(domain_path):
                data_path = os.path.join(domain_path, data_subdir)
                if not os.path.isdir(data_path):
                    continue
                for net_num in os.listdir(data_path):
                    net_path = os.path.join(data_path, net_num)
                    if not os.path.isdir(net_path):
                        continue
                    weights_path = os.path.join(net_path, 'parameters.pkl')
                    if not os.path.exists(weights_path):
                        continue
                    input_folders = [f for f in os.listdir(net_path) if f.startswith('input-x-')]
                    for input_folder in input_folders:
                        input_path = os.path.join(net_path, input_folder)
                        if os.path.isdir(input_path):
                            self.initial_states.append((weights_path, input_path, input_folder))

        self.model = None
        self.inputs = None
        self.splits = None
        self.in_bounds = None
        self.score = None
        self.state = None
        self.relu_status = None

    def reset(self, seed=None, options=None):
        # Randomly select a model and input
        chosen_model, chosen_input, _ = random.choice(self.initial_states)

        # Load model weights
        with open(chosen_model, 'rb') as f:
            self.model = pickle.load(f)
        weights_flat = np.array([w.data for w in self.model], dtype=np.float32)

        # Parse inputs from folder name
        match = re.match(r'input-x-([-\d.]+)-y-([-\d.]+)-eps-([-\d.]+)', os.path.basename(chosen_input))
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            eps = float(match.group(3))
            self.inputs = [x, y, eps]
        else:
            self.inputs = [0.0, 0.0, 0.0]
        inputs_vec = np.array(self.inputs, dtype=np.float32)

        tmp = [Value(x), Value(y)]
        self.in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in tmp}
        # Initial bounds and splits
        self.splits = dict()
        self.score = ...      # TODO: set output node
        relu_nodes = ...      # TODO: get initial relu nodes
        self.relu_status = relu_nodes

        # Build initial state
        self.state = np.concatenate([weights_flat, inputs_vec, relu_nodes])
        return self.state, {}

    def step(self, action):
        # Call bab_step
        next_splits, next_relu_status, done = bab_step(self.score, self.in_bounds, self.splits, action)
        self.splits = next_splits
        self.relu_status = next_relu_status

        # Build next state
        weights_flat = np.array([w.data for w in self.model], dtype=np.float32)
        inputs_vec = np.array(self.inputs, dtype=np.float32)
        next_state = np.concatenate([weights_flat, inputs_vec, self.relu_status])
        self.state = next_state

        reward = -1  # Each split gets -1 reward

        return self.state, reward, done, False, {}

    def render(self):
        pass

    def close(self):
        pass