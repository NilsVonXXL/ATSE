import re
import gymnasium as gym
import numpy as np
import random
import pickle
import os
import json
from collections import deque
from rl.bab_rl import bab_step, Branch, collect_relu_nodes
from micrograd.ibp import Interval, ibp
from micrograd.engine import Value
from micrograd.nn import MLP

class DeepThought42(gym.Env):
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

        self._step_count = 0
        self._reward = -1

        # Gather all (model_path/params_path, input_folder, is_params) tuples
        self.initial_states = []
        use_models_dir = self.models_dir is not None and os.path.exists(self.models_dir)

        # Support both old and new dataset structures
        for domain in os.listdir(self.dataset_dir):
            domain_path = os.path.join(self.dataset_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            # Try to detect if this is a 'data-*' or 'random_*' folder (old vs new)
            subdirs = os.listdir(domain_path)
            # If any subdir starts with 'data-', treat as old structure
            if any(s.startswith('data-') for s in subdirs):
                for data_subdir in subdirs:
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
                                if use_models_dir:
                                    self.initial_states.append((weights_path, input_path, input_folder, False))
                                else:
                                    self.initial_states.append((weights_path, input_path, input_folder, True))
            else:
                # New structure: domain_path/random_*/input-x-*
                for net_num in subdirs:
                    net_path = os.path.join(domain_path, net_num)
                    if not os.path.isdir(net_path):
                        continue
                    weights_path = os.path.join(net_path, 'parameters.pkl')
                    if not os.path.exists(weights_path):
                        continue
                    input_folders = [f for f in os.listdir(net_path) if f.startswith('input-x-')]
                    for input_folder in input_folders:
                        input_path = os.path.join(net_path, input_folder)
                        if os.path.isdir(input_path):
                            if use_models_dir:
                                self.initial_states.append((weights_path, input_path, input_folder, False))
                            else:
                                self.initial_states.append((weights_path, input_path, input_folder, True))

        if not self.initial_states:
            raise RuntimeError(f"No valid initial states found in dataset_dir '{self.dataset_dir}'.\n"
                               f"Checked for both old and new dataset structures.\n"
                               f"Please ensure your dataset is populated and the directory structure is correct.")

        self.model = None
        self.inputs = None
        self.splits = None
        self.in_bounds = None
        self.score = None
        self.state = None
        self.relu_status = None

        # Branch queue for BFS tree traversal
        self.branch_queue = deque()
        self.current_branch = None
        
        print(f"DeepThought42 initialized with {len(self.initial_states)} initial states.")

    def reset(self, seed=None, options=None):
        weights_path, input_path, input_folder, is_params = random.choice(self.initial_states)
        self._step_count = 0  # Reset step counter at the start of each episode

        #print(f"Resetting environment with input_path: {input_path}")
        
        if not is_params:
            # Use model file from models_dir
            weights_path = os.path.normpath(weights_path)
            parts = weights_path.split(os.sep)
            domain = parts[-4]
            net_num = parts[-2]
            model_name = f"model_{domain}_{net_num}.pkl"
            model_path = os.path.join(self.models_dir, model_name)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            # Reconstruct model from parameters.pkl
            self.model = MLP(2, [16, 16, 1])
            with open(weights_path, 'rb') as f:
                params = pickle.load(f)
            for p_model, p_loaded in zip(self.model.parameters(), params):
                p_model.data = p_loaded.data

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

        # Initial splits and bounds
        self.splits = dict()
        self.score = self.model(self.in_bounds)
        self.node_bounds = ibp(self.score, self.in_bounds, return_all=True)

        # Branch queue logic (BFS)
        relu_nodes_list, relu_indexes = collect_relu_nodes(self.score, self.node_bounds, self.splits.keys())
        root_branch = Branch(splits=self.splits.copy(), node_bounds=self.node_bounds.copy(), relu_indexes=relu_indexes.copy())
        self.branch_queue = deque()
        self.current_branch = root_branch

        # Build initial state from current_branch
        self.state = np.concatenate([
            self.weights_flat,
            self.inputs_vec,
            np.array(list(self.current_branch.relu_indexes.values()), dtype=np.float32)
        ])

        info = {
            "score": float(self.score.data),
            # Only include relu_status as a numpy array (not relu_indexes dict)
            "relu_status": self.relu_status.copy()
        }
        return self.state, info

    def step(self, action):

        # BFS: use a queue for branch traversal
        done = False
        children, done = bab_step(self.score, self.in_bounds, self.current_branch, action)
        reward = -1
        info = {}
        self._step_count += 1

        # Debug print for BFS queue and branch info
        #print(f"[BFS DEBUG] Step: {self._step_count}, Queue size: {len(self.branch_queue)}, "
        #      f"Current branch depth: {len(self.current_branch.splits) if self.current_branch else 'None'}, Action: {action}, Done: {done}")

        if len(children) == 0:
            # Both children pruned or invalid action
            if len(self.branch_queue) > 0:
                self.current_branch = self.branch_queue.popleft()
            else:
                self.current_branch = None
                done = True
        else:
            # Valid split: enqueue all children (BFS)
            self.branch_queue.extend(children)
            self.current_branch = self.branch_queue.popleft() if len(self.branch_queue) > 0 else None

        # If queue is empty and no current_branch left, done
        if self.current_branch is None and len(self.branch_queue) == 0:
            done = True

        # Update relu_status for observation
        if self.current_branch is not None:
            self.relu_status = np.array(list(self.current_branch.relu_indexes.values()), dtype=np.float32)
        else:
            self.relu_status = np.zeros(self.relu_dim, dtype=np.float32)
                    

        # Build next state
        next_state = np.concatenate([
            self.weights_flat,
            self.inputs_vec,
            self.relu_status
        ])
        self.state = next_state


        # Only include picklable types in info
        info = {
            # Optionally, just include the number of splits (branch depth) for debug
            "branch_depth": len(self.current_branch.splits) if self.current_branch else 0,
            "relu_status": self.relu_status.copy(),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done)
        }

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