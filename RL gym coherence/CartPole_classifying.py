import gym
import random
import numpy as np
from DQNAgent import DQNAgent
from misc_methods import *
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from NN_GNNs import GraphLevelGCN, GATGraphLevelBinary

# Dataset generation
env_name = "CartPole-v1"
env = gym.make(env_name)
NEAR_ZERO = 1e-9
NUM_REWARD_CALLS = 0
NUM_NON_ZERO_REWARDS = 0
def deterministic_random(*args, lb = -1, ub = 1, sparsity = 0.0, continuous = False):
    """
    Create a deterministic random number generator for a given set of arguments.
    Used to generate deterministic reward functions for the coherence classifier.
    [Edit 4/3/24: adapted to continuous state space]"""
    global NUM_REWARD_CALLS
    NUM_REWARD_CALLS += 1
    unique_seed = f"{args}".encode("utf-8")
    random.seed(unique_seed)
    return random.uniform(lb, ub) if random.random() > sparsity else random.uniform(-NEAR_ZERO, NEAR_ZERO)

### Define and train GCN classifier on NNs

def get_state_shape(env):
    return 1 if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]
def get_state_size(env):
    return env.observation_space.n if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]

if __name__ == '__main__':   
    NUM_TRAIN_R_FUNCS = 50
    NUM_EPS_TRAIN_R = 50
    URS_r_funcs = [lambda *args: deterministic_random(args) for _ in range(NUM_TRAIN_R_FUNCS)]
    URS_agents = [train_dqn(env_name = env_name, 
                            episodes=NUM_EPS_TRAIN_R, reward_function=r_func) for r_func in tqdm(URS_r_funcs)]
    USS_r_funcs = [lambda *args: deterministic_random(args, sparsity=0.99) for _ in range(NUM_TRAIN_R_FUNCS)]
    USS_agents = [train_dqn(env_name = env_name, 
                            episodes=NUM_EPS_TRAIN_R, reward_function=r_func) for r_func in tqdm(USS_r_funcs)]
    UPS_agents = [DQNAgent(get_state_size(env), env.action_space.n) for _ in range(NUM_TRAIN_R_FUNCS)]
    # Training loop
    USS_data = [nn_to_data(agent.model) for agent in USS_agents]
    URS_data = [nn_to_data(agent.model) for agent in URS_agents]
    print(URS_data[0].x.shape)
    UPS_data = [nn_to_data(agent.model) for agent in UPS_agents]
    assert URS_data[0].x.shape == UPS_data[0].x.shape

    # Binary classification between two datasets
    dataset1 = USS_data
    dataset2 = URS_data

    train_data, test_data, num_node_features = generate_data(dataset1, dataset2)
    # Loss and optimizer
    model = GraphLevelGCN(num_node_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_classifier(model, criterion, optimizer, train_data, test_data, epochs = 100, patience = 5)

    # Test GCN model on a "more powerful" NN
    print(model.forward(dataset1[0]))
    print(model.forward(dataset2[0]))
    powerful_models = [nn_to_data(train_dqn(env_name = env_name, episodes = 5 * i).model) 
                    for i in [1, 3, 10]]
    print([model.forward(data) for data in powerful_models])