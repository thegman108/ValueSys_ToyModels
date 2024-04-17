import gym
import random
import numpy as np
from DQNAgent import DQNAgent
from misc_methods import *
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from NN_GNNs import GraphLevelGCN, GATGraphLevelBinary
import matplotlib.pyplot as plt
import os

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

    RENEW_MODELS = False
    # Change directory to the current file's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Check if the directory exists, if not create it
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(torch.tensor([1, 2, 3]), "models/test.pt")

    if not RENEW_MODELS and os.path.exists(f"models/{env_name}_GCN_0.pt"):
        i = 0
        models = []
        while os.path.exists(f"models/{env_name}_GCN_{i}.pt"):
            models.append(torch.load(f"models/{env_name}_GCN_{i}.pt"))
            i += 1
    else:
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
        # print(URS_data[0].x.shape)
        UPS_data = [nn_to_data(agent.model) for agent in UPS_agents]
        assert URS_data[0].x.shape == UPS_data[0].x.shape

        # Binary classification between two datasets
        dataset1 = USS_data
        dataset2 = URS_data

        NUM_CLASSIFIER_TRIES = 40
        threshold = 0.2
        models, test_losses = [], []
        train_data, test_data, num_node_features = generate_data(dataset1, dataset2)
        for i in tqdm(range(NUM_CLASSIFIER_TRIES)):
            # Loss and optimizer
            model = GraphLevelGCN(num_node_features)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            metrics = train_classifier(model, criterion, optimizer, train_data, test_data, 
                                    epochs = 80, patience = 5, verbose = False)
            if metrics['test_loss'] < threshold:
                models.append(model)
                test_losses.append(metrics['test_loss'])
        model = models[np.argmin(test_losses)]
        print(f"Successful test losses: {test_losses}")
        for i in range(len(models)):
            torch.save(models[i], f"models/{env_name}_GCN_{i}.pt")

    # Test GCN model on a "more powerful" NN
    print(model.forward(dataset1[0]))
    print(model.forward(dataset2[0]))
    
    # [test_dqn(env, train_dqn(env_name = env_name, episodes = i)) for i in [10, 30, 100] * 5]
    train_dqn_data = np.zeros((3, 5 * len(models)))
    for j in tqdm(list(range(len(models) * 5))):
        powerful_models = [nn_to_data(train_dqn(env_name = env_name, episodes = i).model) 
                        for i in [5, 15, 50]]
        # print([model.forward(data) for data in powerful_models])
        train_dqn_data[:, j] = np.array(
            [models[j % len(models)].forward(data).item() for data in powerful_models]
        )
    
    plt.figure()
    plt.boxplot(train_dqn_data.T, labels = ['5 episodes', '15 episodes', '50 episodes'])
    plt.title("GCN classification of CartPole NNs")
    plt.ylabel("Classifier output")
    plt.ylim(-0.1, 1.1)
    plt.show()
