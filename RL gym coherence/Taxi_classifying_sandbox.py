# %%
import gym
import random
import numpy as np
from qtable_agent import QTableAgent
from misc_methods import *
from NN_GNNs import GraphLevelGCN, GATGraphLevelBinary
from FCNNBinary import FCNNBinary
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# Now classifying q-table agents
env_name = "Taxi-v3"
NEAR_ZERO = 1e-9
NUM_EPS_TRAIN_R = 1000
NUM_TRAIN_R_FUNCS = 50
NUM_REWARD_CALLS = 0
NUM_CLASSIFIER_TRIES = 40
env = gym.make(env_name)
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
def get_state_shape(env):
    return 1 if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]
def get_state_size(env):
    return env.observation_space.n if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]
    
def qtable_to_feat(qtable: torch.Tensor, label):
    # In qtable, rows are states and columns are actions taken in that state
    return Data(x = torch.flatten(qtable), y = label) # Naive approach


def generate_fcnn_data(dataset1, dataset2):
    indices = np.random.permutation(len(dataset1) + len(dataset2))
    data = [dataset1[i] if i < len(dataset1) else dataset2[i - len(dataset1)] for i in indices]
    for i in range(len(data)):
        data[i].y = 1.0 if indices[i] < len(dataset1) else 0.0 # Binary labels for each node; 1 = URS, 0 = UPS
        # Hence roughly speaking, 1 = more coherent, 0 = less coherent

    train_data_ratio = 0.8
    train_data, test_data = data[:int(train_data_ratio * len(data))], data[int(train_data_ratio * len(data)):]
    num_node_features = data[0].x.shape[0] # Number of features for each node
    return train_data, test_data, num_node_features

def train_fcnn_classifier(model, criterion, optimizer, train_data, test_data, epochs = 40, patience = 3, 
                          epochs_without_improvement = 0, best_loss = float('inf')):
    for epoch in range(epochs):
        avg_train_loss = 0
        for datapt in train_data:
            model.train()
            optimizer.zero_grad()

            out = model.forward(datapt)
            assert isinstance(out, torch.Tensor), f"Expected model.forward to return a tensor, but got {out}"
            loss = criterion(out, torch.tensor([datapt.y]))  # Adjust shape as necessary
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_data)

        avg_test_loss = 0
        for datapt in test_data:
            model.eval()
            with torch.no_grad():
                out = model.forward(datapt)
                loss = criterion(out, torch.tensor([datapt.y]))
                avg_test_loss += loss.item()
        avg_test_loss /= len(test_data)
        
        print(f'Epoch {epoch+1}: Average Train Loss: {avg_train_loss}, Average Test Loss: {avg_test_loss}')
        
        # Early Stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

def generate_UQS_qagent(rand_qtable, gamma, env: gym.Env, episodes = 500):
    """
    Train a Q-table agent based on a reward function uniformly sampled from the set of 
    possible reward functions compatible with the given random Q-table."""
    # Generate the reward function using the Bellman equation
    r_table = np.zeros(rand_qtable.shape)
    for s in range(rand_qtable.shape[0]):
        for a in range(rand_qtable.shape[1]):
            env.reset()
            env.unwrapped.s = s
            ns = env.step(a)[0]
            r_table[s, a] = rand_qtable[s, a] - gamma * np.max(rand_qtable[ns]) #assuming greedy policy
    
    # Train the agent
    r_func = lambda s, a, *args: r_table[s, a]
    return train_qtable(env_name = env.spec.id, episodes = episodes, reward_function = r_func)


def generate_UVS_qagent(rand_values, gamma, env: gym.Env, episodes = 500, lb = -1, ub = 1):
    """
    Train a Q-table agent based on a reward function uniformly sampled from the set of 
    possible reward functions compatible with the given values for each state.
    Assumes a uniform distribution between [lb, ub]."""
    r_table = np.zeros((len(rand_values), env.action_space.n))
    for s in range(len(rand_values)):
        next_states = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            env.reset()
            env.unwrapped.s = s
            next_states[a] = env.step(a)[0]
        #v(s) = max_a(R(s, a) + gamma * v(s'))
        reward_ub = rand_values[s] - np.array([gamma * rand_values[int(ns)] for ns in next_states])
        taut_probs = np.zeros(env.action_space.n)
        for i in range(env.action_space.n):
            all_except_i = np.delete(np.arange(env.action_space.n), i)
            taut_probs[i] = np.prod((reward_ub[all_except_i] + 1) / 2)
            # probability that all other rewards at action j are less than reward_ub[j]
        taut_probs /= np.sum(taut_probs)
        taut = np.random.choice(env.action_space.n, p = taut_probs) 
        #index of the action where the reward is equal to the maximum

        rewards = np.full(env.action_space.n, float('inf'))
        while np.any(rewards >= reward_ub): #while any of the rewards are greater than the upper bound
            rewards = np.random.uniform(-1, 1, env.action_space.n)
        rewards[taut] = reward_ub[taut]
        r_table[s] = rewards
    
    r_func = lambda s, a, *args: r_table[s, a]
    return train_qtable(env_name = env.spec.id, episodes = episodes, reward_function = r_func)

# UVS_agents = [generate_UVS_qagent(np.random.uniform(-1, 1, env.unwrapped.s), 0.9, env, episodes = NUM_EPS_TRAIN_R) for _ in range(NUM_TRAIN_R_FUNCS)]
# this currently takes way too long so it has been commented out

### Only attaching reward to terminal states (kind of like UUS? but with the inductive biases)

def det_rand_terminal(done: bool, *args, lb = -1, ub = 1, sparsity = 0.0):
    """
    Create a deterministic random number generator for a given set of arguments.
    Used to generate deterministic reward functions for the coherence classifier. """
    global NUM_REWARD_CALLS
    NUM_REWARD_CALLS += 1
    if not done:
        return random.uniform(-NEAR_ZERO, NEAR_ZERO)
    unique_seed = f"{args}".encode("utf-8")
    random.seed(unique_seed)
    return random.uniform(lb, ub) if random.random() > sparsity else random.uniform(-NEAR_ZERO, NEAR_ZERO)

def greedy_policy(q_table):
    return torch.tensor(np.argmax(q_table, axis=1).reshape(-1, 1).astype(np.float32))
def random_policy(state_dim):
    return torch.randint(0, 6, (state_dim, 1)).float()
def prep_qtable(q_table):
    return torch.tensor(q_table, dtype=torch.float32)

class Node:
    def __init__(self, state, parent=None, action=None, q_values=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.children = []
        self.visits = 1  # Initialize to avoid division by zero
        self.value = 0
        self.q_values = q_values  # This should be a dictionary or similar structure

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self, env):
        return len(self.children) == env.action_space.n

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def rollout_policy(state, q_table, env):
    # Use the Q-table to select the best action if this state has been seen
    if state in q_table:
        return np.argmax(q_table[state])
    else:
        # Otherwise, select a random action
        return env.action_space.sample()

def selection(node, env):
    while not node.is_fully_expanded(env):
        if not node.children:
            return expansion(node, env)
        else:
            node = node.best_child()
    return node

def expansion(node, env):
    tried_actions = [child.action for child in node.children]
    for action in range(env.action_space.n):
        if action not in tried_actions:
            env.env.s = node.state  # Set environment to current node's state
            next_state, _, _, _ = env.step(action)
            new_node = Node(next_state, parent=node, action=action, q_values=node.q_values)
            node.add_child(new_node)
            return new_node
    return node  # In case all actions were tried

def simulation(node, env, max_steps=100):
    total_reward = 0
    current_state = node.state
    steps = 0

    while steps < max_steps:
        action = rollout_policy(current_state, node.q_values, env)
        env.env.s = current_state
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        current_state = next_state
        steps += 1
        if done:
            break

    return total_reward

def backpropagation(node, reward):
    while node is not None:
        node.update(reward)
        node = node.parent

def mcts(root, env, iterations=1000):
    for _ in range(iterations):
        leaf = selection(root, env)
        reward = simulation(leaf, env)
        backpropagation(leaf, reward)

### Test MCTS

def choose_action(node):
    # Choose the child with the highest visit count
    if node.children:
        return max(node.children, key=lambda child: child.visits).action
    else:
        return None

def simulate_episode_from_root(env, root_node):
    total_reward = 0
    done = False
    current_node = root_node
    env.reset()
    env.env.s = current_node.state
    
    while not done and current_node is not None:
        action = choose_action(current_node)
        if action is None:
            # No more information in the tree; choose random action
            action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)  # Execute the chosen action
        total_reward += reward
        
        # Move to the next node in the tree, if it exists
        next_node = None
        for child in current_node.children:
            if child.action == action:
                next_node = child
                break
        current_node = next_node

    return total_reward

# %%
if __name__ == '__main__':
    # %%
    ### Turn the state and action space of Taxi-v3 into a graph
    from collections import defaultdict
    taxi_env = gym.make("Taxi-v3")
    taxi_env.reset()
    # Initialize containers for graph data
    edges = defaultdict(list)
    edge_attr = defaultdict(list)

    # A helper function to encode the state into a single number (node index)
    def state_to_node(taxi_row, taxi_col, pass_loc, dest_idx):
        # This encoding assumes specific knowledge about the Taxi-v3 state space size
        return taxi_row * 100 + taxi_col * 20 + pass_loc * 4 + dest_idx
        # max = 4 * 100 + 4 * 20 + 4 * 4 + 3 = 400 + 80 + 16 + 3 = 499

    # Iterate through all possible states and actions to construct the graph
    for taxi_row in range(5):
        for taxi_col in range(5):
            for pass_loc in range(5):  # 4 locations + 1 for 'in taxi'
                for dest_idx in range(4):
                    current_state = state_to_node(taxi_row, taxi_col, pass_loc, dest_idx)
                    for action in range(taxi_env.action_space.n):
                        # Set the environment to the current state
                        taxi_env.unwrapped.s = current_state
                        # Take action and observe the next state and reward
                        next_state, reward, done, _ = taxi_env.step(action)
                        # Add edge from current state to next state
                        edges[current_state].append(next_state)
                        # Optionally, use rewards as edge attributes
                        # edge_attr[(current_state, next_state)].append(reward)
                        taxi_env.reset()

    # Convert edges and edge attributes to tensors
    edge_index = []
    for src, dsts in edges.items():
        for dst in dsts:
            edge_index.append([src, dst])
    edge_index = torch.tensor(edge_index).t().contiguous()

    # Change directory to the current file's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Check if the directory exists, if not create it
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = "models/Taxi_GCN_sparsity_0.pt"
    num_node_features = 1 # 1 if policy, 6 if q-table

    MAKE_NEW_MODELS = True
    if os.path.exists(model_path) and not MAKE_NEW_MODELS: # if classifiers have already been trained
        i = 0
        models = []
        while os.path.exists(f"models/Taxi_GCN_sparsity_{i}.pt"):
            model = GraphLevelGCN(num_node_features)
            model.load_state_dict(torch.load(f"models/Taxi_GCN_sparsity_{i}.pt"))
            models.append(model)
            i += 1
        print(f"Loaded {len(models)} models")
    else:
        # Generate the dataset
        base = 10
        # NUM_TRAIN_R_FUNCS = 1
        # NUM_EPS_TRAIN_R = 1
        sparsities = 1 - np.logspace(-3, 0, 2 * NUM_TRAIN_R_FUNCS, base = base) 
        # asymptotically close to 1; 10^-3 to 10^0
        np.random.shuffle(sparsities)
        r_funcs = [lambda *args: deterministic_random(args, sparsity=s) for s in sparsities]
        USS_agents = [train_qtable(env_name = env_name, episodes=NUM_EPS_TRAIN_R,
                                    reward_function = r_func) for r_func in tqdm(r_funcs)]
        
        labels = np.log10(1 - sparsities, dtype=np.float32) / float(np.log10(base)) / 3 + 1
        train_test_split = int(0.8 * len(sparsities))
        # %%
        train_data = [
            Data(x = greedy_policy(USS_agents[i].q_table), edge_index = edge_index, y = labels[i]) 
            for i in range(train_test_split) 
            # labeled with the sparsity of the r_func the policy was trained on
        ]
        test_data = [
            Data(x = greedy_policy(USS_agents[i].q_table), edge_index = edge_index, y = labels[i]) 
            for i in range(train_test_split, len(sparsities)) 
            # labeled with the sparsity of the r_func the policy was trained on
        ]
        num_node_features = 1
        # %%
        train_data = [
            Data(x = prep_qtable(USS_agents[i].q_table), edge_index = edge_index, y = labels[i])
            for i in range(train_test_split) 
            # labeled with the sparsity of the r_func the policy was trained on
        ]
        test_data = [
            Data(x = prep_qtable(USS_agents[i].q_table), edge_index = edge_index, y = labels[i])
            for i in range(train_test_split, len(sparsities)) 
            # labeled with the sparsity of the r_func the policy was trained on
        ]
        num_node_features = env.action_space.n
        models, test_losses = [], []
        THRESHOLD = 0.01
        for _ in tqdm(range(NUM_CLASSIFIER_TRIES)):
            model = GraphLevelGCN(num_node_features)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            metrics = train_classifier(
                model, criterion, optimizer, train_data, test_data, epochs = 80, patience = 5,
                verbose = True
            )
            if metrics['test_loss'] < THRESHOLD:
                test_losses.append(metrics['test_loss'])
                models.append(model)
        print(f"Successful GCN classifier test losses: {test_losses}")
        # Choose the model with the lowest test loss
        model = models[np.argmin(test_losses)]
        print(f"Number of successful models: {len(models)}")
        for i in range(len(models)):
            torch.save(models[i].state_dict(), f"models/Taxi_GCN_sparsity_{i}.pt")
        
        print(torch.transpose(test_data[0].x, 0, -1)[0:10, 0:10]) # Example greedy policies        print(f"Sample dataset1 classification: {model.forward(dataset1[0])}")
        print(f"Sample test classification: {model.forward(test_data[0])}")
    
    # Caveat: when models are loaded, model is the last model in the list, 
    # not the model with the lowest test loss

    # %%
    interval = 100
    total_taxi_eps = 20000
    taxi_classifier_ratings = np.zeros((len(models) * 5, interval))
    for i in tqdm(list(range(len(models) * 5))):
        taxi_model = None
        for j in range(interval):
            taxi_model = train_qtable(
                env_name = "Taxi-v3", episodes = int(total_taxi_eps / interval), verbose = False, 
                # print_every = total_taxi_eps / 10, 
                return_reward = False, pretrained_agent = taxi_model
            )
            taxi_data = Data(x = greedy_policy(taxi_model.q_table).detach(), edge_index = edge_index)

            out = models[i % len(models)].forward(taxi_data).item()
            # Recover original sparsity: 1 - base ** (3 * (out - 1))
            taxi_classifier_ratings[i][j] = out
        # test_qtable(gym.make("Taxi-v3"), taxi_model, episodes = 1000)
        # Generate tabular policy from MCTS and feed through classifier

    # %%
    plt.figure()
    x_axis = np.arange(0, total_taxi_eps, total_taxi_eps / interval)
    for i in range(len(models) * 5):
        plt.scatter(x_axis, taxi_classifier_ratings[i])
    # print(x_axis, taxi_classifier_ratings)
    plt.xlabel("Episodes")
    plt.ylabel("Classifier output")
    plt.title("GCN classification of Taxi NNs")
    # plt.legend()
    plt.show()
