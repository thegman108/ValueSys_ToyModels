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
from functools import partial
import pickle


# Now classifying q-table agents
env_name = "Taxi-v3"
NEAR_ZERO = 1e-9
NUM_EPS_TRAIN_R = 10000
NUM_TRAIN_R_FUNCS = 50
NUM_REWARD_CALLS = 0
NUM_CLASSIFIER_TRIES = 40
env = gym.make(env_name)
def deterministic_random(*args, lb = -1, ub = 1, sparsity = 0.0, continuous = False, seed: str = ""):
    """
    Create a deterministic random number generator for a given set of arguments.
    Used to generate deterministic reward functions for the coherence classifier.
    """
    global NUM_REWARD_CALLS
    NUM_REWARD_CALLS += 1
    unique_seed = (seed + f"{args}").encode("utf-8") # Unique seed for each set of arguments
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

def det_rand_terminal(done: bool, *args, lb = -1, ub = 1, sparsity = 0.0, seed: str = ""):
    """
    Create a deterministic random number generator for a given set of arguments.
    Used to generate deterministic reward functions for the coherence classifier. 
    Test: deterministic_random(10, seed = "6742") = det_rand_terminal(True, 10, seed = "6742") = -0.18942340676023695
    """
    global NUM_REWARD_CALLS
    NUM_REWARD_CALLS += 1
    unique_seed = (seed + f"{args}").encode("utf-8")
    random.seed(unique_seed) # seed before sampling reward
    if not done:
        return random.uniform(-NEAR_ZERO, NEAR_ZERO)
    return random.uniform(lb, ub) if random.random() > sparsity else random.uniform(-NEAR_ZERO, NEAR_ZERO)

def greedy_policy(q_table):
    """Create a greedy policy from a Q-table.
    Fix 5/8/24: in case of ties, choose randomly from the best actions."""
    out = torch.zeros(q_table.shape[0])
    for idx in range(q_table.shape[0]):
        row = q_table[idx]
        out[idx] = np.random.choice(np.where(np.isclose(row, row.max()))[0])
    return out.reshape(-1, 1)

def random_policy(state_dim):
    return torch.randint(0, 6, (state_dim, 1)).float()
def prep_qtable(q_table):
    return torch.tensor(q_table, dtype=torch.float32)

def rand_argmax(arr: np.ndarray):
    """
    Shorthand class to return argmax with random order in case of ties.
    """
    return np.random.choice(np.flatnonzero(arr == arr.max()))

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
        return self.children[rand_argmax(np.array(choices_weights))]

def rollout_policy(state, q_table, env):
    # Use the Q-table to select the best action if this state has been seen
    if state in range(q_table.shape[0]):
        # print(state)
        return rand_argmax(q_table[state])
    else:
        # Otherwise, select a random action
        # print("Random action")
        return env.action_space.sample()

def selection(node, env):
    """Select a child node from the tree until a leaf node is reached."""
    # while not node.is_fully_expanded(env):
    #     if not node.children: # If the node has no children, expand it
    #         return expansion(node, env)
    #     else:
    #         node = node.best_child()
    # return node
    if node.is_fully_expanded(env):
        return selection(node.best_child(), env)
    else:
        return expansion(node, env)

def expansion(node: Node, env: gym.Env):
    """Expand the node by adding a new child node for an untried action."""
    tried_actions = [child.action for child in node.children]
    # print(tried_actions)
    actions = list(range(env.action_space.n))
    np.random.shuffle(actions)
    for action in actions:
        if action not in tried_actions:
            # print(action)
            env.reset()
            env.env.s = node.state  # Set environment to current node's state
            next_state = env.step(action)[0]
            new_node = Node(next_state, parent=node, action=action, q_values=node.q_values)
            node.add_child(new_node)
            return new_node
    return node  # In case all actions were tried

def simulation(node, env, max_steps=100, r_func = None):
    total_reward = 0
    current_state = node.state
    steps = 0

    while steps < max_steps:
        action = rollout_policy(current_state, node.q_values, env)
        env.env.s = current_state
        next_state, reward, term, trunc = env.step(action)[:4]
        done = term or trunc
        if r_func:
            reward = r_func(done, next_state)
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

def mcts(root, env, iterations=1000, r_func = None):
    for _ in range(iterations):
        leaf = selection(root, env)
        reward = simulation(leaf, env, r_func = r_func)
        backpropagation(leaf, reward)

### Test MCTS

def choose_action(node):
    # Choose the child with the highest visit count
    # if node.children:
    #     return max(node.children, key=lambda child: child.visits).action
    # else:
    #     return None
    # Alternatively: choose the child with the highest average value
    if node.children:
        return node.best_child().action
    else:
        return None

def simulate_episode_from_root(env, root_node, reward_function = None):
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
        
        next_state, reward, term, trunc = env.step(action)[:4]  
        # Execute the chosen action
        done = term or trunc
        if reward_function:
            reward = reward_function(done, next_state)
        total_reward += reward
        
        # Move to the next node in the tree, if it exists
        next_node = None
        for child in current_node.children:
            if child.action == action:
                next_node = child
                break
        current_node = next_node

    return total_reward

def extract_policy(root_node, env, baseline_policy = None):
    policy = baseline_policy[:] if not (baseline_policy is None) else np.random.randint(0, env.action_space.n, env.observation_space.n)
    # default action is random in case the state is not in the tree
    node_queue = [root_node]
    
    not_random = set()
    while node_queue:
        current_node = node_queue.pop(0)
        # print(len(current_node.children))
        if current_node.is_fully_expanded(env):
            best_action = current_node.best_child().action
            not_random.add(current_node.state)
            policy[current_node.state] = best_action
            np.random.shuffle(current_node.children)
            node_queue.extend(current_node.children)
        else:
            # If the node isn't fully expanded, we take the best action tried so far
            # This is rare in fully run MCTS but can happen if the tree isn't deep enough
            if current_node.children:
                # Find the maximum visit count
                max_visits = max(child.visits for child in current_node.children)

                # Filter actions that have the maximum visit count
                best_actions = [child.action for child in current_node.children if child.visits == max_visits]

                # Sample randomly from the best actions
                best_action = random.choice(best_actions)
                policy[current_node.state] = best_action
                node_queue.extend(current_node.children)

    num_not_random = len(not_random)
    # print(num_not_random)
    return policy, num_not_random

# %%
if __name__ == '__main__':
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
                        next_state, reward, done = taxi_env.step(action)[:3]
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
    model_path = "models/Taxi_GCN_USS_URS_"
    num_node_features = 1 # 1 if policy, 6 if q-table
    MAKE_NEW_MODELS = False
    if os.path.exists(model_path + "0.pt") and not MAKE_NEW_MODELS: # if classifiers have already been trained
        i = 0
        models = []
        while os.path.exists(model_path + f"{i}.pt"):
            model = GraphLevelGCN(num_node_features)
            model.load_state_dict(torch.load(model_path + f"{i}.pt"))
            models.append(model)
            i += 1
        print(f"Loaded {len(models)} models")
    else:
        # Desiderata for reward functions:
        # Must be deterministic for the same state and action input
        # Must be distinct by seed
        def seed_deterministic_random(seed, **kwargs):
            return partial(deterministic_random, seed = seed, **kwargs)
        UPS_agents = [QTableAgent(get_state_size(env), env.action_space.n) for _ in range(NUM_TRAIN_R_FUNCS)]
        UPS_r_funcs = [lambda *args: 0 for _ in range(NUM_TRAIN_R_FUNCS)]
        URS_rand = np.random.randint(10000, size = NUM_TRAIN_R_FUNCS)
        URS_r_funcs = [seed_deterministic_random(str(seed)) for seed in URS_rand]
        # URS_agents = [train_qtable(env_name = env_name, episodes=NUM_EPS_TRAIN_R, 
        #                         reward_function = r_func) for r_func in tqdm(URS_r_funcs)]
        print("Halfway there!")
        USS_rand = np.random.randint(10000, size = NUM_TRAIN_R_FUNCS)
        USS_r_funcs = [seed_deterministic_random(str(seed), sparsity = 0.99) for seed in USS_rand]
        # USS_agents = [train_qtable(env_name = env_name, episodes=NUM_EPS_TRAIN_R,
        #                             reward_function = r_func) for r_func in tqdm(USS_r_funcs)]
        with open('URS_storage.pickle', 'rb') as handle:
            URS_agents = pickle.load(handle)
        with open('USS_storage.pickle', 'rb') as handle:
            USS_agents = pickle.load(handle)

        # The Q-Table is already one-hot encoded, so we don't need to convert it to a Data object
        from torch_geometric.data import Data
        for agent in UPS_agents:
            for row in agent.q_table:
                for i in range(len(row)):
                    row[i] = np.random.uniform(-1, 1) # set each value to a random number between -1 and 1
        # dataset1 = [qtable_to_feat(torch.tensor(agent.q_table, dtype=torch.float32), 1) for agent in USS_agents]
        # dataset2 = [qtable_to_feat(torch.tensor(agent.q_table, dtype=torch.float32), 0) for agent in URS_agents] # URS = 1, UPS = 0
        # train_data, test_data, num_node_features = generate_fcnn_data(dataset1, dataset2)
        # print(num_node_features)
        # model = FCNNBinary(num_node_features)
        # criterion = torch.nn.BCELoss()  
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # train_fcnn_classifier(model, criterion, optimizer, train_data, test_data)
        # UQS_agents = [generate_UQS_qagent(agent.q_table, 0.9, env, episodes = NUM_EPS_TRAIN_R) for agent in UPS_agents]

        # %%
        def seed_det_rand_terminal(seed, **kwargs):
            return partial(det_rand_terminal, seed = seed, **kwargs)
        UUS_r_funcs = [seed_det_rand_terminal(str(seed)) for seed in USS_rand]
        UUS_agents = [train_qtable(
            env_name = env_name, episodes = NUM_EPS_TRAIN_R, 
            reward_function = r_func
        ) for r_func in tqdm(UUS_r_funcs)]
        # dataset2 = [Data(x = random_policy(agent.q_table.shape[0]), edge_index = edge_index, y = 0) for agent in UPS_agents]
        # ^ random_policy = UPS sampling
        # print(dataset1[0].x.shape)
        assert len(UUS_r_funcs) == len(URS_r_funcs) == len(URS_agents) == len(UUS_agents)
        assert URS_r_funcs[0](0) == URS_r_funcs[0](0)
        assert URS_r_funcs[1](0) != URS_r_funcs[0](0)
        assert URS_r_funcs[0](0) != URS_r_funcs[0](1)
        assert UUS_r_funcs[0](0) == UUS_r_funcs[0](0)
        assert UUS_r_funcs[1](0) != UUS_r_funcs[0](0)
        assert UUS_r_funcs[0](0) != UUS_r_funcs[0](1)

        # %%
        # To prevent overfitting to Q-learning inductive biases, we put some of the policies through MCTS
        dataset1 = [Data(x = greedy_policy(agent.q_table), edge_index = edge_index, y = 1) for agent in URS_agents]
        dataset2 = [Data(x = greedy_policy(agent.q_table), edge_index = edge_index, y = 0) for agent in UPS_agents] # URS = 1, UPS = 0
        for i in tqdm(range(len(dataset1))):
            agents = URS_agents if i < len(dataset1) / 2 else UPS_agents
            dataset = dataset1 if i < len(dataset1) / 2 else dataset2
            r_funcs = URS_r_funcs if i < len(dataset1) / 2 else UPS_r_funcs
            env_name = "Taxi-v3"
            env = gym.make(env_name)
            initial_state = env.reset()[0]
            root_node = Node(initial_state, q_values=agents[i % int(len(dataset1) / 2)].q_table)
            mcts(root_node, env, iterations=1000, r_func = r_funcs[i % int(len(dataset1) / 2)])

            # # Test the policy derived from the MCTS root node
            # env = gym.make('Taxi-v3')
            average_reward = np.mean([simulate_episode_from_root(env, root_node, reward_function = r_funcs[i % int(len(dataset1) / 2)]) for _ in range(100)])
            print(f"Average Reward (100 samples) from the MCTS policy: {average_reward}")
            # average_rewards.append(average_reward)
            # # test_qtable(env, taxi_model, episodes = 100)
            mcts_policy, num_not_random = extract_policy(root_node, env, baseline_policy = greedy_policy(agents[i % int(len(dataset1) / 2)].q_table).T[0].__array__())
            print(f"Number of states not generated from random (or from baseline policy): {num_not_random}")
            dataset[i % int(len(dataset1) / 2)].x = torch.tensor(mcts_policy.reshape(-1, 1).astype(np.float32))

        # %%
        # Train GCN classifier
        # dataset1 = [Data(x = greedy_policy(agent.q_table), edge_index = edge_index, y = 1) for agent in USS_agents]
        # dataset2 = [Data(x = greedy_policy(agent.q_table), edge_index = edge_index, y = 0) for agent in URS_agents]
        train_data, test_data, num_node_features = generate_data(dataset1, dataset2)
        models, test_losses = [], []
        threshold = 0.6
        epochs = 80
        NUM_CLASSIFIER_TRIES = 40
        train_losses_detail, test_losses_detail = np.zeros((NUM_CLASSIFIER_TRIES, epochs)), np.zeros((NUM_CLASSIFIER_TRIES, epochs))
        for k in tqdm(range(NUM_CLASSIFIER_TRIES)):
            model = GraphLevelGCN(num_node_features)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
            metrics, train_losses_epoch, test_losses_epoch = train_classifier(
                model, criterion, optimizer, train_data, test_data, epochs = epochs, patience = 5,
                verbose = True, return_epoch_losses = True
            )
            train_losses_detail[k, :len(train_losses_epoch)] = train_losses_epoch
            test_losses_detail[k, :len(test_losses_epoch)] = test_losses_epoch
            if metrics['test_loss'] < threshold:
                test_losses.append(metrics['test_loss'])
                models.append(model)
            print(metrics['test_loss'])
        print(f"Successful GCN classifier test losses: {test_losses}")
        # Choose the model with the lowest test loss
        model = models[np.argmin(test_losses)]
        print(f"Number of successful models: {len(models)}")
        # if MAKE_NEW_MODELS:
        #     for i in range(len(models)):
        #         torch.save(models[i].state_dict(), model_path + f"{i}.pt")
        
        print(torch.transpose(dataset1[0].x, 0, -1)[0:10, 0:10]) # Example greedy policies
        print(torch.transpose(dataset2[0].x, 0, -1)[0:10, 0:10])
        print(f"Sample dataset1 classification: {model.forward(dataset1[0])}")
        print(f"Sample dataset2 classification: {model.forward(dataset2[0])}")
    
    # Caveat: when models are loaded, model is the last model in the list, 
    # not the model with the lowest test loss

    # %%
    taxi_classifier_ratings = []
    with open('taxi_storage.pickle', 'rb') as handle:
        taxi_models = pickle.load(handle)
    # taxi_models = []
    taxi_episodes = 10000
    for i in tqdm(list(range(len(models))) * 1):
        # taxi_model = train_qtable(env_name = "Taxi-v3", episodes = taxi_episodes, verbose = False, print_every = 2000, 
        #                         return_reward = False)
        taxi_model = taxi_models[i]
        taxi_data = Data(x = greedy_policy(taxi_model.q_table).detach(), edge_index = edge_index)
        #taxi_data = Data(x = prep_qtable(taxi_model.q_table).detach(), edge_index = edge_index)

        taxi_classifier_ratings.append(models[i].forward(taxi_data).item())
        # test_qtable(gym.make("Taxi-v3"), taxi_model, episodes = 1000)
        # Generate tabular policy from MCTS and feed through classifier
        taxi_models.append(taxi_model)

    # %%
    # Train taxi agents with denser rewards
    def dense_reward(state, action, *args):
        return

    # %%
    # Assume taxi_model.q_table is your pre-trained Q-table
    # It should be a dictionary where keys are states and values are arrays of Q-values for each action

    mcts_classifier_ratings = []
    average_rewards = []
    mcts_policies = []
    for i in tqdm(list(range(len(models))) * 1):
        # Example usage
        env_name = "Taxi-v3"
        env = gym.make(env_name)
        initial_state = env.reset()[0]
        root_node = Node(initial_state, q_values=taxi_models[i].q_table)
        mcts(root_node, env, iterations=1000)

        # Test the policy derived from the MCTS root node
        env = gym.make('Taxi-v3')
        average_reward = np.mean([simulate_episode_from_root(env, root_node) for _ in range(100)])
        # print(f"Average Reward from the MCTS policy: {average_reward}")
        average_rewards.append(average_reward)
        # test_qtable(env, taxi_model, episodes = 100)
        mcts_policy, num_not_random = extract_policy(root_node, env, baseline_policy = greedy_policy(taxi_models[i].q_table).T[0].__array__())
        mcts_policies.append(mcts_policy)
        mcts_classifier_ratings.append(
            models[i].forward(
                Data(x = torch.tensor(mcts_policy.reshape(-1, 1).astype(np.float32)), edge_index = edge_index)
            ).item()
        )

    # %%
    ### A la Wentworth's definition of coherence, we create policies that do and do not "contradict"
    # themselves, i.e. there exists a value function consistent with the policy, and pass them
    # through the classifier

    c_diffs, c_ratings, ic_ratings = [], [], []
    for index in tqdm(list(range(len(models))) * 1):
        coherent_policy = greedy_policy(taxi_models[index].q_table).detach()
        incoherent_policy = coherent_policy.clone()
        
        for _ in range(300):
            env.reset()
            i = env.unwrapped.s # +100 for moving one row, + 20 for moving one column
            # print(i)
            env.step(coherent_policy[i].item())
            j = env.unwrapped.s
            if coherent_policy[i][0] % 2 == 0:
                incoherent_policy[j][0] = coherent_policy[i][0] + 1
            else:
                incoherent_policy[j][0] = coherent_policy[i][0] - 1 # if 0, then 1; if 1, then 0
            # point is to put incoherent_policy in a loop
        
        # for i in np.random.randint(500, size = 300):
        #     if incoherent_policy[i][0] >= 4:
        #         incoherent_policy[i][0] = np.random.randint(4)

        # oops, i is already taken as a variable name here
        c_diffs.append((coherent_policy != incoherent_policy).nonzero().shape[0])
        c_ratings.append(models[index].forward(Data(x = coherent_policy.detach(), edge_index = edge_index)).item())
        ic_ratings.append(models[index].forward(Data(x = incoherent_policy.detach(), edge_index = edge_index)).item())
    
    print(f"Coherent vs incoherent policy scorings: ")
    print(c_diffs)
    print(c_ratings)
    print(ic_ratings)

    class PolicyAgent:
        def __init__(self, policy, epsilon = 0.0):
            self.policy, self.epsilon = policy, epsilon
        def act(self, state, epsilon):
            if random.random() > epsilon:
                action = self.policy[state]
            else:
                action = random.randrange(self.action_dim)
            return action
        
    c_agent, ic_agent = PolicyAgent(np.array(coherent_policy.T[0])), PolicyAgent(np.array(incoherent_policy.T[0]))
    test_qtable(env, c_agent, episodes = 1000)
    test_qtable(env, ic_agent, episodes = 1000)

    # %%
    # Shapiro Wilkes test for classifier ratings
    from scipy.stats import shapiro, wilcoxon
    print(shapiro(taxi_classifier_ratings))
    print(shapiro(mcts_classifier_ratings))
    print(shapiro(ic_ratings))

    # %%
    # Convert logistic to linear ratings with logit function
    from scipy.special import logit
    binary = True
    if binary:
        taxi_classifier_ratings = [logit(x) for x in taxi_classifier_ratings]
        mcts_classifier_ratings = [logit(x) for x in mcts_classifier_ratings]
        c_ratings = [logit(x) for x in c_ratings]
        ic_ratings = [logit(x) for x in ic_ratings]

    # %%
    # Convert linear to logistic ratings
    from scipy.special import expit
    taxi_classifier_ratings = [expit(x) for x in taxi_classifier_ratings]
    mcts_classifier_ratings = [expit(x) for x in mcts_classifier_ratings]
    c_ratings = [expit(x) for x in c_ratings]
    ic_ratings = [expit(x) for x in ic_ratings]

    # %%
    # Plot the classifier ratings
    plt.figure()
    plt.boxplot(
        [taxi_classifier_ratings, mcts_classifier_ratings, ic_ratings],
        labels = ["Taxi Q-tables", "MCTS", "Incoherent"]
    )
    plt.errorbar(
        [1, 2, 3], 
        [np.mean(taxi_classifier_ratings), np.mean(mcts_classifier_ratings), np.mean(ic_ratings)],
        np.array([np.std(taxi_classifier_ratings), np.std(mcts_classifier_ratings), np.std(ic_ratings)]) * 2 / np.sqrt(len(taxi_classifier_ratings)),
        capsize = 5, 
        label = "Mean ± 2 SEM"
    ) # assuming normal distribution
    plt.title("Classifier Ratings for Different Policies")
    plt.ylabel("Classifier Output" if max(abs(np.max(taxi_classifier_ratings)), abs(np.min(taxi_classifier_ratings))) < 1.1 else "Logit Classifier Output")
    plt.legend()
    # plt.ylim(-0.1, 1.1)
    plt.show()

    # Plot classifier test losses
    # plt.figure()
    # plt.boxplot([test_losses], labels = ["GCN"])
    # plt.title("Classifier Test Losses")
    # plt.ylabel("Test Loss")
    # plt.show()

    # %%
    train_qtable_data = np.zeros((2, len(models)))
    for j in tqdm(range(len(models))):
        """
        powerful_models = [greedy_policy(train_qtable(env_name = env_name, episodes = i).q_table) 
                        for i in [1000, 3000, 10000]]
        # print([model.forward(data) for data in powerful_models])
        train_qtable_data[:, j] = np.array(
            [models[j % len(models)].forward(Data(x = data, edge_index = edge_index)).item() 
             for data in powerful_models]
        )
        """
        episodes = int(np.random.lognormal(3, 1)) * 10 # median is about e^3 = 20
        test_taxi_model = greedy_policy(train_qtable(env_name = env_name, episodes = episodes).q_table)
        train_qtable_data[0, j] = episodes
        train_qtable_data[1, j] = models[j % len(models)].forward(
            Data(x = test_taxi_model, edge_index = edge_index)
        ).item()
    
    plt.figure()
    # plt.boxplot(train_qtable_data.T, labels = ['1000 episodes', '3000 episodes', '10000 episodes'])
    plt.scatter(train_qtable_data[0], train_qtable_data[1])
    plt.ylim(-0.1, 1.1)
    plt.title("Classifier Ratings for Taxi Agents")
    plt.ylabel("Classifier Output")
    plt.xlabel("Episodes")
    plt.show()

    # %%
    # Testing linear (BCE) probes on q-tables
    class BinLogClassifier(nn.Module):
        def __init__(self, input_size):
            super(BinLogClassifier, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred
    linear_probe = BinLogClassifier(500)
    criterion = nn.BCELoss()
    test_criterion = lambda x, y: 1 if abs(x.item() - y.item()) < 0.5 else 0
    # Test accuracy
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-4)
    epochs =40
    patience = 5
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        avg_train_loss, avg_test_loss = 0, 0
        for j in range(len(train_data)):
            optimizer.zero_grad()
            out = linear_probe(train_data[j].x.reshape(1, -1))
            loss = criterion(out, torch.tensor([[train_data[j].y]]))
            avg_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss /= len(train_data)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}: Average Train Loss: {avg_train_loss}")
        
        avg_test_accuracy = 0
        for j in range(len(test_data)):
            with torch.no_grad():
                out = linear_probe(test_data[j].x.reshape(1, -1))
                print(out.item(), test_data[j].y)
                accuracy = test_criterion(out, torch.tensor([[test_data[j].y]]))
                avg_test_accuracy += accuracy
                loss = criterion(out, torch.tensor([[test_data[j].y]]))
                avg_test_loss += loss.item()
        avg_test_loss /= len(test_data)
        avg_test_accuracy /= len(test_data)
        print(f"Epoch {epoch + 1}: Average Test Accuracy: {avg_test_accuracy}")
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_accuracy)
    
    plt.figure()
    plt.plot(test_accuracies)
    plt.title("Linear Probe Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.show()

    plt.figure()
    plt.plot(test_losses, label = "Test Loss")
    plt.plot(train_losses, label = "Train Loss")
    plt.title("Linear Probe Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.show()
    
# %%
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error

    regr = LogisticRegression(max_iter = 1000).fit(
        [d.x.T[0].__array__() for d in train_data], 
        [d.y for d in train_data]
    )
    y_pred = regr.predict([d.x.T[0].__array__() for d in test_data])
    print(mean_squared_error([d.y for d in test_data], y_pred))
# %%
    plt.figure()
    for loss in train_losses_detail:
        nonzero_loss = loss[loss != 0]
        plt.plot(nonzero_loss, color='b')

    # Plot test losses without label
    for loss in test_losses_detail:
        nonzero_loss = loss[loss != 0]
        plt.plot(nonzero_loss, color='r')
    # Create custom legend handles
    train_handle = plt.Line2D([], [], color='b', label='Train Loss')
    test_handle = plt.Line2D([], [], color='r', label='Test Loss')

    plt.title("Graph Neural Network Classifier Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(handles = [train_handle, test_handle])
    plt.show()
# %%
