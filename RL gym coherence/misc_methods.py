import numpy as np
import gym
import math
from qtable_agent import QTableAgent
from DQNAgent import DQNAgent
import torch
from torch_geometric.data import Data
import torch.nn as nn

### train_dqn
NUM_NON_ZERO_REWARDS = 0
def one_hot_state(state, env):
    state_arr = np.zeros(env.observation_space.n)
    state_arr[state] = 1
    return state_arr

def train_dqn(env_name="CartPole-v1", episodes=500, epsilon_start=1.0, epsilon_final=0.01, 
              epsilon_decay=500, reward_function = None, verbose = False, return_reward = False, 
              print_every=50, **kwargs):
    """
    Train a DQN agent on the specified environment.
    
    Args:
        env_name: str
            Name of the environment to train the agent on.
        episodes: int
            Number of episodes to train the agent for.
        epsilon_start: float
            Initial epsilon value for epsilon-greedy action selection.
        epsilon_final: float
            Final epsilon value for epsilon-greedy action selection.
        epsilon_decay: float
            Decay rate for epsilon.
        reward_function: function
            Optional reward function to use for training.
        verbose: bool
            Whether to print training progress.

    Returns:
        DQNAgent: trained DQN agent. 
    """
    global NUM_NON_ZERO_REWARDS
    env = gym.make(env_name)
    if len(env.observation_space.shape) == 0:
        state_dim = env.observation_space.n
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, **kwargs)
    
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)
    
    rewards = np.zeros(episodes) 
    is_state_discrete = hasattr(env.observation_space, 'n')
    for episode in range(episodes):
        state = env.reset() # Reset the environment, reward
        if is_state_discrete and state_dim == env.observation_space.n:
            state = one_hot_state(state, env)
        episode_reward = 0
        while True:
            epsilon = epsilon_by_frame(episode)
            # One-hot encode the state
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if is_state_discrete and state_dim == env.observation_space.n:
                next_state = one_hot_state(next_state, env)

            if reward_function: #custom reward function
                reward = reward_function(done, state, action, next_state)
            NUM_NON_ZERO_REWARDS += 0 if math.isclose(reward, 0) else 1
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            agent.update()
            
            if done:
                break
            # print(f"Episode: {episode+1}, Total reward: {episode_reward}, Epsilon: {epsilon:.2f}")

        rewards[episode] = episode_reward
        # Optional: Render the environment to visualize training progress
        if verbose and episode % print_every == print_every - 1:
        #     render_env(env, agent)
            print(f"Episode: {episode+1}, Average total reward: {np.average(rewards[episode - print_every + 1 : episode])}, Epsilon: {epsilon:.2f}")

    env.close()
    return agent if not return_reward else (agent, rewards)

# Optional: Function to render the environment with the current policy
def render_env(env, agent):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, 0)  # Using 0 epsilon for greedy action selection
        # print(env.step(action))
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state

def train_qtable(env_name="CartPole-v1", episodes=500, epsilon_start=1.0, epsilon_final=0.01, 
              epsilon_decay=500, reward_function = None, verbose = False, return_reward = False, 
              print_every=50, **kwargs):
    """
    Train a Q-table agent on the specified environment."""
    global NUM_NON_ZERO_REWARDS
    env = gym.make(env_name)
    if len(env.observation_space.shape) == 0:
        state_dim = env.observation_space.n
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = QTableAgent(state_dim, action_dim, **kwargs)

    rewards = np.zeros(episodes)
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            epsilon = epsilon_by_frame(episode)
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if reward_function:
                reward = reward_function(done, state, action, next_state)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break

        rewards[episode] = episode_reward
        if verbose and episode % print_every == print_every - 1:
            print(f"Episode: {episode+1}, Average total reward: {np.average(rewards[episode - print_every + 1 : episode])}, Epsilon: {epsilon:.2f}")
        
    env.close()
    return agent if not return_reward else (agent, rewards)
    
### test_dqn
NEAR_ZERO = 1e-9
def test_dqn(env, agent, episodes=10, reward_function=None, verbose = False):
    print(f"Maximum reward: {env.spec.reward_threshold}")
    average_value = 0
    for episode in range(episodes):
        # if episode == 0:
        #     render_env(env, agent)
        state = env.reset()
        if len(env.observation_space.shape) == 0:
            state = one_hot_state(state, env)
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, 0)  # Using 0 epsilon for greedy action selection
            next_state, reward, done, _ = env.step(action)
            if len(env.observation_space.shape) == 0:
                next_state = one_hot_state(next_state, env)
            if reward_function:
                reward = reward_function(done, state, action, next_state)
            episode_reward += reward
            state = next_state
        if verbose:
            print(f"Episode: {episode+1}, Total reward: {episode_reward}")
        average_value += episode_reward
    average_value /= episodes
    print(f"Average reward: {average_value}")
    
def test_qtable(env, agent, episodes=10, reward_function=None, verbose = False):
    """
    Test a Q-table agent on the specified environment.
    (This is basically test_dqn but without the one-hot encoding.)
    """
    print(f"Maximum reward: {env.spec.reward_threshold}")
    average_value = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, 0)  # Using 0 epsilon for greedy action selection
            next_state, reward, done, _ = env.step(action)
            if reward_function:
                reward = reward_function(done, state, action, next_state)
            episode_reward += reward
            state = next_state
        if verbose:
            print(f"Episode: {episode+1}, Total reward: {episode_reward}")
        average_value += episode_reward
    average_value /= episodes
    print(f"Average reward: {average_value}")

def nn_to_data(model: nn.Module) -> Data:
    edges = []

    # Counter for global neuron index
    idx = 0

    # Iterate over each layer in the network
    base = next(model.children())
    if isinstance(base, nn.Sequential):
        layers = list(base.children())
        layers2 = list(base.children())
    else:
        layers = list(model.children()) # iterator over the layers of the model
        layers2 = list(model.children())
    
    num_nodes = layers2[0].weight.shape[1] + sum([layer.weight.shape[0] for layer in layers2 if isinstance(layer, nn.Linear)])
    num_node_features = num_nodes
    node_features = torch.zeros(num_nodes, num_node_features)
    # shape = (num_nodes, num_node_features), where the node features are the bias of each node
    # and the weights of the edges to each node (zero if there is no edge)

    for layer in layers:
        if isinstance(layer, nn.Linear):
            # Update edges based on the weight matrix
            input_dim = layer.weight.shape[1]
            output_dim = layer.weight.shape[0]
            for i in range(input_dim):  # Input neurons (e.g. 4)
                for j in range(output_dim):  # Output neurons (e.g. 64)
                    edges.append((idx + i, idx + input_dim + j))
            
            # Update node features (e.g., biases)
            biases = torch.tensor(layer.bias.detach().numpy())
            edge_weights = torch.tensor(layer.weight.detach().numpy().T)
            node_features[idx + input_dim:idx + input_dim + output_dim, 0] = biases
            node_features[idx:idx + input_dim, 1+idx:1+idx+output_dim] = edge_weights
            node_features[idx + input_dim:idx + input_dim + output_dim, 1+idx:1+idx+input_dim] = edge_weights.T
            
            # Update the global neuron index
            idx += input_dim

    # Convert lists to PyTorch tensors
    num_nonzero = [np.count_nonzero(node_features[i]) for i in range(node_features.shape[0])]
    # print(num_nonzero)
    row_mean, row_median, row_var = torch.mean(node_features[:, 1:], dim=1), torch.median(node_features[:, 1:], dim=1)[0], torch.var(node_features[:, 1:], dim=1)
    x = torch.stack([node_features[:, 0], row_mean, row_median, row_var]).T
    # print(x.shape)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def generate_data(dataset1, dataset2):
        indices = np.random.permutation(len(dataset1) + len(dataset2))
        data = [dataset1[i] if i < len(dataset1) else dataset2[i - len(dataset1)] for i in indices]
        for i in range(len(data)):
            data[i].y = 1.0 if indices[i] < len(dataset1) else 0.0 # Binary labels for each node; 1 = URS, 0 = UPS
            # Hence roughly speaking, 1 = more coherent, 0 = less coherent

        train_data_ratio = 0.8
        train_data, test_data = data[:int(train_data_ratio * len(data))], data[int(train_data_ratio * len(data)):]
        num_node_features = data[0].x.shape[1] # Number of features for each node
        return train_data, test_data, num_node_features

def train_classifier(model, criterion, optimizer, train_data, test_data, epochs = 40, patience = 3, 
                     epochs_without_improvement = 0, best_loss = float('inf'), verbose = True):
    for epoch in range(epochs):
        avg_train_loss = 0
        for datapt in train_data:
            model.train()
            optimizer.zero_grad()

            # print(f"datapt.x shape: {datapt.x.shape}")  # Should be [num_nodes, num_node_features]
            # print(f"datapt.edge_index shape: {datapt.edge_index.shape}")  # Should be [2, num_edges]
            out = model.forward(datapt)
            # print(out.size())
            # print(torch.tensor([[datapt.y]]).size())
            loss = criterion(out, torch.tensor([[datapt.y]]))  # Adjust shape as necessary
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_data)

        avg_test_loss = 0
        for datapt in test_data:
            model.eval()
            with torch.no_grad():
                out = model.forward(datapt)
                loss = criterion(out, torch.tensor([[datapt.y]]))
                avg_test_loss += loss.item()
        avg_test_loss /= len(test_data)
        
        if verbose:
            print(f'Epoch {epoch+1}: Average Train Loss: {avg_train_loss}, Average Test Loss: {avg_test_loss}')
        
        # Early Stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if verbose and epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
    metrics = {'train_loss': avg_train_loss, 'test_loss': avg_test_loss}
    return metrics