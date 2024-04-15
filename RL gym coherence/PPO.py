import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)  # Adding a second layer for better abstraction
        self.fc_pi = nn.Linear(128, 2)  # Action space is 2
        self.fc_v = nn.Linear(128, 1)   # Value function outputs a single scalar
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    # print(values.shape, next_value.T[0].shape)
    values = torch.concat([values, next_value.T[0]], dim=0)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return torch.tensor(returns)

def ppo_update(model, states, actions, log_probs, returns, advantages, clip_param=0.2):
    total_loss = 0
    count_steps = 0

    for _ in range(6):  # Iterate over multiple epochs
        sampler = torch.randperm(states.size(0))
        for i in range(0, states.size(0), 32):  # Batch size of 32
            indices = sampler[i:i+32]
            sampled_states = states[indices]
            sampled_actions = actions[indices]
            sampled_log_probs = log_probs[indices]
            sampled_returns = returns[indices]
            sampled_advantages = advantages[indices]

            pi, value = model(sampled_states)
            dist = Categorical(logits=pi)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(sampled_actions.squeeze())

            ratio = torch.exp(new_log_probs - sampled_log_probs)
            surr1 = ratio * sampled_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * sampled_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (sampled_returns - value.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            count_steps += 1

    return total_loss / count_steps

class PPOAgent:
    def __init__(self):
        self.model = PPO()
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        pi, _ = self.model(state)
        dist = Categorical(logits=pi)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        states, actions, log_probs, rewards, next_states, dones, values = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        values = torch.FloatTensor(values)

        _, next_values = self.model(next_states)

        returns = compute_gae(next_values, rewards, 1 - dones, values)
        advantages = torch.tensor(returns) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_update(self.model, states, actions.unsqueeze(1), log_probs.unsqueeze(1), returns.unsqueeze(1), advantages.unsqueeze(1))

        self.memory = []  # Clear memory

    def run(self, episodes, print_every: int = 20):
        env = gym.make("CartPole-v1")
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                _, value = self.model(torch.FloatTensor(state).unsqueeze(0))
                self.store_transition((state, action, log_prob, reward, next_state, done, value.item()))
                state = next_state
                total_reward += reward

                if len(self.memory) >= 20:
                    self.train()

            if episode % print_every == print_every - 1:
                print(f'Episode: {episode+1}, Total Reward: {total_reward}')
        env.close()

if __name__ == '__main__':
    agent = PPOAgent()
    agent.run(1000)
