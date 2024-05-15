import numpy as np
import random

class QTableAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim
    
    def update(self, state, action, reward, next_state, done):
        q_value = self.q_table[state, action]
        next_q_value = np.max(self.q_table[next_state])
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        self.q_table[state, action] += self.lr * (expected_q_value - q_value)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            # action = np.argmax(self.q_table[state])
            # return action randomly in case of ties
            action = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))
        else:
            action = random.randrange(self.action_dim)
        return action