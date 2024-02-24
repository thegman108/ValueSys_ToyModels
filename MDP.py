### Defining a markov decision process
### Thanks ChatGPT (again)

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class MDP:
    def __init__(self, states, actions, transition_function, reward_function, gamma):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function  # Function: (s, a, s') -> Probability
        self.reward_function = reward_function  # Function: (s, a, s') -> Reward, now passed during initialization
        self.gamma = gamma

    def get_possible_actions(self, state):
        return self.actions

    def transition(self, state, action):
        """
        Given a state and action, return a list of (probability, next_state, reward) triples.
        """
        return [(self.transition_function(state, action, next_state),
                 next_state,
                 self.reward_function(state, action, next_state))
                for next_state in self.states]

    def compute_optimal_policy_value(self, values):
        # Initialize the policy for each state to None
        policy = {state: None for state in self.states}

        # After the value function has converged, determine the optimal policy
        for state in self.states:
            # Calculate the expected value of each action using the converged value function
            action_values = {
                action: sum(p * (reward + self.gamma * values[next_state])
                            for p, next_state, reward in self.transition(state, action))
                for action in self.get_possible_actions(state)
            }

            # Select the action that has the highest expected value
            # print(action_values)
            max_value = float("-inf")
            max_action = None
            for action, value in action_values.items():
                if value > max_value or abs(value - max_value) < 1e-9 and random.choice([True, False]):
                    max_value = value
                    max_action = action
            policy[state] = max_action

        # Return the optimal policy and the converged value function
        return policy
    
    def compute_optimal_policy_q_value(self, q_values):
        # Initialize the policy for each state to None
        policy = {state: None for state in self.states}

        # After the value function has converged, determine the optimal policy
        for state in self.states:
            # Calculate the expected value of each action using the converged value function
            action_values = {
                action: q_values[state, action]
                for action in self.get_possible_actions(state)
            }

            # Select the action that has the highest expected value
            max_value = float("-inf")
            max_action = None
            for action, value in action_values.items():
                if value > max_value or abs(value - max_value) < 1e-9 and random.choice([True, False]):
                    max_value = value
                    max_action = action
            policy[state] = max_action

        # Return the optimal policy and the converged value function
        return policy
    
    def compute_optimal_policy(self, epsilon=0.01, max_iterations=1000):
        """
        Compute the optimal policy using value iteration.
        
        Args:
            epsilon (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
        
        Returns:
            dict: The optimal policy
        """
        # Initialize the value function for each state to zero
        values = {state: 0 for state in self.states}

        # Repeat until convergence or maximum iterations reached
        for _ in range(max_iterations):
            # Keep track of the maximum change in the value function across all states
            delta = 0

            # Update the value function for each state
            for state in self.states:
                # Store the current value function for the state
                v = values[state]

                # Calculate the value for all possible actions from the current state
                action_values = [
                    sum(p * (reward + self.gamma * values[next_state])
                        for p, next_state, reward in self.transition(state, action))
                    for action in self.get_possible_actions(state)
                ]

                # Update the value function to the maximum value across all actions
                values[state] = max(action_values)

                # Update delta to the maximum change in the value function
                delta = max(delta, abs(v - values[state]))

            # If the maximum change in the value function is less than epsilon, we've converged
            if delta < epsilon:
                # print(f"Converged after {_} iterations")
                break
        return self.compute_optimal_policy_value(values)

    def plot_graph(self):
        G = nx.MultiDiGraph()

        # Add nodes for each state
        for state in self.states:
            G.add_node(state)

        # Add edges for each action and transition
        for state in self.states:
            for action in self.actions:
                for next_state in self.states:
                    if self.transition_function(state, action, next_state) > 0:
                        reward = self.reward_function(state, action, next_state)
                        G.add_edge(state, next_state, action=action, weight=reward)

        # plt.figure(figsize=(12, 8))  # Increase the plot size
        pos = nx.spring_layout(G, k=0.5, iterations=20)  # k: Optimal distance between nodes. Increase/decrease to spread nodes out

        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_labels(G, pos)

        # Draw edges and add labels
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.5)

        edge_labels = {(u, v): f"{d['action']}:{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        
        # Calculate edge label positions at the midpoint of the edges
        edge_label_pos = {edge: ((pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2) 
                        for edge in G.edges()}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        plt.axis('off')
        plt.tight_layout()  # Adjust layout to prevent cutting off edge labels
        plt.show()

    def __str__(self):
        states_str = ', '.join(self.states)
        actions_str = ', '.join(self.actions)
        return (f"MDP Summary:\n"
                f"States: {states_str}\n"
                f"Actions: {actions_str}\n"
                f"Discount Factor: {self.gamma}")


if __name__ == "__main__":
    # Example MDP setup
    states = ['s1', 's2', 's3']
    actions = ['a1', 'a2']
    def transition_function(s, a, s_prime): #simple function for demonstration
        if a == actions[0]:
            return 1 if s_prime == s else 0
        else: # s1 to s2, s2 to s3
            return 1 if (s == 's1' and s_prime == 's2') or (s == 's2' and s_prime == 's3') or (s == 's3' and s_prime == 's3') else 0

    # Define a reward function that assigns random rewards for each state-action pair
    def random_reward_function(states, actions, transition_function = transition_function):
        """Initializes a reward function that assigns random rewards to state-action pairs."""
        rewards = {}
        for state in states:
            for action in actions:
                for next_state in states:
                    rewards[(state, action, next_state)] = np.random.normal(0, 1) if transition_function(state, action, next_state) > 0 else 0
                    # Gaussian distribution, mean=0, variance=1
        # print(rewards)
        return lambda s, a, s_prime: rewards[(s, a, s_prime)]

    gamma = 0.9 # Discount factor
    # Create an MDP object with the randomized reward function

    mdp = MDP(states, actions, transition_function, random_reward_function(states, actions), gamma)
    optimal_policy = mdp.compute_optimal_policy()
    mdp.plot_graph()
    print(mdp)
    print("Optimal Policy:", optimal_policy)