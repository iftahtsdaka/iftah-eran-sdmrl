import numpy as np
import gymnasium as gym


# Import your environment and wrappers

def tabular_q_learning(env, total_episodes=50000, learning_rate=0.1, gamma=0.99,
                       epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """
    A simple tabular Q-learning implementation.

    Args:
        env: A Gym environment with discrete observation and action spaces.
        num_episodes: Number of episodes for training.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration probability.
        epsilon_decay: Multiplicative decay for epsilon per episode.
        min_epsilon: Minimum exploration probability.

    Returns:
        Q: The learned Q-table (2D NumPy array).
        rewards_all_episodes: List of total rewards per episode.
    """
    # Retrieve number of states and actions from the wrapped environment.
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))
    rewards_all_episodes = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        current_timestep = 0

        while not done and not truncated:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, done, truncated, info = env.step(action)
            current_timestep += 1
            total_reward += reward

            # Q-learning update rule
            Q[state, action] += learning_rate * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_all_episodes.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

    return Q, rewards_all_episodes



class TabularQ():
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.Q = np.zeros((n_states, n_actions))
    
    def train(self, env, total_episodes=50000, learning_rate=0.1, gamma=0.99,
                       epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.Q, rewards_all_episodes = tabular_q_learning(env,
                                                        total_episodes,
                                                        learning_rate,
                                                        gamma,
                                                        epsilon,
                                                        epsilon_decay,
                                                        min_epsilon)
        return rewards_all_episodes
    
    def predict(self, state):
        return np.argmax(self.Q[state, :])
