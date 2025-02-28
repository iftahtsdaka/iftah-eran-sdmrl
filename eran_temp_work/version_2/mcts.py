import gymnasium as gym
import numpy as np
import math
import random
import itertools
from env_nosell import SimplifiedWaterSupplyEnv, create_env, DiscreteObservation
import itertools
from qlearning import TabularQ
from gymnasium.spaces import MultiDiscrete



def copy_water_env(env: SimplifiedWaterSupplyEnv):
    water_env = env.unwrapped
    new_env = create_env(water_env.max_cycles,
                         water_env.time_bucket_count,
                         water_env.water_bucket_count,
                         water_env.discrete_observations,
                         water_env.discrete_actions) # TODO: add Normalize Observations + Actions
    new_env.unwrapped.set_state(water_env.get_state())
    return DiscretizedActionWrapper(new_env, granularity=env.granularity)

# ------------------------------
# Environment Simulator Using the Real Environment
# ------------------------------
def env_simulator(env:SimplifiedWaterSupplyEnv, state, action, high_level_model):
    """
    Uses the actual environment to simulate one low-level step.    
    Args:
        env: Gym environment instance.
        state: The saved state
        action: The discrete low-level action.
        high_level_model: high level model to provide simulated high level decisions
    
    Returns:
        next_state: The new state after the action.
        reward: The reward obtained.
        done: Whether the episode terminated.
    """
    # Restore environment to the given state
    env.unwrapped.set_state(state)
    high_level_action = high_level_model.predict(env.observation_translator.observation(state))
    high_level_action = env.action(high_level_action)
    new_action = np.array(action) + np.array(high_level_action)
    # Execute the high-level action (for example, using the bin index directly)
    observation, reward, done, truncated, info = env.unwrapped.step(new_action)
    next_state = env.unwrapped.get_state()

    return next_state, reward, done

# ------------------------------
# MCTS Implementation (as before)
# ------------------------------
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # The discrete action that led to this node
        self.children = {}    # Mapping: action -> child node
        self.visits = 0
        self.total_reward = 0.0

    def uct_score(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        q_value = self.total_reward / self.visits
        bonus = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return q_value + bonus

    def expand(self, available_actions, env, high_level_model, horizon, c_param=1.41):
        for action in available_actions:
            if action not in self.children:
                next_state, reward, done = env_simulator(env, self.state, action, high_level_model)
                child_node = MCTSNode(next_state, parent=self, action=action)
                child_node.total_reward = reward
                child_node.visits = 1
                self.children[action] = child_node
                return child_node
        return None

def mcts_search(root_state, #TODO: add first high level model action
                env:SimplifiedWaterSupplyEnv,
                initial_high_level_action,
                high_level_model,
                available_actions,
                n_iters=100,
                horizon=5,
                c_param=1.41):
    root = MCTSNode(root_state)
    simulation_env = copy_water_env(env)
    root.visits = 1
        # Apply the initial high-level action first
    for low_level_action in available_actions:
            # Ensure the simulation starts from the same root state.
            simulation_env.unwrapped.set_state(root_state)
            # Convert the provided high-level action into the environment's low-level representation.
            high_level_component = simulation_env.action(initial_high_level_action)
            # Compute the full action as the sum of the high-level component and the candidate low-level action.
            full_action = np.array(high_level_component) + np.array(low_level_action)
            # Take a step in the environment.
            next_state, reward, done, truncated, info = simulation_env.unwrapped.step(full_action)
            # Create a child node from the result of this combined action.
            child_node = MCTSNode(next_state, parent=root, action=low_level_action)
            child_node.total_reward = reward
            child_node.visits = 1
            root.children[low_level_action] = child_node

    for i in range(n_iters):
        node = root
        depth = 0

        # Selection
        while node.children and depth < horizon:
            best_action = max(node.children, key=lambda a: node.children[a].uct_score(c_param))
            node = node.children[best_action]
            depth += 1
            if node.parent is None:
                break

        # Expansion
        if depth < horizon:
            child = node.expand(available_actions,
                                simulation_env,
                                high_level_model,
                                horizon,
                                c_param)
            if child is not None:
                node = child
                depth += 1

        # Simulation (Rollout)
        sim_reward = 0.0
        sim_state = node.state
        discount = 1.0
        for d in range(depth, horizon):
            action = random.choice(available_actions)
            sim_state, reward, done = env_simulator(simulation_env, sim_state, action, high_level_model)
            sim_reward += discount * reward
            discount *= 0.95  # high-level discount factor
            if done:
                break

        # Backpropagation
        total_reward = sim_reward
        while node is not None:
            node.visits += 1
            node.total_reward += total_reward
            node = node.parent

    # Choose best low-level action from root
    best_action = None
    best_avg = -float('inf')
    for action, child in root.children.items():
        avg_reward = child.total_reward / child.visits
        if avg_reward > best_avg:
            best_avg = avg_reward
            best_action = action
    return best_action


class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, granularity=5):
        super().__init__(env)
        self.granularity = granularity
        buckets = self.unwrapped.water_bucket_count
        self.available_actions = [item for item in itertools.product(np.arange(0, buckets, self.granularity),
                                                                     np.arange(0, buckets, self.granularity))]
        
        self.action_space = gym.spaces.Discrete((buckets // granularity)**2)
        self.observation_translator = DiscreteObservation(env,
                                                env.unwrapped.time_bucket_count,
                                                env.unwrapped.water_bucket_count)
    
    def action(self, action):
        new_action = self.available_actions[action]
        return new_action

class HighLevelTabularQ():
    def predict(self, state):
        if not hasattr(self, "Q"):
            raise(RuntimeError("Please train the model before predicting"))
        return np.argmax(self.Q[state, :])
    
    def tabular_q_learning(self, env, mcts_available_actions,
                           total_episodes=50000, learning_rate=0.1, gamma=0.99,
                           epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):

        # Retrieve number of states and actions from the wrapped environment.
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))
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
                    action = np.argmax(self.Q[state, :])
                
                mcts_action = mcts_search(
                    env.unwrapped.get_state(),
                    env,
                    initial_high_level_action=action,
                    high_level_model=self,     
                    available_actions=mcts_available_actions,
                    n_iters=5,
                    horizon=3)
                
                new_state, reward, done, truncated, info = env.unwrapped.step(mcts_action)
                current_timestep += 1
                total_reward += reward
                observation = env.observation_translator.observation(new_state)
                # Q-learning update rule
                self.Q[state, action] += learning_rate * (reward + gamma * np.max(self.Q[observation, :]) - self.Q[state, action])
                state = observation

            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            rewards_all_episodes.append(total_reward)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}: Total Reward = {total_reward}")


    def train(self, env, total_episodes=50000, learning_rate=0.1, gamma=0.99,
                       epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, granularity=10):
        wrapped_env = DiscretizedActionWrapper(env, granularity=granularity)
        mcts_available_actions = [item for item in itertools.product(range(granularity),
                                                                           range(granularity))]
        self.Q, rewards_all_episodes = self.tabular_q_learning(wrapped_env,
                                                               mcts_available_actions,
                                                                total_episodes,
                                                                learning_rate,
                                                                gamma,
                                                                epsilon,
                                                                epsilon_decay,
                                                                min_epsilon)
        return rewards_all_episodes
    
    
     


        
        