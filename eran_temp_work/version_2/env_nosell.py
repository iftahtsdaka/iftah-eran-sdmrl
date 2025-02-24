# Imports
# -----------------------
# general
# -----------------------
import os

# -----------------------
# Gymnasium
# -----------------------
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# -----------------------
# Stable Baselines 3
# -----------------------

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


# -----------------------
# Global Constants
# -----------------------
TOTAL_DAILY_DEMAND = 1000
PENALTY_PER_WATER_UNIT = 1000
AGENT_WATER_VOLUME_MAX = 300
HOURS_IN_A_WEEK = 168

# -----------------------
# Helper Functions
# -----------------------
def discretize(value, bucket_size):
    """Rounds the value to the nearest multiple of bucket_size."""
    return int(round(value / bucket_size)) * bucket_size

def default_price_function(amount_to_buy, base_price):
    return min(amount_to_buy * base_price + 0.005 * base_price * (amount_to_buy - 1) ** 2,
               PENALTY_PER_WATER_UNIT)

def get_hourly_demand_pattern():
    hourly_demand = np.array([2, 2, 2, 2, 3, 5, 10, 12, 10, 8, 6, 5, 5, 5, 5, 6, 7, 9, 10, 9, 6, 4, 3, 2])
    hourly_demand = (hourly_demand / hourly_demand.sum()) * TOTAL_DAILY_DEMAND
    # For a week: 6 full days and 1 day at half demand
    hourly_demand = np.append(np.tile(hourly_demand, 6), hourly_demand / 2)
    return hourly_demand

def sample_demand(hour, std=10):
    pattern = get_hourly_demand_pattern()
    mean_demand = pattern[hour]
    return max(0, np.random.normal(mean_demand, std))

def get_water_prices(hours):
    base_prices = np.ones(24)
    base_prices[8:16] = 2  # More expensive between 8:00 and 16:00
    base_prices = np.tile(base_prices, 7)[:hours]
    return base_prices

# -----------------------
# Simplified Environment
# -----------------------
class SimplifiedWaterSupplyEnv(gym.Env):
    def __init__(self,
                 max_cycles=10,
                 hours_per_cycle=HOURS_IN_A_WEEK,
                 time_bucket_count=168,
                 water_bucket_count=300,
                 price_function=default_price_function):
        """
        max_cycles: number of cycles (e.g., weeks)
        hours_per_cycle: number of hours in one cycle (e.g., 168)
        time_bucket_count: discrete time steps per cycle (aggregated from hours)
        water_bucket_count: number of discrete water/demand levels
        price_function: function for computing water cost
        """
        super(SimplifiedWaterSupplyEnv, self).__init__()
        self.max_cycles = max_cycles
        self.hours_per_cycle = hours_per_cycle
        self.time_bucket_count = time_bucket_count
        self.price_function = price_function

        # How many original hours per bucket?
        self.aggregation_interval = hours_per_cycle // time_bucket_count

        # Determine the bucket size for water/demand.
        self.water_bucket_count = water_bucket_count
        self.water_bucket_size = AGENT_WATER_VOLUME_MAX / water_bucket_count

        # Observation: [water_level, price_A, price_B, demand, current_time_bucket]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        # Action: how much water to buy from each source (continuous; later wrapped to discrete)
        self.action_space = spaces.Box(low=0, high=AGENT_WATER_VOLUME_MAX, shape=(2,), dtype=np.float32)

        # Pre-compute hourly prices and then aggregate them into time buckets.
        self.base_hourly_prices = get_water_prices(hours_per_cycle)
        self.source_A_base_prices = self._aggregate_prices(self.base_hourly_prices)
        self.source_B_base_prices = 1.5 * self.source_A_base_prices

        self.reset()

    def _aggregate_prices(self, hourly_prices):
        """Aggregate hourly prices into time buckets by averaging."""
        aggregated = []
        for i in range(self.time_bucket_count):
            start = i * self.aggregation_interval
            end = (i + 1) * self.aggregation_interval
            agg_price = np.mean(hourly_prices[start:end])
            aggregated.append(agg_price)
        return np.array(aggregated)

    def _aggregate_demand(self, start_hour):
        """Aggregate demand over the time bucket and discretize it."""
        demands = [sample_demand(h % self.hours_per_cycle) for h in range(start_hour, start_hour + self.aggregation_interval)]
        avg_demand = np.mean(demands)
        return discretize(avg_demand, self.water_bucket_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_cycle = 0
        self.current_time_bucket = 0

        # Start with full water (discretized)
        self.water_level = discretize(AGENT_WATER_VOLUME_MAX, self.water_bucket_size)
        self.total_reward = 0.0

        # Initialize demand and prices for the first time bucket
        self.demand = self._aggregate_demand(0)
        self.price_A = self.source_A_base_prices[self.current_time_bucket]
        self.price_B = self.source_B_base_prices[self.current_time_bucket]
        return self._get_obs(), self._get_info()

    def step(self, action):
        buy_from_A, buy_from_B = action

        # Calculate penalty for unmet demand
        unmet_demand = max(0, self.demand - self.water_level)
        unmet_demand_penalty = unmet_demand * PENALTY_PER_WATER_UNIT

        # Subtract demand (and discretize afterward)
        self.water_level = max(0, self.water_level - self.demand)

        # Compute cost using the pricing function
        cost_A = self.price_function(buy_from_A, self.price_A)
        cost_B = self.price_function(buy_from_B, self.price_B)

        # Add purchased water and re-discretize
        self.water_level += (buy_from_A + buy_from_B)
        self.water_level = discretize(self.water_level, self.water_bucket_size)

        reward = - cost_A - cost_B - unmet_demand_penalty
        self.total_reward += reward

        # Advance the time bucket
        self.current_time_bucket += 1
        if self.current_time_bucket >= self.time_bucket_count:
            self.current_cycle += 1
            self.current_time_bucket = 0

        done = self.current_cycle >= self.max_cycles

        if not done:
            start_hour = self.current_time_bucket * self.aggregation_interval
            self.demand = self._aggregate_demand(start_hour)
            self.price_A = self.source_A_base_prices[self.current_time_bucket]
            self.price_B = self.source_B_base_prices[self.current_time_bucket]

        return self._get_obs(), reward, done, False, self._get_info()

    def _get_obs(self):
        return np.array([self.water_level, self.price_A, self.price_B, self.demand, self.current_time_bucket], dtype=np.float32)

    def _get_info(self):
        return {
            "water_level": self.water_level,
            "price_A": self.price_A,
            "price_B": self.price_B,
            "demand": self.demand,
            "current_time_bucket": self.current_time_bucket,
            "current_cycle": self.current_cycle,
            "total_reward": self.total_reward
        }

    def render(self):
        info = self._get_info()
        print(info)


# -----------------------
# Discrete Wrappers
# -----------------------
class DiscreteActions(gym.ActionWrapper):
    """
    Wraps a continuous action space into a discrete one.
    The agent can buy water in increments of size_of_purchase.
    """
    def __init__(self, env, max_water_volume=AGENT_WATER_VOLUME_MAX, size_of_purchase=10):
        super().__init__(env)
        self.action_amount = max_water_volume // size_of_purchase + 1
        self.size_of_purchase = size_of_purchase
        # Flattened action space: two sources => action_amount^2 possible actions.
        self.action_space = spaces.Discrete(self.action_amount * self.action_amount)

    def action_to_quantity(self, action):
        from_source_1 = action // self.action_amount
        from_source_2 = action % self.action_amount
        return [from_source_1 * self.size_of_purchase, from_source_2 * self.size_of_purchase]

    def action(self, action):
        return self.action_to_quantity(action)

class DiscreteObservation(gym.ObservationWrapper):
    """
    Wraps the observation space into a single discrete index.
    Each component is discretized according to the provided resolutions.
    """
    def __init__(self, env, water_level_resolution=10, price_resolution=1, demand_resolution=10, time_resolution=1):
        super().__init__(env)
        self.water_level_resolution = water_level_resolution
        self.price_resolution = price_resolution
        self.demand_resolution = demand_resolution
        self.time_resolution = time_resolution

        self.amount_of_water = AGENT_WATER_VOLUME_MAX // water_level_resolution + 1
        self.amount_of_price = PENALTY_PER_WATER_UNIT // price_resolution + 1
        self.amount_of_demand = TOTAL_DAILY_DEMAND // demand_resolution + 1
        self.amount_of_time = self.env.time_bucket_count

        # The flattened observation index
        self.observation_space = spaces.Discrete(self.amount_of_water * 
                                                 self.amount_of_price * 
                                                 self.amount_of_price * 
                                                 self.amount_of_demand * 
                                                 self.amount_of_time)

    def observation(self, observation):
        # observation: [water_level, price_A, price_B, demand, current_time_bucket]
        water_level, price_A, price_B, demand, current_time_bucket = observation

        water_idx = int(water_level // self.water_level_resolution)
        price_A_idx = int(price_A // self.price_resolution)
        price_B_idx = int(price_B // self.price_resolution)
        demand_idx = int(demand // self.demand_resolution)
        time_idx = int(current_time_bucket // self.time_resolution)

        discrete_obs = (
            water_idx +
            self.amount_of_water * (
                price_A_idx +
                self.amount_of_price * (
                    price_B_idx +
                    self.amount_of_price * (
                        demand_idx +
                        self.amount_of_demand * time_idx
                    )
                )
            )
        )
        return discrete_obs
    


    
# -----------------------
# Environment Creation Function
# -----------------------
def create_env(time_buckets=5, water_buckets=5):
    """
    time_buckets: number of discrete time steps per cycle (also hours_per_cycle).
    water_buckets: number of discrete water levels (e.g., 5 levels).
    """
    env = SimplifiedWaterSupplyEnv(
        max_cycles=5,
        hours_per_cycle=time_buckets,          # each bucket represents one hour
        time_bucket_count=time_buckets,        # number of time steps equals time_buckets
        water_bucket_count=water_buckets - 1   # water_bucket_count divisions yield water_buckets levels
    )
    # Determine the purchase increment: AGENT_WATER_VOLUME_MAX divided by (water_buckets - 1)
    purchase_increment = AGENT_WATER_VOLUME_MAX // (water_buckets - 1)
    
    env = DiscreteActions(env, max_water_volume=AGENT_WATER_VOLUME_MAX, size_of_purchase=purchase_increment)
    env = DiscreteObservation(
        env,
        water_level_resolution=purchase_increment,  # This yields 5 water levels: 0,75,150,225,300
        price_resolution=250,                         # 1000//250 + 1 = 5 price buckets
        demand_resolution=250,                        # 1000//250 + 1 = 5 demand buckets
        time_resolution=1                             # each time bucket is one step
    )
    return env


