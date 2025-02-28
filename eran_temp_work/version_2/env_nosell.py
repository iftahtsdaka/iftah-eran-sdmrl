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
from gymnasium.utils import seeding

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
PENALTY_PER_WATER_UNIT = 10000000
AGENT_WATER_VOLUME_MAX = 300
HOURS_IN_A_WEEK = 168
PRICE_A = 1 # base price A
PREMIUM_FACTOR = 2 # how much a price get pricer on expensive hours
PRICE_A_PREMIUM = PREMIUM_FACTOR * PRICE_A # price A on expensive hours
PRICE_B_FACTOR = 1.5 # how much base price B is pricier than price A
PRICE_B = PRICE_B_FACTOR * PRICE_A # base price B
PRICE_B_PREMIUM = PREMIUM_FACTOR * PRICE_B # price B on expensive hours
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
    """
    Returns 168 array of demands first 6 days sums to TOTAL_DAILY_DEMAND demand. 
    The last day sums to TOTAL_DAILY_DEMAND / 2 
    """
    hourly_demand = np.array([2, 2, 2, 2, 3, 5, 10, 12, 10, 8, 6, 5, 5, 5, 5, 6, 7, 9, 10, 9, 6, 4, 3, 2])
    hourly_demand = (hourly_demand / hourly_demand.sum()) * TOTAL_DAILY_DEMAND
    # For a week: 6 full days and 1 day at half demand
    hourly_demand = np.append(np.tile(hourly_demand, 6), hourly_demand / 2)
    return hourly_demand

def sample_demand(hour, rng=None, std=10):
    """ Sample[hour]: base hourly demand + normal noise """
    pattern = get_hourly_demand_pattern()
    mean_demand = pattern[hour]
    return max(0, rng.normal(mean_demand, std))

def get_water_prices(hours):
    """
    Hourly prices over a day, 8am-4pm more expensive
    """
    base_prices = np.ones(24) * PRICE_A
    base_prices[8:16] = PRICE_A_PREMIUM  # More expensive between 8:00 and 16:00
    base_prices = np.tile(base_prices, 7)[:hours]
    return base_prices

# -----------------------
# Simplified Environment
# -----------------------
class SimplifiedWaterSupplyEnv(gym.Env):
    def __init__(self,
                 max_cycles=10,
                 hours_per_cycle=HOURS_IN_A_WEEK,
                 time_bucket_count=HOURS_IN_A_WEEK,
                 water_bucket_count=AGENT_WATER_VOLUME_MAX,
                 discrete_observations = False,
                 discrete_actions = False,
                 normalize_observations=False,
                 normalize_actions=False,
                 price_function=default_price_function):
        """
        max_cycles: number of cycles (e.g. weeks)
        hours_per_cycle: number of hours in one cycle (e.g. 168)
        time_bucket_count: discrete time steps per cycle (aggregated from hours)
        water_bucket_count: number of discrete water/demand levels
        price_function: function for computing water cost

        Note: Simplifying by using buckets is only relevant for discrete,
        as continus has infinite number of options in any case!
        """
        super(SimplifiedWaterSupplyEnv, self).__init__()
        self.seed = None
        self.max_cycles = max_cycles # Finite horizon
        self.hours_per_cycle = hours_per_cycle
        self.time_bucket_count = time_bucket_count
        self.discrete_observations = discrete_observations
        self.discrete_actions = discrete_actions
        self.price_function = price_function

        # Time Interval size: How many original hours per bucket? 
        self.aggregation_interval = hours_per_cycle / time_bucket_count

        # Determine the bucket size for water/demand.
        self.water_bucket_count = water_bucket_count
        self.water_bucket_size = AGENT_WATER_VOLUME_MAX / water_bucket_count

    
        # Observation: [water_level, price_A, price_B, demand, current_time_bucket]

        # - water_level: [0,300] float
        # - price_A: [1,2] float
        # - price_B: [1.5,3] float
        # - demand: [0,inf) float
        # - current_time_bucket: [0, time_bucket_count]

        low_bounds = np.array([
            0,    # water_level min
            PRICE_A,    # price_A min
            PRICE_B,  # price_B min
            0,    # demand min
            0     # current_time_bucket min
        ], dtype=np.float32)

        high_bounds = np.array([
            AGENT_WATER_VOLUME_MAX,                # water_level max
            PRICE_A_PREMIUM,                  # price_A max
            PRICE_B_PREMIUM,                  # price_B max
            AGENT_WATER_VOLUME_MAX,             # demand max (assumed)
            self.time_bucket_count  # current_time_bucket max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # Action: how much water to buy from each source (continuous; later wrapped to discrete)
        self.action_space = spaces.Box(low=0, high=AGENT_WATER_VOLUME_MAX, shape=(2,), dtype=np.float32)

        # Pre-compute hourly prices and then aggregate them into time buckets.
        self.base_hourly_prices = get_water_prices(hours_per_cycle)
        # Price B is more expensive than Price A
        # self.source_A_base_prices = self._aggregate_prices(self.base_hourly_prices) # Simplify the Env
        self.source_A_base_prices =self.base_hourly_prices[::int(self.aggregation_interval)]
        self.source_B_base_prices = 1.5 * self.source_A_base_prices

        self.reset()

    def _aggregate_prices(self, hourly_prices):
        """Aggregate (by Averaging) hourly prices into time buckets."""
        aggregated = []
        for i in range(self.time_bucket_count):
            start = int(i * self.aggregation_interval)
            end = int((i + 1) * self.aggregation_interval)
            agg_price = np.mean(hourly_prices[start:end])
            aggregated.append(agg_price)
        return np.array(aggregated)

    def _aggregate_demand(self, start_hour):
        """Aggregate (by Averaging) demand over the time bucket and discretize it."""
        demands = [sample_demand(
            hour=(h % self.hours_per_cycle),
            rng=self.np_random,
            ) 
            for h in range(
                int(start_hour), 
                int(start_hour) + 
                int(self.aggregation_interval)
                )]
        avg_demand = np.mean(demands)
        return avg_demand

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_cycle = 0
        self.current_time_bucket = 0
        self.np_random, self.seed = seeding.np_random(seed)


        # Start with full water (discretized)
        self.water_level = AGENT_WATER_VOLUME_MAX
        if self.discrete_observations:
            self.water_level = discretize(self.water_level, self.water_bucket_size)
        self.total_reward = 0.0

        # Initialize demand and prices for the first time bucket
        self.demand = self._aggregate_demand(0)
        self.price_A = self.source_A_base_prices[self.current_time_bucket]
        self.price_B = self.source_B_base_prices[self.current_time_bucket]
        return self._get_obs(), self._get_info()
    
    def get_raw_demands(self):
       return [self._aggregate_demand(i) for i in range(self.time_bucket_count)]

    def step(self, action):
        buy_from_A, buy_from_B = action
        self.current_time_bucket = int(self.current_time_bucket)

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
        self.water_level = min(self.water_level, AGENT_WATER_VOLUME_MAX)
        if self.discrete_observations:
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
            start_hour = self.current_time_bucket * int(self.aggregation_interval)
            self.demand = self._aggregate_demand(start_hour)
            if self.discrete_observations:
                self.demand = discretize(self.demand, self.water_bucket_size)
            self.price_A = self.source_A_base_prices[self.current_time_bucket]
            self.price_B = self.source_B_base_prices[self.current_time_bucket]

        return self._get_obs(), reward, done, False, self._get_info()
    

    def _get_obs(self):
        return np.array([self.water_level, self.price_A, self.price_B, self.demand, self.current_time_bucket], dtype=np.float32)
    
    def get_obs(self):
        return self._get_obs()

    def get_state(self, stripped=True):
        ret ={
            "water_level": self.water_level,
            "price_A": self.price_A,
            "price_B": self.price_B,
            "demand": self.demand,
            "current_time_bucket": self.current_time_bucket,
        }

        if stripped:
            return list(ret.values())
        else:
            return ret
            

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
    
    # def extract_features_from_discrete_state(self,discrete_obs):
    #     # discrete_obs is an Integer
    #     amount_of_water = self.water_bucket_count
    #     amount_of_price = 2
    #     print(discrete_obs, amount_of_water)
    #     water_idx = discrete_obs % amount_of_water
    #     remainder = discrete_obs // amount_of_water

    #     # Extract price_A_idx
    #     price_A_idx = remainder % amount_of_price
    #     remainder //= amount_of_price

    #     # Extract price_B_idx
    #     price_B_idx = remainder % amount_of_price
    #     remainder //= amount_of_price

    #     # Extract demand_idx and time_idx
    #     amount_of_demand= amount_of_water
    #     demand_idx = remainder % amount_of_demand
    #     time_idx = remainder // amount_of_demand

    #     return water_idx, price_A_idx, price_B_idx, demand_idx, time_idx


    def set_state(self, state):
            self.water_level, self.price_A, self.price_B, self.demand, self.current_time_bucket = state


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
        self.action_amount = max_water_volume // size_of_purchase
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
    def __init__(self, env, time_buckets, water_buckets):
        super().__init__(env)
        self.water_buckets = water_buckets
        # Assumer demand is always less than AGENT_WATER_VOLUME_MAX 
        self.water_resolution = (AGENT_WATER_VOLUME_MAX+1) / (water_buckets) # how much water each discrete "water volume interval" holds
        # self.time_resolution = (HOURS_IN_A_WEEK) / time_buckets # how much hour each dicrete interval represents

        # amount of intervals per feature
        self.amount_of_water = water_buckets
        self.amount_of_price = 2 # (price A: 1, 2.  Price B: 1.5, 3)
        
        self.amount_of_demand = water_buckets
        self.amount_of_time = time_buckets

        # The flattened observation index
        self.observation_space = spaces.Discrete(self.amount_of_water * 
                                                 self.amount_of_price * 
                                                 self.amount_of_price * 
                                                 self.amount_of_demand * 
                                                 self.amount_of_time)

    def get_discreteobservationwarpper_info(self):
        return {
            "amount_of_water": self.amount_of_water,
            "amount_of_price": self.amount_of_price,
            "amount_of_demand": self.amount_of_demand,
            "amount_of_time": self.amount_of_time,
        }
    
    def observation(self, observation):
        # observation: [water_level, price_A, price_B, demand, current_time_bucket]
        #               0-300        1,2      1.5 3    0-300   timebucket
        water_level, price_A, price_B, demand, current_time_bucket = observation
        
        # assert demand <= AGENT_WATER_VOLUME_MAX, f"Assertion failed: demand={demand} exceeds AGENT_WATER_VOLUME_MAX ({AGENT_WATER_VOLUME_MAX})"
        water_idx = int(water_level // self.water_resolution)
        price_A_idx = 0 if price_A == PRICE_A else 1
        price_B_idx = 0 if price_B == PRICE_B else 1
        demand_idx = int(demand // self.water_resolution)
        time_idx = int(current_time_bucket) # DELETE: // self.time_resolution)
        
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
    

class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes continuous observation features to [0,1]:
        - water_level: [0,300]  -> normalized value = water_level / 300
        - price_A: [1,2]       -> normalized value = (price_A - 1) / (2 - 1)
        - price_B: [1.5,3]     -> normalized value = (price_B - 1.5) / (3 - 1.5)
        - demand: [0, TOTAL_DAILY_DEMAND] -> normalized value = demand / TOTAL_DAILY_DEMAND
        - current_time_bucket: [0, hours_per_cycle] -> normalized value = current_time_bucket / hours_per_cycle
    """
    def __init__(self, env, hours_per_cycle, discrete_observations):
        super().__init__(env)
        self.hours_per_cycle = hours_per_cycle
        self.discrete_observations = discrete_observations

        if not discrete_observations:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=env.observation_space.shape,
                                            dtype=np.float32)

    def observation(self, obs):
        if self.discrete_observations:
            return obs
        
        water_level, price_A, price_B, demand, current_time = obs
        norm_water = water_level / AGENT_WATER_VOLUME_MAX
        norm_price_A = (price_A - PRICE_A) / (PREMIUM_FACTOR * PRICE_A - PRICE_A)
        norm_price_B = (price_B - PRICE_B) / (PREMIUM_FACTOR * PRICE_B - PRICE_B)
        norm_demand = demand / AGENT_WATER_VOLUME_MAX
        norm_time = current_time / self.hours_per_cycle
        return np.array([norm_water, norm_price_A, norm_price_B, norm_demand, norm_time],
                        dtype=np.float32)
    

class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, discrete_actions):
        super().__init__(env)
        self.discrete_actions = discrete_actions

        if not self.discrete_actions:
            self.action_space= spaces.Box(low=0, high=1,
                                            shape=env.action_space.shape,
                                            dtype=np.float32)
    def action(self, action):
        buy_from_A, buy_from_B = action # ranges [0,1], [0,1] 
        # convert to env value range [0,300]
        buy_from_A = AGENT_WATER_VOLUME_MAX * buy_from_A
        buy_from_B = AGENT_WATER_VOLUME_MAX * buy_from_B
        return [buy_from_A, buy_from_B]
        

# create_env(time_buckets=5, water_buckets=5, discrete_observations=True, discrete_actions=True, normalize=False):
# -----------------------
# Environment Creation Function
# -----------------------
def create_env(max_cycles=5,
               time_buckets=HOURS_IN_A_WEEK,
               water_buckets=AGENT_WATER_VOLUME_MAX,
               discrete_observations=True,
               discrete_actions=True,
               normalize_observations=False,
               normalize_actions=False):
    """
    time_buckets: number of discrete time steps per cycle (also hours_per_cycle).
    water_buckets: number of discrete water levels (e.g., 5 levels).
    """

    assert not (normalize_actions and discrete_actions) # Normalize wrapper assumes non-discrete action space
    env = SimplifiedWaterSupplyEnv(
        max_cycles=max_cycles,
        hours_per_cycle=HOURS_IN_A_WEEK,       # How long is the original raw cycle (should remain 168)       
        time_bucket_count=time_buckets,        # number of time steps equals time_buckets
        water_bucket_count=water_buckets,   # water_bucket_count divisions yield water_buckets levels
        discrete_observations=discrete_observations,
        discrete_actions=discrete_actions,
        normalize_observations=normalize_observations,
        normalize_actions=normalize_actions,
    )
    # Determine the purchase increment unit
    purchase_increment = AGENT_WATER_VOLUME_MAX // (water_buckets)
    if discrete_actions:    
        env = DiscreteActions(
            env, 
        max_water_volume=AGENT_WATER_VOLUME_MAX, 
        size_of_purchase=purchase_increment
        )
    
    if discrete_observations:
        env = DiscreteObservation(
        env,
        time_buckets=time_buckets,
        water_buckets=water_buckets,
    )
    
    if normalize_observations:
        env = NormalizeObservationWrapper(
            env,
            hours_per_cycle=time_buckets,
            discrete_observations=discrete_observations,
        )
    if normalize_actions:
        env = NormalizeActionWrapper(
            env,
            discrete_actions=discrete_actions,
        )
        
    return env


#DEMO
# tb = 8
# wb = 5

# env = make_vec_env(lambda: create_env(
#     time_buckets=tb,
#     water_buckets=wb, 
#     discrete_actions=True, 
#     discrete_observations=True,
#     ), n_envs=1)

