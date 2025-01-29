import gymnasium as gym
from gymnasium import spaces
import numpy as np


########################
### Global variables ###
########################

# Size of the agent's pool
AGENT_WATER_VOLUME_MAX = 300
# Price of missing a 1 water unit of consumer: 
# e.g agent's water volume = 10, consumer demand = 20, price = (20-10) * PENALTY_PER_WATER_UNIT
PENALTY_PER_WATER_UNIT = 1000

TOTAL_DAILY_DEMAND = 1000



#########################
### Price calculation ###
#########################

def hourly_demand_means():
    # Hourly water demand according to: 
    hourly_demand = np.array([
        2, 2, 2, 2, 3, 5, 10, 12, 10, 8, 6, 5, 5, 5, 5, 6, 7, 9, 10, 9, 6, 4, 3, 2
    ])
    # Normalize: Daily demand is 1000
    hourly_demand = (hourly_demand / hourly_demand.sum()) * TOTAL_DAILY_DEMAND

    # on day 7: less demand
    hourly_demand = np.append(np.tile(hourly_demand,6), hourly_demand / 2)
    return hourly_demand



def sample_demand(hour, std=10):
    # Sample from a normal distribution, ensure demand is non-negative
    hourly_demand = hourly_demand_means()
    mean_demand = hourly_demand[hour] # Mean demand follows a daily pattern, higher during the day (6 AM to 6 PM)
    return max(0, np.random.normal(mean_demand, std)) # When std_dev is relatively low we will get lines that are very very close to the original function.


def calculate_price(amount_to_buy: float, base_price: float) -> float:
    return amount_to_buy * base_price + 0.05 * base_price * (amount_to_buy -1) ** 2


def get_water_prices():
    # expensive hours: 8:00-16:00
    source_1_price = np.ones(24)
    source_1_price[8:16] = 2
    source_1_price = np.tile(source_1_price,7)
    return source_1_price


##############################################
### Environment Definition : WaterSupplyEnv ###
##############################################

class WaterSupplyEnv(gym.Env):

    def __init__(self, max_weeks=3):
        super().__init__()
        # State: [water_level, price_A, price_B, demand, current_hour]
        self.max_weeks = max_weeks
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Actions: [buy_from_A, buy_from_B]
        self.action_space = spaces.Box(low=0, high=AGENT_WATER_VOLUME_MAX, shape=(2,), dtype=np.float32)

        # Environment parameters
        self.max_water_level = AGENT_WATER_VOLUME_MAX
        self.source_A_base_prices = get_water_prices()
        
        # Source B is more expensive
        self.source_B_base_prices = 1.5 * self.source_A_base_prices

    def _get_obs(self):
        return np.array([self.water_level,
                         self.price_A,
                         self.price_B,
                         self.demand,
                         self.current_hour])
    def _get_info(self):
        # return more expicit info, should be used only for debugging
        return {"water_level": self.water_level,
                "price_A": self.price_A,
                "price_B": self.price_B,
                "demand": self.demand,
                "current_hour": self.current_hour}

    def reset(self, seed = None, options = None):
        # seed self.np_random
        super().reset(seed=seed)

        # initialize env state
        self.total_hours = 0
        self.current_hour = 0
        self.water_level = self.max_water_level # Initial water level
        self.demand = sample_demand(self.current_hour) # Initial demand
        self.price_A = self.source_A_base_prices[self.current_hour]
        self.price_B = self.source_B_base_prices[self.current_hour]
        self.total_reward = 0
        return self._get_obs(), self._get_info() # obs, info

    def step(self, action):
        buy_from_A, buy_from_B = action

        # Calculate penalty for unmet demand
        unmet_demand_penalty = max(0, (self.demand - self.water_level) * PENALTY_PER_WATER_UNIT)
        self.water_level = max(0, self.water_level - self.demand)

        
        # Costs and revenues
        cost_A = calculate_price(buy_from_A, self.source_A_base_prices[self.current_hour])
        cost_B = calculate_price(buy_from_B, self.source_B_base_prices[self.current_hour])
        # Update water stock
        self.water_level += buy_from_A + buy_from_B

        # Calculate reward
        reward = (- cost_A - cost_B) - unmet_demand_penalty # reward function
        self.total_reward += reward

        # Ensure constraints
        self.water_level = min(self.max_water_level, self.water_level)  # Ensure water level does not surpass maximum

        # Update state
        self.total_hours += 1
        self.current_hour = (self.current_hour + 1) % 168
        self.demand = sample_demand(self.current_hour)
        self.price_A = self.source_A_base_prices[self.current_hour]
        self.price_B = self.source_B_base_prices[self.current_hour]

        observation = self._get_obs()

        # terminated = False  # TODO: Define terminal conditions if necessary
        terminated = True if (self.max_weeks and self.total_hours >= 168 * self.max_weeks) else False
        truncated = False # TODO: Define truncated conditions if necessary
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode="human", **kwargs):
        if kwargs.get("close", False):
            # If 'close' is passed, do nothing
            return
        print(f"Water Stock: {self.water_level}, Price A: {self.price_A}, Price B: {self.price_B}, Demand: {self.demand}, MONEY: {self.total_reward}")

