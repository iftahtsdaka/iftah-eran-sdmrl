import numpy as np
import gym
from stable_baselines3.common.env_checker import check_env

# Define a set of expected keys from env.info()
EXPECTED_INFO_KEYS = {
    "water_level",
    "price_A",
    "price_B",
    "demand",
    "current_time_bucket",
    "current_cycle",
    "total_reward",
}

def test_reset_environment():
    """Test that the environment resets properly."""
    # Create environment with a couple of cycles for a quick test
    env = SimplifiedWaterSupplyEnv(max_cycles=2, hours_per_cycle=168,
                                   time_bucket_count=168, water_bucket_count=300)
    obs, info = env.reset()
    
    # Observation should be a numpy array with shape (5,)
    assert isinstance(obs, np.ndarray), "Observation is not a numpy array."
    assert obs.shape == (5,), f"Expected observation shape (5,), got {obs.shape}"
    
    # Info should include all expected keys
    for key in EXPECTED_INFO_KEYS:
        assert key in info, f"Missing key '{key}' in info dictionary."

def test_step_environment():
    """Test a single step of the environment."""
    env = SimplifiedWaterSupplyEnv(max_cycles=2, hours_per_cycle=168,
                                   time_bucket_count=168, water_bucket_count=300)
    obs, info = env.reset()
    
    # Use a sample continuous action (example: buy 10 units from each source)
    action = np.array([10.0, 10.0])
    new_obs, reward, done, truncated, info = env.step(action)
    
    # Check observation shape and type
    assert isinstance(new_obs, np.ndarray), "New observation is not a numpy array."
    assert new_obs.shape == (5,), f"Expected new_obs shape (5,), got {new_obs.shape}"
    
    # Check reward and termination flags
    assert isinstance(reward, float), "Reward is not a float."
    assert isinstance(done, bool), "Done flag is not boolean."
    assert isinstance(truncated, bool), "Truncated flag is not boolean."
    
    # Check info dictionary again
    for key in EXPECTED_INFO_KEYS:
        assert key in info, f"Missing key '{key}' in info dictionary after step."

def test_env_with_check_env():
    """Run gym's check_env to ensure the environment conforms to the Gym API."""
    env = SimplifiedWaterSupplyEnv()
    # This will raise an error if the environment does not follow the Gym API.
    check_env(env)

def test_discrete_actions_wrapper():
    """Test the DiscreteActions wrapper functionality."""
    env = SimplifiedWaterSupplyEnv()
    # Wrap the env to get discrete actions (e.g., actions in increments of 10)
    wrapped_env = DiscreteActions(env, max_water_volume=300, size_of_purchase=10)
    
    # Check that the action_space is now Discrete
    assert isinstance(wrapped_env.action_space, gym.spaces.Discrete), \
        "Action space is not Discrete after wrapping."
    
    # Test conversion: for a given discrete action index, the wrapper should return a list with two values
    discrete_index = 5  # example discrete action index
    continuous_action = wrapped_env.action(discrete_index)
    assert isinstance(continuous_action, list), "Returned action is not a list."
    assert len(continuous_action) == 2, "Expected two actions (one per source)."
    
    # Check that each action quantity is a multiple of the size_of_purchase (10)
    for q in continuous_action:
        assert q % 10 == 0, f"Action {q} is not a multiple of 10."

def test_discrete_observation_wrapper():
    """Test the DiscreteObservation wrapper functionality."""
    env = SimplifiedWaterSupplyEnv()
    # Wrap the env so that observations are converted to a single discrete index.
    wrapped_env = DiscreteObservation(env, water_level_resolution=10,
                                      price_resolution=1,
                                      demand_resolution=10,
                                      time_resolution=1)
    
    # Check that the observation_space is now Discrete
    assert isinstance(wrapped_env.observation_space, gym.spaces.Discrete), \
        "Observation space is not Discrete after wrapping."
    
    # Reset the base env and then transform the observation
    obs, _ = env.reset()
    discrete_obs = wrapped_env.observation(obs)
    # For discrete wrappers, the observation is typically an integer index.
    assert isinstance(discrete_obs, int), "Discrete observation is not an integer."


test_reset_environment()
test_step_environment()
test_env_with_check_env()
test_discrete_actions_wrapper()
test_discrete_observation_wrapper()
print("All tests passed successfully.")
