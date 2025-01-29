from gymnasium.envs.registration import register

register(
    id="WaterSupplyNoSellEnv/WaterSupplyEnv-v0",
    entry_point="WaterSupplyNoSellEnv.envs:WaterSupplyEnv",
)
