# Water Supply Simulation with Reinforcement Learning

This repository contains implementations for simulating a water supply environment and training reinforcement learning agents. The simulation is built using Gymnasium, and the project features both basic tabular Q-learning as well as an MCTS (Monte Carlo Tree Search) finetuning approach to improve decision making.

## Files

- **Part1-2.ipynb**  
  A Jupyter Notebook that demonstrates experiments, analysis, and results on the water supply simulation environment.

- **part 3.ipynb**  
  A Jupyter Notebook that demonstrates experiments, analysis, and results of the novel approach on the water supply environment.

- **qlearning.py**  
  Contains a tabular Q-learning implementation along with a `TabularQ` class.

- **env_nosell.py**  
  Implements the water supply simulation environment using Gymnasium. It includes:
  - The `SimplifiedWaterSupplyEnv` class that simulates water demand, pricing, and supply constraints.
  - Several wrappers for discretizing and normalizing observations and actions, making the environment suitable for reinforcement learning.

- **mcts_finetuned_model.py**  
  Implements a Monte Carlo Tree Search (MCTS) finetuning approach for the Q-learning agent.

## Setup & Installation

Ensure you have the following Python packages installed:

- numpy
- gymnasium
- stable-baselines3
- tqdm
- optuna
- rliable

You can install these packages using pip:

```bash
pip install numpy gymnasium stable-baselines3 tqdm optuna rliable
