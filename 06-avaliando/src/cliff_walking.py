"""
Cliff Walking Problem - Q-Learning and Sarsa
"""

import gymnasium as gym
from src.q_learning import QLearning
from src.sarsa import Sarsa
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

def cliff_walking(
        hyperparameters: Hyperparameters,
        parameters: Parameters,
        algorithm_choice: str,
        verbose: bool = False
    ):
    """
    Function to implement the Taxi Driver problem using Q-Learning and Sarsa.
    """
    render_mode = "human" if (verbose and not parameters.train) else "ansi"
    env = gym.make("CliffWalking-v0", render_mode=render_mode).env

    algorithm = None
    if algorithm_choice == "Q-Learning":
        parameters.update_filename("q-learning")
        algorithm = QLearning(
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )
    if algorithm_choice == "Sarsa":
        parameters.update_filename("sarsa")
        algorithm = Sarsa(
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )

    algorithm.train()

    env.close()
