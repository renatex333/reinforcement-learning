"""
Cliff Walking Problem - Q-Learning and Sarsa
"""

from time import sleep
import gymnasium as gym
from IPython.display import clear_output
from numpy import loadtxt, argmax
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

    q_table = None
    if parameters.train:
        algorithm = None
        if algorithm_choice == "Q-Learning":
            print(f"Algorithm: Q-Learning. Training the agent for {parameters.episodes} episodes.")
            algorithm = QLearning(
                env,
                hyperparameters=hyperparameters,
                parameters=parameters
            )
        elif algorithm_choice == "Sarsa":
            print(f"Algorithm: Sarsa. Training the agent for {parameters.episodes} episodes.")
            algorithm = Sarsa(
                env,
                hyperparameters=hyperparameters,
                parameters=parameters
            )
        else:
            print(f"Algorithm: Q-Learning. Training the agent for {parameters.episodes} episodes.")
            algorithm = QLearning(
                env,
                hyperparameters=hyperparameters,
                parameters=parameters
            )
        q_table = algorithm.train()
    else:
        print("Loading the Q-table from file.")
        q_table = loadtxt(f"{parameters.data_dir}/q-table-{parameters.filename}.csv", delimiter=",")

    # Evaluate the agent after training
    (state, _) = env.reset()
    rewards = 0
    actions = 0
    done = False

    while not done:
        action = argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)

        rewards += reward
        actions += 1

    print(f"Actions taken: {actions}")
    print(f"Rewards: {rewards}")
    print("\n")
