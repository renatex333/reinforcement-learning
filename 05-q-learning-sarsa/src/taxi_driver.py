"""
Taxi Driver Problem - Q-Learning and Sarsa
"""

from time import sleep
import gymnasium as gym
from IPython.display import clear_output
from numpy import loadtxt, argmax
from src.q_learning import QLearning
from src.sarsa import Sarsa
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

def print_frames(frames):
    """
    Function to print each frame of the animation
    """
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame["frame"])
        #print(frame["frame"].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

def taxi_driver(
        hyperparameters: Hyperparameters,
        parameters: Parameters,
        algorithm_choice: str,
        verbose: bool = False
    ):
    """
    Function to implement the Taxi Driver problem using Q-Learning and Sarsa.
    """
    env = gym.make("Taxi-v3", render_mode="ansi").env

    q_table = None
    if parameters.train:
        algorithm = None
        print(f"Algorithm: {algorithm_choice}. Training the agent for {parameters.episodes} episodes.")
        if algorithm_choice == "Q-Learning":
            algorithm = QLearning(
                env,
                hyperparameters=hyperparameters,
                parameters=parameters
            )
        elif algorithm_choice == "Sarsa":
            algorithm = Sarsa(
                env,
                hyperparameters=hyperparameters,
                parameters=parameters
            )
        else:
            raise ValueError("Invalid algorithm choice. Please choose Q-Learning or Sarsa.")
        q_table = algorithm.train()
    else:
        print(f"Algorithm: {algorithm_choice}. Loading the Q-Table from the file.")
        q_table = loadtxt(f"{parameters.data_dir}/q-table-{parameters.filename}.csv", delimiter=",")

    # Evaluate the agent after training
    (state, _) = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    frames = [] # for animation

    while (not done) and (epochs < 100):
        action = argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            "frame": env.render(),
            "state": state,
            "action": action,
            "reward": reward
        })
        epochs += 1

    if verbose and not parameters.train:
        clear_output(wait=True)
        print_frames(frames)

    print(f"Timesteps taken: {epochs}")
    print(f"Penalties incurred: {penalties}")
    print("\n")
