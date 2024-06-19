import gymnasium as gym
from numpy import argmax, loadtxt
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters
from src.q_learning import QLearning
from src.sarsa import Sarsa

def train(
        hyperparameters: Hyperparameters,
        parameters: Parameters,
        algorithm_choice: str
    ):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).env

    algorithm = None
    parameters.update_filename(algorithm_choice.lower())
    if algorithm_choice == "Q-Learning":
        algorithm = QLearning(
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )
    if algorithm_choice == "Sarsa":
        algorithm = Sarsa(
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )

    algorithm.train()

    env.close()

    return 0

def test(
        parameters: Parameters
    ):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).env
    q_table = loadtxt(f"{parameters.data_dir}/q-table-{parameters.filename}.csv", delimiter=",")

    # Evaluate the agent for 100 episodes
    rewards = 0
    for _ in range(100):
        (state, _) = env.reset()
        epochs = 0
        truncated = False
        done = False

        while (not done) and (epochs < 100) and (not truncated):
            action = argmax(q_table[state])
            state, reward, done, truncated, _ = env.step(action)
            epochs += 1
            rewards += reward

    env.close()
    return int(rewards)
