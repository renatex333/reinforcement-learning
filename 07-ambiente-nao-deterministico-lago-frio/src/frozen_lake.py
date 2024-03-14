"""
Taxi Driver Problem - Q-Learning
"""

import sys
import gymnasium as gym
from numpy import loadtxt, argmax
from src.q_learning import QLearning
from src.sarsa import Sarsa
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters
from src.arg_parser import parse_args

def frozen_lake(
        hyperparameters: Hyperparameters,
        parameters: Parameters,
        algorithm_name: str = "q-learning"
    ) -> int:
    """
    Main function
    """
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).env

    algorithms = {
        "q-learning": QLearning,
        "sarsa": Sarsa
    }
    if parameters.train:
        print(f"Training the agent for {parameters.episodes} episodes.")
        algorithm = algorithms[algorithm_name](
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )
        algorithm.train()
    else:
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

        return int(rewards)
    return 0

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    hyperparameters_wrapper = Hyperparameters(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_dec=args.epsilon_dec
    )
    parameters_wrapper = Parameters(
        filename=args.filename,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        train=args.train,
        episodes=args.episodes,
        randomize_actions=args.random,
        only_exploit=args.only_exploit
    )
    frozen_lake(
        hyperparameters=hyperparameters_wrapper,
        parameters=parameters_wrapper,
        algorithm_name=args.algorithm
    )
