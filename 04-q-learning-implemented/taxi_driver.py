"""
Taxi Driver Problem - Q-Learning
"""

import sys
from time import sleep
import gymnasium as gym
from IPython.display import clear_output
from numpy import loadtxt, argmax
from src.q_learning import QLearning
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters
from src.arg_parser import parse_args

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

def main(
        hyperparameters: Hyperparameters,
        parameters: Parameters
    ):
    """
    Main function
    """
    env = gym.make("Taxi-v3", render_mode="ansi").env

    q_table = None
    if parameters.train:
        print(f"Training the agent for {parameters.episodes} episodes.")
        qlearn = QLearning(
            env,
            hyperparameters=hyperparameters,
            parameters=parameters
        )
        q_table = qlearn.train()
    else:
        print("Loading the Q-table from file.")
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

    clear_output(wait=True)

    print_frames(frames)

    print("\n")
    print(f"Timesteps taken: {epochs}")
    print(f"Penalties incurred: {penalties}")

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
    main(
        hyperparameters=hyperparameters_wrapper,
        parameters=parameters_wrapper
    )
