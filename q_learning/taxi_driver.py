"""
Taxi Driver Problem - Q-Learning

The Taxi Problem is a problem described in the "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" paper by Tom Dietterich. The problem is as follows:
    
        There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    
        There are six primitive actions in this environment:
    
            - south
            - north
            - east
            - west
            - pickup
            - dropoff
    
        There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.

The environment is a 5x5 grid, with four locations marked as R(ed), G(reen), Y(ellow), and B(lue). The taxi operates in this environment, and the passenger is picked up and dropped off at one of the four locations. The taxi has to learn to move to the passenger's location, pick up the passenger, move to the destination, and drop off the passenger.

Using a Q-Learning algorithm, the agent learns to act in the environment and is then evaluated.

This script is the main script to run the Q-Learning algorithm for the Taxi Driver problem.

Usage:

    python taxi_driver.py --train --episodes 50000 --input "q_table.csv" --output "q_table.csv" --alpha 0.1 --gamma 0.99 --epsilon 0.7 --epsilon_min 0.05 --epsilon_dec 0.99
    
    """

import sys
from time import sleep
import gymnasium as gym
from IPython.display import clear_output
from numpy import loadtxt, argmax
from q_learning import QLearning
from hyperparameters_data import Hyperparameters
from parameters_data import Parameters
from arg_parser import parse_args

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
        q_table = qlearn.train(parameters.input_file, parameters.output_file)
    else:
        print(f"Loading the Q-table from {parameters.input_file}.")
        q_table = loadtxt(parameters.input_file, delimiter=",")

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
        input_file=args.input,
        output_file=args.output,
        train=args.train,
        episodes=args.episodes,
        randomize_actions=args.random,
        only_exploit=args.only_exploit
    )
    main(
        hyperparameters=hyperparameters_wrapper,
        parameters=parameters_wrapper
    )
