"""
Main module to call the Q-Learning and SARSA algorithms for Taxi Driver or Cliff Walking problems.
"""

import sys
from numpy import loadtxt
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters
from src.arg_parser import parse_args
from src.taxi_driver import taxi_driver
from src.cliff_walking import cliff_walking
from src.plotter import plot_learning_curve

def main(
        hyperparameters: Hyperparameters,
        parameters: Parameters
    ) -> int:
    """
    Main function to call the Q-Learning and SARSA algorithms
    for Taxi Driver or Cliff Walking problems.
    """
    choice = int(input("Enter 1 for Taxi Driver or 2 for Cliff Walking: "))
    if choice not in [1, 2]:
        raise ValueError(f"Invalid choice {choice}. Must be 1 or 2.")
    verbose = input("Verbose mode? (y/N): ").lower() == "y"
    problem = "Taxi-Driver" if choice == 1 else "Cliff-Walking"
    problem_call = taxi_driver if choice == 1 else cliff_walking

    # Q-Learning
    print("Q-Learning")
    parameters.update_filename(f"Q-Learning-{problem}")
    problem_call(
        hyperparameters=hyperparameters,
        parameters=parameters,
        algorithm_choice="Q-Learning",
        verbose=verbose
    )
    qlearning = loadtxt(
        f"{parameters.data_dir}/rewards-{parameters.filename}.csv",
        delimiter=","
    )

    print("*" * 80)

    # Sarsa
    print("Sarsa")
    parameters.update_filename(f"Sarsa-{problem}")
    problem_call(
        hyperparameters=hyperparameters,
        parameters=parameters,
        algorithm_choice="Sarsa",
        verbose=verbose
    )
    sarsa = loadtxt(
        f"{parameters.data_dir}/rewards-{parameters.filename}.csv",
        delimiter=","
    )

    # Plot the learning curve
    return plot_learning_curve(
        problem_name=problem,
        qlearning=qlearning,
        sarsa=sarsa,
        parameters=parameters
    )

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
        base_filename=args.base_filename,
        filename=args.base_filename,
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
