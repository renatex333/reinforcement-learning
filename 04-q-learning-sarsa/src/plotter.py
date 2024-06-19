"""
Module to implement plotter for Q-Learning and SARSA learning curves.
"""

import os
import matplotlib.pyplot as plt
from numpy import cumsum, ndarray
from src.parameters_data import Parameters

# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy?newreg=184191285a40426480ef44312c6679ea
def moving_average(data: ndarray, window_size: int = 3):
    ret = cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def plot_learning_curve(
        problem_name: str,
        qlearning: ndarray,
        sarsa: ndarray,
        parameters: Parameters
    ) -> int:
    """
    Plots the learning curve.
    """
    # Calculate the rolling average of 50 values for both series
    qlearning_avg = moving_average(data=qlearning, window_size=50)
    sarsa_avg = moving_average(data=sarsa, window_size=50)

    # Plotting the rolling average series
    plt.plot(qlearning_avg, label="Q-Learning")
    plt.plot(sarsa_avg, label="SARSA")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.ylim(-250, 50)
    plt.title(f"Q-Learning vs SARSA for {problem_name} problem")
    plt.legend()

    if not os.path.isdir(parameters.results_dir):
        os.makedirs(parameters.results_dir)
    parameters.update_filename(problem_name.lower().replace(" ", "-"))
    plt.savefig(f"{parameters.results_dir}/plot-learning-{parameters.filename}.jpg")
    plt.close()

    return 0
