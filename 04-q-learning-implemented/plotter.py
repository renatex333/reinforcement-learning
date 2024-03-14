"""
Module to create the plots needed.
"""

import os
from random import randint
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import loadtxt

class Plotter:
    """
    Plotter class to create the plots needed.
    """
    def __init__(self):
        self.data_dir = "data"
        self.results_dir = "results"

    def read_q_tables(self):
        """
        Function to read the Q-table from files.
        """
        states = ["initial", "middle", "final"]
        files = ["alpha_0_01", "epsilon_0", "epsilon_1", "gamma_0_95"]
        q_tables = []
        for state in states:
            q_tables.append(loadtxt(f"{self.data_dir}/q-table-states-{state}-{files[0]}.csv", delimiter=","))
        return q_tables
    
    def read_rewards(self):
        """
        Function to read the rewards from files.
        """
        rewards = []
        files = ["alpha_0_01", "epsilon_1", "gamma_0_95"]
        for file in files:
            rewards.append(loadtxt(f"{self.data_dir}/rewards-{file}.csv", delimiter=","))
        return rewards

    def plot_graph(self, data):
        """
        Function to plot the graph.
        """
        # labels = ["Alpha 0.01", "Random Actions", "Q-Table Exploitation", "Gamma 0.95"]
        labels = ["Alpha 0.01", "Q-Table Exploitation", "Gamma 0.95"]
        for i, slice in enumerate(data):
            plt.plot(slice, label=f"{labels[i]}")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Reward vs Episodes")
        plt.xlim(0, 50001)
        plt.ylim(-200, 50)
        plt.legend(loc="best")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/rewards-vs-episodes-plot.jpg")
        plt.close()

    def plot_heatmap(self, data):
        """
        Function to plot the heatmap.
        """
        state = randint(0, len(data[0])-1)
        states = ["Initial", "Middle", "Final"]
        actions_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        actions = [0, 1, 2, 3, 4, 5]
        fig, ax = plt.subplots(ncols=3, figsize=(12, 12))
        ax[0] = sns.heatmap([actions, data[0][state]], ax=ax[0], cmap="hot", cbar=False, xticklabels=actions_labels, yticklabels=True)
        ax[0].set_title(f"Q-Table State {state} - {states[0]}")
        ax[1] = sns.heatmap([actions, data[1][state]], ax=ax[1], cmap="hot", cbar=False, xticklabels=actions_labels, yticklabels=True)
        ax[1].set_title(f"Q-Table State {state} - {states[1]}")
        ax[2] = sns.heatmap([actions, data[2][state]], ax=ax[2], cmap="hot", cbar=False, xticklabels=actions_labels, yticklabels=True)
        ax[2].set_title(f"Q-Table State {state} - {states[2]}")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/q-table-heatmap-state-{state}.jpg")
        plt.close()

def main():
    """
    Main function
    """
    plotter = Plotter()
    q_tables = plotter.read_q_tables()
    rewards = plotter.read_rewards()
    plotter.plot_graph(rewards)
    plotter.plot_heatmap(q_tables)

if __name__ == "__main__":
    main()