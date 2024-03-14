"""
Module to implement the Q-Learning algorithm.
"""

import os
import sys
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import zeros, argmax, amax, savetxt
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

class QLearning:
    """
    Class to implement the Q-Learning algorithm.
    Can be used to create agents to act in some Gymansyium project environments.
    """

    def __init__(
            self,
            env,
            hyperparameters: Hyperparameters,
            parameters: Parameters
        ):
        self.env = env
        self.q_table = zeros([env.observation_space.n, env.action_space.n])
        self.hyperparameters = hyperparameters
        self.episodes = parameters.episodes
        self.randomize_actions = parameters.randomize_actions
        self.only_exploit = parameters.only_exploit
        self.filename = parameters.filename
        self.data_dir = parameters.data_dir
        self.results_dir = parameters.results_dir

    def select_action(self, state):
        """
        Selects an action to be taken by the agent.
        Depending on the value of epsilon, it can either:
        - Explore the action space or
        - Exploit the Q-table.
        """
        rv = random.uniform(0, 1)
        if not self.only_exploit and (self.randomize_actions or rv < self.hyperparameters.epsilon):
            return self.env.action_space.sample()
        return argmax(self.q_table[state])

    def plot_actions(self, actions_per_episode):
        """
        Plots the number of actions taken per episode.
        """
        plt.plot(actions_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("# Actions")
        plt.title("# Actions vs Episodes")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/actions-{self.filename}.jpg")
        plt.close()

    def plot_rewards(self, rewards_per_episode):
        """
        Plots the number of actions taken per episode.
        """
        plt.plot(rewards_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Rewards vs Episodes")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/rewards-{self.filename}.jpg")
        plt.close()

    def plot_training(self, rewards_per_episode, actions_per_episode):
        """
        Plots the the number of actions taken per episode and the rewards per episode
        to visualize the training process.
        """
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Number of Episodes")
        ax1.set_ylabel("Received Rewards", color=color)
        ax1.plot(rewards_per_episode, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel("Number of Actions", color=color)
        ax2.plot(actions_per_episode, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()

        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/plot-{self.filename}.jpg")
        plt.close()

    def plot_q_table(self, q_table_states):
        """
        Plots the Q-table as a heatmap to visualize the training process.
        """
        fig, ax = plt.subplots(ncols=3, figsize=(8, 8))
        ax[0] = sns.heatmap(self.q_table, ax=ax[0], cmap="hot", cbar=False)
        ax[0].set_title("Q-Table Initial")
        ax[1] = sns.heatmap(q_table_states[len(q_table_states) // 2], ax=ax[1], cmap="hot", cbar=False)
        ax[1].set_title("Q-Table Middle")
        ax[2] = sns.heatmap(q_table_states[-1], ax=ax[2], cmap="hot", cbar=False)
        ax[2].set_title("Q-Table Final")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        plt.savefig(f"{self.results_dir}/q-table-heatmap-{self.filename}.jpg")
        plt.close()

    def save_partial_data(self, data, filename):
        """
        Saves the data to a file.
        """
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        savetxt(f"{self.data_dir}/{filename}.csv", data, delimiter=",")

    def train(self):
        """
        Trains the agent to act in the environment.
        """
        actions_per_episode = []
        rewards_per_episode = []
        q_table_states = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0

            while not done:
                action = self.select_action(state)
                new_state, reward, done, _, _ = self.env.step(action)
                # Q(s,a) -> Q(s,a) + alpha * [r + gamma max(Q(s',A')) - Q(s,a)]
                self.q_table[state, action] = (
                    self.q_table[state, action]
                    + self.hyperparameters.alpha
                    * (
                        reward
                        + (self.hyperparameters.gamma * amax(self.q_table[new_state]))
                        - self.q_table[state, action]
                    )
                )
                state = new_state
                actions=actions+1
                rewards=rewards+reward

            actions_per_episode.append(actions)
            rewards_per_episode.append(rewards)
            if i % 100 == 0:
                q_table_states.append(deepcopy(self.q_table))
                sys.stdout.write("Episodes: " + str(i) +"\r")
                sys.stdout.flush()

            if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
                self.hyperparameters.epsilon = (
                    self.hyperparameters.epsilon
                    * self.hyperparameters.epsilon_dec
                )

        # if not os.path.isdir(self.data_dir):
        #     os.makedirs(self.data_dir)
        # savetxt(f"{self.data_dir}/q-table-{self.filename}.csv", self.q_table, delimiter=",")
        # self.plot_actions(actions_per_episode)
        # self.plot_rewards(rewards_per_episode)
        # self.plot_training(rewards_per_episode, actions_per_episode)
        # self.plot_q_table(q_table_states)

        self.save_partial_data(rewards_per_episode, f"rewards-{self.filename}")
        self.save_partial_data(q_table_states[0], f"q-table-states-initial-{self.filename}")
        self.save_partial_data(q_table_states[len(q_table_states) // 2], f"q-table-states-middle-{self.filename}")
        self.save_partial_data(q_table_states[-1], f"q-table-states-final-{self.filename}")

        return self.q_table
