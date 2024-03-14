"""
Module to implement the Q-Learning algorithm.
"""

import os
import sys
import random
import matplotlib.pyplot as plt
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
        self.parameters = parameters

    def train(self):
        """
        Trains the agent to act in the environment.
        """
        rewards_per_episode = []
        for i in range(1, self.parameters.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False

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
                rewards=rewards+reward

            rewards_per_episode.append(rewards)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +"\r")
                sys.stdout.flush()

            if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
                self.hyperparameters.epsilon = (
                    self.hyperparameters.epsilon
                    * self.hyperparameters.epsilon_dec
                )

        # self.plot_rewards(rewards_per_episode)

        if not os.path.isdir(self.parameters.data_dir):
            os.makedirs(self.parameters.data_dir)
        savetxt(
            f"{self.parameters.data_dir}/q-table-{self.parameters.filename}.csv",
            self.q_table,
            delimiter=","
        )

        savetxt(
            f"{self.parameters.data_dir}/rewards-{self.parameters.filename}.csv",
            rewards_per_episode,
            delimiter=","
        )

        return self.q_table

    def select_action(self, state):
        """
        Selects an action to be taken by the agent.
        Depending on the value of epsilon, it can either:
        - Explore the action space or
        - Exploit the Q-table.
        """
        rv = random.uniform(0, 1)
        if not self.parameters.only_exploit \
            and (self.parameters.randomize_actions or rv < self.hyperparameters.epsilon):
            return self.env.action_space.sample()
        return argmax(self.q_table[state])

    def plot_rewards(self, rewards_per_episode):
        """
        Plots the number of actions taken per episode and the rewards per episode
        to visualize the learning process.
        """
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Number of Episodes")
        ax1.set_ylabel("Received Rewards", color=color)
        ax1.plot(rewards_per_episode, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()

        if not os.path.isdir(self.parameters.results_dir):
            os.makedirs(self.parameters.results_dir)
        plt.savefig(f"{self.parameters.results_dir}/plot-learning-{self.parameters.filename}.jpg")
        plt.close()
