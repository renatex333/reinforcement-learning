"""
Module to implement the Q-Learning algorithm.
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from hyperparameters_data import Hyperparameters
from parameters_data import Parameters

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
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.hyperparameters = hyperparameters
        self.episodes = parameters.episodes
        self.randomize_actions = parameters.randomize_actions
        self.only_exploit = parameters.only_exploit

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
        return np.argmax(self.q_table[state])

    def plot_actions(self, plot_file, actions_per_episode):
        """
        Plots the number of actions taken per episode.
        """
        plt.plot(actions_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("# Actions")
        plt.title("# Actions vs Episodes")
        plt.savefig(plot_file+".jpg")
        plt.close()

    def train(self, filename, plot_file):
        """
        Trains the agent to act in the environment.
        """
        actions_per_episode = []
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
                        + (self.hyperparameters.gamma * np.max(self.q_table[new_state]))
                        - self.q_table[state, action]
                    )
                )
                state = new_state
                actions=actions+1
                rewards=rewards+reward

            actions_per_episode.append(actions)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +"\r")
                sys.stdout.flush()

            if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
                self.hyperparameters.epsilon = (
                    self.hyperparameters.epsilon
                    * self.hyperparameters.epsilon_dec
                )

        np.savetxt(filename, self.q_table, delimiter=",")
        if plot_file is not None:
            self.plot_actions(plot_file, actions_per_episode)
        return self.q_table
