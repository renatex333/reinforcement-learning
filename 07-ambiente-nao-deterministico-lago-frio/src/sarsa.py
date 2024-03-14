"""
Module to implement the Q-Learning algorithm.
"""

import os
import sys
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import zeros, argmax, amax, savetxt
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

class Sarsa:
    """
    Class to implement the Sarsa algorithm.
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

    def train(self):
        """
        Trains the agent to act in the environment.
        """
        df = pd.DataFrame(
            columns=["rewards", "actions"],
            index=range(1, self.episodes + 1)
        )
        actions_per_episode = []
        rewards_per_episode = []
        q_table_states = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0
            action = self.select_action(state)

            while not done:
                new_state, reward, done, _, _ = self.env.step(action)
                new_action = self.select_action(new_state)
                # Q(s,a) -> Q(s,a) + alpha * [r + gamma(Q(s',a')) - Q(s,a)]
                self.q_table[state, action] += (
                    self.hyperparameters.alpha
                    * (
                        reward
                        + (self.hyperparameters.gamma * (self.q_table[new_state, new_action]))
                        - self.q_table[state, action]
                    )
                )
                state = new_state
                action = new_action
                actions=actions+1
                rewards=rewards+reward
                df.loc[i, "rewards"] = rewards
                df.loc[i, "actions"] = actions

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

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        df.to_feather(
            f"{self.data_dir}/{self.filename}.feather"
        )
        savetxt(f"{self.data_dir}/q-table-{self.filename}.csv", self.q_table, delimiter=",")
        return self.q_table
