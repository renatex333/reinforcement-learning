"""
Module to implement the Sarsa algorithm.
"""

import os
import random
import pandas as pd
from numpy import zeros, argmax
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
        self.parameters = parameters

    def train(self):
        """
        Trains the agent to act in the environment.
        """
        df = pd.DataFrame(
            columns=["rewards", "actions"],
            index=range(1, self.parameters.episodes + 1)
        )
        for i in range(1, self.parameters.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            num_steps = 0
            action = self.select_action(state)

            while not done:
                num_steps += 1
                new_state, reward, done, _, _ = self.env.step(action)
                new_action = self.select_action(new_state)
                # Q(s,a) -> Q(s,a) + alpha [r + gamma Q(s', a') - Q(s,a)]
                self.q_table[state, action] = (
                    self.q_table[state, action]
                    + self.hyperparameters.alpha
                    * (
                        reward
                        + (self.hyperparameters.gamma * self.q_table[new_state, new_action])
                        - self.q_table[state, action]
                    )
                )
                state = new_state
                action = new_action
                rewards += reward

            df.loc[i, "rewards"] = rewards
            df.loc[i, "actions"] = num_steps

            if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
                self.hyperparameters.set_epsilon(
                    self.hyperparameters.epsilon
                    * self.hyperparameters.epsilon_dec
                )

        if not os.path.isdir(self.parameters.data_dir):
            os.makedirs(self.parameters.data_dir)
        df.to_feather(
            f"{self.parameters.data_dir}/{self.parameters.filename}.feather"
        )

    def select_action(self, state):
        """
        Selects an action to be taken by the agent.
        Depending on the value of epsilon, it can either:
        - Explore the action space or
        - Exploit the Q-table.
        """
        rv = random.uniform(0, 1)
        if rv < self.hyperparameters.epsilon:
            return self.env.action_space.sample()
        return argmax(self.q_table[state])
