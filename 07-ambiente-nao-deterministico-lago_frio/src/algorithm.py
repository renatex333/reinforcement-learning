"""
Module to implement the Algorithm Superclass.
"""

import os
import random
import pandas as pd
from time import perf_counter
from numpy import zeros, argmax, savetxt
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

class Algorithm():
    """
    Superclass to implement different algorithms based on Q-Table.
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
        self.algorithm = self.__class__.__name__

    def train(self) -> None:
        df = pd.DataFrame(
            columns=["rewards", "actions"],
            index=range(1, self.parameters.episodes + 1)
        )
        df.index.name = "episode"

        start_time = perf_counter()
        for i in range(1, self.parameters.episodes + 1):
            rewards, num_steps = self.train_episode()

            df.loc[i, "rewards"] = rewards
            df.loc[i, "actions"] = num_steps

            if self.hyperparameters.epsilon > self.hyperparameters.epsilon_min:
                self.hyperparameters.set_epsilon(
                    self.hyperparameters.epsilon
                    * self.hyperparameters.epsilon_dec
                )

            if i % 100 == 0:
                print(f"Algorithm: {self.algorithm}, Episode: {i}", end="\r")

        end_time = perf_counter()
        print(f"Algorithm: {self.algorithm}, Episode: {i}, Time: {end_time - start_time:.2f} seconds")
        if not os.path.isdir(self.parameters.data_dir):
            os.makedirs(self.parameters.data_dir)
        df.to_feather(
            f"{self.parameters.data_dir}/{self.parameters.filename}.feather"
        )
        savetxt(
            f"{self.parameters.data_dir}/q-table-{self.parameters.filename}.csv",
            self.q_table,
            delimiter=","
        )

    def train_episode(self) -> tuple[float, int]:
        return 1.0, 1

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
