"""
Module to implement the hyperparameters dataclass.
"""

from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """
    Dataclass to store the hyperparameters used in the Q-Learning algorithm.
    """
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_dec: float

    def __post_init__(self):
        self.epsilon = self.epsilon_start

    def set_epsilon(self, epsilon):
        """
        Method to update the epsilon value.
        """
        self.epsilon = epsilon
