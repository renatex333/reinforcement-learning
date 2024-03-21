"""
Module to implement the Q-Learning algorithm.
"""

from numpy import amax
from src.algorithm import Algorithm

class QLearning(Algorithm):
    """
    Class to implement the Q-Learning algorithm.
    Can be used to create agents to act in some Gymansyium project environments.
    """

    def train_episode(self):
        (state, _) = self.env.reset()
        rewards = 0
        done = False
        num_steps = 0

        while not done:
            num_steps += 1
            action = self.select_action(state)
            new_state, reward, done, _, _ = self.env.step(action)
            self.q_table[state, action] = self.update_q_table(
                state, action, reward, new_state
            )
            state = new_state
            rewards += reward

        return rewards, num_steps

    def update_q_table(self, state, action, reward, new_state):
        """
        Q(s,a) -> Q(s,a) + alpha * [r + gamma max(Q(s',A')) - Q(s,a)]
        """
        return (
            self.q_table[state, action]
            + self.hyperparameters.alpha
            * (
                reward
                + (self.hyperparameters.gamma * amax(self.q_table[new_state]))
                - self.q_table[state, action]
            )
        )
