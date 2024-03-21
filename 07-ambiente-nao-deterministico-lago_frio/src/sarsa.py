"""
Module to implement the Sarsa algorithm.
"""

from src.algorithm import Algorithm

class Sarsa(Algorithm):
    """
    Class to implement the Sarsa algorithm.
    Can be used to create agents to act in some Gymansyium project environments.
    """

    def train_episode(self):
        (state, _) = self.env.reset()
        rewards = 0
        done = False
        num_steps = 0
        action = self.select_action(state)

        while not done:
            num_steps += 1
            new_state, reward, done, _, _ = self.env.step(action)
            new_action = self.select_action(new_state)
            self.q_table[state, action] = self.update_q_table(
                state, action, reward, new_state, new_action
            )
            state = new_state
            action = new_action
            rewards += reward

        return rewards, num_steps

    def update_q_table(self, state, action, reward, new_state, new_action):
        """
        Q(s,a) -> Q(s,a) + alpha [r + gamma Q(s', a') - Q(s,a)]
        """
        return (
            self.q_table[state, action]
            + self.hyperparameters.alpha
            * (
                reward
                + (self.hyperparameters.gamma * self.q_table[new_state, new_action])
                - self.q_table[state, action]
            )
        )
