"""
Module to implement the Minimax algorithm
"""

from copy import copy
import numpy as np

def get_possible_moves(action_mask: np.ndarray) -> np.ndarray:
    """
    Get the possible moves from the given observation
    :param action_mask: action mask of the current state
    """
    return np.where(action_mask == 1)[0]

def evaluate(obs: np.ndarray) -> tuple[int, bool]:
    """
    Function to evaluate the current state of the game.
    Evaluates relative to the player that called the function.
    :param obs: observation of the current state
    :return: reward and if the game is finished
    """
    win_cases_player = np.array([
        obs[0, :, 0],  # first row
        obs[1, :, 0],  # second row
        obs[2, :, 0],  # third row
        obs[:, 0, 0],  # first column
        obs[:, 1, 0],  # second column
        obs[:, 2, 0],  # third column
        np.diag(obs[:, :, 0]),  # main diagonal
        np.diag(np.fliplr(obs[:, :, 0]))  # anti-diagonal
    ])

    win_cases_enemy = np.array([
        obs[0, :, 1],  # first row
        obs[1, :, 1],  # second row
        obs[2, :, 1],  # third row
        obs[:, 0, 1],  # first column
        obs[:, 1, 1],  # second column
        obs[:, 2, 1],  # third column
        np.diag(obs[:, :, 1]),  # main diagonal
        np.diag(np.fliplr(obs[:, :, 1]))  # anti-diagonal
    ])

    if np.any(np.sum(win_cases_player, axis=1) == 3):
        return 1, True
    if np.any(np.sum(win_cases_enemy, axis=1) == 3):
        return -1, True
    if np.sum(obs[:, :, 0] + obs[:, :, 1]) == 9:
        return 0, True
    return 0, False

def update_obs(obs: np.ndarray, action_mask: np.ndarray, action: int, player: bool) -> np.ndarray:
    """
    Update the observation with the new action.
    :param obs: observation of the current state
    :param action: action to be played
    :param player: player that is playing
    :return: new observation
    """

    new_obs = copy(obs)
    new_action_mask = copy(action_mask)
    new_action_mask[action] = 0
    if player:
        new_obs[action // 3, action % 3, 0] = 1

    else:
        new_obs[action // 3, action % 3, 1] = 1

    return new_obs, new_action_mask

def minimax(obs: np.ndarray, action_mask: np.ndarray, depth: int, is_maximizing: bool) -> int:
    """
    Minimax algorithm to find the best move.
    :param obs: observation of the current state
    :param action_mask: mask of possible actions
    :param depth: depth of the tree
    :param is_maximizing: if the current player is maximizing
    :return: best action value to be played
    """
    reward, finished = evaluate(obs)

    if finished or depth == 0:
        return reward

    if is_maximizing:
        max_eval = float('-inf')
        for move in get_possible_moves(action_mask):
            # Faz uma cópia do estado atual para simular o movimento sem alterar o estado original
            observation = np.copy(obs)
            action_mask_copy = np.copy(action_mask)

            # Atualiza o estado com o movimento simulado
            observation, action_mask_copy = update_obs(observation, action_mask_copy, move, is_maximizing)

            eval = minimax(observation, action_mask_copy, depth - 1, False)
            max_eval = max(max_eval, eval)

        return max_eval

    else:
        min_eval = float('inf')
        for move in get_possible_moves(action_mask):
            # Faz uma cópia do estado atual para simular o movimento sem alterar o estado original
            observation = np.copy(obs)
            action_mask_copy = np.copy(action_mask)

            # Atualiza o estado com o movimento simulado
            observation, action_mask_copy = update_obs(observation, action_mask_copy, move, is_maximizing)

            eval = minimax(observation, action_mask_copy, depth - 1, True)
            min_eval = min(min_eval, eval)

        return min_eval
