"""
Main file to play the game of tic-tac-toe using the PettingZoo library.

This project aims to implement the minimax algorithm to play the game
of tic-tac-toe using the PettingZoo library.

Authors:
* Pedro Henrique Britto Aragão Andrade
* Renato Laffranchi Falcão 

Reference: https://pettingzoo.farama.org/environments/classic/tictactoe/
"""

import os
import sys
import numpy as np
from pettingzoo.classic import tictactoe_v3
from algorithm import minimax, get_possible_moves, update_obs
from arg_parser import parse_args

def random_policy(agent: str, obs: dict, env: tictactoe_v3.env) -> int:
    """
    Simple policy to play the game randomly.
    :param agent: agent name
    :param obs: observation of the current state
    :param env: game environment
    :return: action to be played
    """
    action = env.action_space(agent).sample()
    while obs["action_mask"][action] != 1:
        action = env.action_space(agent).sample()
    return action

def human_policy() -> int:
    """
    Policy to play the game using the human input.
    :return: action to be played
    """
    action = int(input("Enter the action: "))
    return action

def minimax_policy(obs: dict) -> int:
    """
    Policy to play the game using the minimax algorithm.
    :param obs: observation dictionary containing "observation" and "action_mask"
    :return: action to be played
    """
    original_observation = obs["observation"]
    original_action_mask = obs["action_mask"]

    best_score = float('-inf')
    best_move = None

    for move in get_possible_moves(original_action_mask):
        # Cria cópias para simulação
        observation = np.copy(original_observation)
        action_mask = np.copy(original_action_mask)

        # Atualiza o estado do jogo com o movimento atual
        observation, action_mask = update_obs(observation, action_mask, move, True)

        # Assume-se que minimax retorna o melhor reward para o jogador atual
        score = minimax(observation, action_mask, 30, False)

        if score > best_score:
            best_move = move
            best_score = score

    return best_move

def main(render_mode="") -> int:
    """
    Main function to play the game
    """
    env = tictactoe_v3.env(render_mode=render_mode)
    env.reset()

    finish = False
    while not finish:
        for agent in ["player_1", "player_2"]:
            observation, _, termination, truncation, _ = env.last()
            if termination or truncation:
                finish = True
            else:
                if agent == "player_1":
                    action = random_policy(agent, observation, env)
                else:
                    action = minimax_policy(observation)
                    # action = human_policy()
                env.step(action)

    reward = env.rewards
    env.close()
    return reward

if __name__ == "__main__":
    PLAYER_1_WINS = 0
    PLAYER_2_WINS = 0
    DRAWS = 0
    args = parse_args(sys.argv[1:])
    RENDER_MODE = "human" if args.render else ""
    for i in range(args.episodes):
        rewards = main(RENDER_MODE)
        if rewards["player_1"] == 1:
            PLAYER_1_WINS += 1
        elif rewards["player_2"] == 1:
            PLAYER_2_WINS += 1
        else:
            DRAWS += 1

        os.system('cls' if os.name == 'nt' else 'clear')  # 'cls' - Windows, 'clear' - Unix/Linux
        print(f"Player 1 wins: {PLAYER_1_WINS} \nPlayer 2 wins: {PLAYER_2_WINS} \nDraws: {DRAWS}")
