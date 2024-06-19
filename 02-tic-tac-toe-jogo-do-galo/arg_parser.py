"""
Module to implement the parser for the command line arguments.
"""

import argparse

def parse_args(args: list[str] = None) -> argparse.Namespace:
    """
    Parse the command line arguments.
    :param args: list of command line arguments
    :return: parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Play the game of tic-tac-toe using the PettingZoo library."
    )
    parser.add_argument("-r", "--render", type=bool, default=False, help="Render the game interface")
    parser.add_argument("-n", "--episodes", type=int, default=1, help="Number of episodes to play")

    return parser.parse_args(args)
