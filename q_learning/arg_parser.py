"""
Module for parsing command line arguments
"""

import argparse

def parse_args(args: list[str] = None) -> argparse.Namespace:
    """
    Parse the command line arguments.
    :param args: list of command line arguments
    :return: parsed arguments
    """

    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument(
        "--input",
        help="Input file",
        required=False,
        type=str,
        default="data/q-table-taxi-driver.csv"
    )
    parser.add_argument(
        "--output",
        help="Output file",
        required=False,
        type=str,
        default="results/actions_taxidriver"
    )
    parser.add_argument(
        "--train",
        help="Train the model",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--episodes",
        help="Number of episodes",
        required=False,
        type=int,
        default=50000
    )
    parser.add_argument(
        "--alpha",
        help="Learning rate",
        required=False,
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--gamma",
        help="Discount factor",
        required=False,
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--epsilon",
        help="Exploration rate",
        required=False,
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--epsilon-min",
        help="Minimum exploration rate",
        required=False,
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--epsilon-dec",
        help="Exploration decay rate",
        required=False,
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--random",
        help="Randomize all agent actions",
        required=False,
        action="store_true"
    )

    parser.add_argument(
        "--only-exploit",
        help="Only exploit the Q-table",
        required=False,
        action="store_true"
    )

    return parser.parse_args(args)
