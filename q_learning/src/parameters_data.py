"""
Module to store parameters for the model
"""

from dataclasses import dataclass

@dataclass
class Parameters:
    """
    Class to store parameters for the model
    """
    filename: str
    train: bool
    episodes: int
    randomize_actions: bool
    only_exploit: bool
    data_dir: str
    results_dir: str
