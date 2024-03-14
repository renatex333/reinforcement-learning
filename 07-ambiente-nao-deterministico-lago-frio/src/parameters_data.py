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
    data_dir: str
    results_dir: str
    episodes: int = 0
    randomize_actions: bool = False
    only_exploit: bool = False
