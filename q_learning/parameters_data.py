"""
Module to store parameters for the model
"""

from dataclasses import dataclass

@dataclass
class Parameters:
    """
    Class to store parameters for the model
    """
    input_file: str
    output_file: str
    train: bool
    episodes: int
    randomize_actions: bool
    only_exploit: bool
