"""
Module to store parameters for the model
"""

from dataclasses import dataclass

@dataclass
class Parameters:
    """
    Class to store parameters for the model
    """
    base_filename: str
    filename: str
    train: bool
    episodes: int
    randomize_actions: bool
    only_exploit: bool
    data_dir: str
    results_dir: str

    def update_filename(self, name: str) -> None:
        """
        Update the filename based on the algorithm
        """
        if self.only_exploit:
            self.filename = f"{name.lower()}-only-exploit"
        elif self.randomize_actions:
            self.filename = f"{name.lower()}-random"
        else:
            self.filename = f"{name.lower()}"

        if self.base_filename:
            self.filename = f"{self.filename}-{self.base_filename}"
