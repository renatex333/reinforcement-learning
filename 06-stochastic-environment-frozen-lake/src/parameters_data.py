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
    episodes: int
    data_dir: str
    results_dir: str
    filename: str = None

    def __post_init__(self):
        self.filename = self.base_filename

    def update_base_filename(self, name: str) -> None:
        """
        Update the base filename
        """
        self.base_filename = name.lower()

    def update_filename(self, name: str) -> None:
        """
        Update the filename based on the algorithm
        """
        if self.base_filename:
            self.filename = f"{name.lower()}-{self.base_filename}"
    