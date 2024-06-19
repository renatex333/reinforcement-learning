""" This module contains the Cell class which is used to represent a cell in the grid."""

import dataclasses

@dataclasses.dataclass
class Cell:
    """ Class representing a cell in the grid. """
    parent_i: int = 0  # Parent cell's row index
    parent_j: int = 0  # Parent cell's column index
    # Total cost of the cell (cost from start + estimated cost to goal)
    total_cost: float = float("inf")
    # Cost from start to this cell
    cost_from_start: float = float("inf")
    # Heuristic cost from this cell to destination
    estimated_cost_to_goal: int = 0
