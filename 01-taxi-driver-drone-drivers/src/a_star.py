""" 
Module providing the A* search algorithm for path planning.
Implementation based on the following tutorial:
https://www.geeksforgeeks.org/a-search-algorithm/
Accessed on 2024-02-18.
"""

import heapq
import numpy as np
from src.cell import Cell

def is_valid(row: int, col: int, max_row: int, max_col: int):
    """ Check if a cell is valid (within the grid). """
    return (0 <= row < max_row) and (0 <= col < max_col)

def is_unblocked(map_matrix: np.array, row: int, col: int):
    """ Check if a cell is unblocked. """
    return map_matrix[row][col] == 1

def is_destination(row: int, col: int, dest: tuple):
    """ Check if a cell is the destination. """
    return row == dest[0] and col == dest[1]

def calculate_heuristic_value(row: int, col: int, dest: tuple):
    """Calculate the heuristic value of a cell (Manhattan distance to destination)."""
    return abs(row - dest[0]) + abs(col - dest[1])

def trace_path(cell_details: list, dest: tuple):
    """ Trace the path from source to destination. """
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (
        cell_details[row][col].parent_i == row
        and cell_details[row][col].parent_j == col
    ):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()
    return path

def a_star_search(
        map_matrix: np.array,
        source: tuple[int],
        destination: tuple[int],
        max_row: int,
        max_col: int
    ):
    """ Implement the A* search algorithm. """
    # Check if the source and destination are valid
    if (
        not is_valid(source[0], source[1], max_row, max_col)
        or not is_valid(destination[0], destination[1], max_row, max_col)
    ):
        raise ValueError("Source or destination is invalid")

    # Check if the source and destination are unblocked
    if (
        not is_unblocked(map_matrix, source[0], source[1])
        or not is_unblocked(map_matrix, destination[0], destination[1])
    ):
        raise ValueError("Source or destination is blocked")

    # Check if we are already at the destination
    if is_destination(source[0], source[1], destination):
        return [source]

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(max_col)] for _ in range(max_row)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(max_col)] for _ in range(max_row)]

    # Initialize the start cell details
    i = source[0]
    j = source[1]
    cell_details[i][j].total_cost = 0
    cell_details[i][j].cost_from_start = 0
    cell_details[i][j].estimated_cost_to_goal = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    path_traveled = None
    while len(open_list) > 0:
        # Pop the cell with the smallest total_cost value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]
        for direction in directions:
            new_i = i + direction[0]
            new_j = j + direction[1]

            # If the successor is valid, unblocked, and not visited
            if (
                is_valid(new_i, new_j, max_row, max_col)
                and is_unblocked(map_matrix, new_i, new_j)
                and not closed_list[new_i][new_j]
            ):
                # If the successor is the destination
                if is_destination(new_i, new_j, destination):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    # Trace and print the path from source to destination
                    path_traveled = trace_path(cell_details, destination)
                    found_dest = True
                    return path_traveled
                else:
                    # Calculate the new total_cost, cost_from_start
                    # and estimated_cost_to_goal values
                    new_cost_from_start = cell_details[i][j].cost_from_start + 1.0
                    new_estimated_cost_to_goal = calculate_heuristic_value(
                        new_i,
                        new_j,
                        destination
                    )
                    new_total_cost = new_cost_from_start + new_estimated_cost_to_goal

                    # If the cell is not in the open list or the new total_cost value is smaller
                    if (
                        cell_details[new_i][new_j].total_cost == float("inf")
                        or cell_details[new_i][new_j].total_cost > new_total_cost
                    ):
                        # Add the cell to the open list
                        heapq.heappush(open_list, (new_total_cost, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].total_cost = new_total_cost
                        cell_details[new_i][new_j].cost_from_start = new_cost_from_start
                        cell_details[new_i][new_j].estimated_cost_to_goal = new_estimated_cost_to_goal
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        raise ValueError("Destination cell is not found")
