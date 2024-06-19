""" This module contains tests for the solution of the problem. """

import os
import pytest
from src.map_generator import generate_random_map
from src.input_parser import InputParser
from main import solve_map

NUM_RANDOM_MAPS = 10

def is_valid(row, col, max_row, max_col):
    """ Check if a cell is valid (within the grid). """
    return (row >= 0) and (row < max_row) and (col >= 0) and (col < max_col)

def is_unblocked(grid, row, col):
    """ Check if a cell is unblocked. """
    return grid[row][col] == 1

def path_solves(path_algorithm, map_data) -> bool:
    """ Check if the path solves the map. """
    for position in path_algorithm:
        is_valid_position = is_valid(*position, map_data.lines, map_data.columns)
        if not is_valid_position or not is_unblocked(map_data.map_matrix, *position):
            return False
    return True


def is_solution_correct(file_path: str):
    """ Check if the solution is correct. """
    map_data = parse_test_map(file_path)
    path_pickup, path_dropoff = solve_map(file_path, verbose=False)
    assert map_data.taxi_initial_position == path_pickup[0]
    assert map_data.passenger_initial_position == path_pickup[-1]
    assert map_data.drop_off_point == path_dropoff[-1]
    assert path_solves(path_pickup, map_data) is True
    assert path_solves(path_dropoff, map_data) is True




def parse_test_map(map_path: str = "maps/map_1.txt"):
    """ Parse the map from the input file. """
    input_parser = InputParser()
    map_data = input_parser.parse_from_input(map_path)
    return map_data


def test_map_variables_parse():
    """ Test if the map variables are parsed correctly."""
    map_data = parse_test_map()
    assert map_data.columns == 4
    assert map_data.lines == 8
    assert map_data.taxi_initial_position == (1, 2)
    assert map_data.passenger_initial_position == (7, 0)
    assert map_data.drop_off_point == (7, 2)

def test_map_1():
    """ Test if the solution is correct for map_1.txt. """
    is_solution_correct("maps/map_1.txt")

def test_map_2():
    """ Test if the solution is correct for map_2.txt. """
    is_solution_correct("maps/map_2.txt")   

def test_map_3():
    """ Test if the solution is correct for map_3.txt. """
    is_solution_correct("maps/map_3.txt")   

def test_map_4():
    """ Test if the solution is correct for map_4.txt. """
    is_solution_correct("maps/map_4.txt")   

def test_map_5():
    """ Test if the solution is correct for map_5.txt. """
    is_solution_correct("maps/map_5.txt")   


def test_random_maps():
    """ Test if the solution is correct for random maps. """
    for i in range(NUM_RANDOM_MAPS):
        filename = f"map_{i}_temp.txt"
        generate_random_map(
            folder_name="maps", min_size=5, max_size=15, file_name=filename
        )
        is_solution_correct("maps/" + filename)
        os.remove("maps/" + filename)
