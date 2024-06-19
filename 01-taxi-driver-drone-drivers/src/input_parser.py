""" Module to parse the input from a txt file. """

import numpy as np
from src.map_data import MapData

class InputParser:
    """
    Class to parse the input from a txt file.
    """

    MATRIX_START_INDEX = 4

    def __init__(self) -> None:
        self.lines = 0
        self.columns = 0

    def parse_from_input(self, input_file: str) -> MapData:
        """
        Reads information from format specified in `maps/example.txt`.
        """
        map_data = None
        with open(input_file, "r", encoding="utf-8") as file:
            full_input = file.readlines()
            self.lines, self.columns = self.read_int_tuple(full_input[0])

            taxi_initial_position = self.read_int_tuple(full_input[1])

            passenger_initial_position = self.read_int_tuple(full_input[2])

            drop_off_point = self.read_int_tuple(full_input[3])

            input_map = self.read_map_matrix(full_input)

            map_data = MapData(
                self.lines,
                self.columns,
                taxi_initial_position,
                passenger_initial_position,
                drop_off_point,
                input_map,
            )
        return map_data

    def read_int_tuple(self, line: str) -> tuple[int]:
        """ Reads a line and returns a tuple of integers. """
        values = line.split(" ")
        first_value = int(values[0])
        second_value = int(values[1])
        return (first_value, second_value)

    def read_map_matrix(self, buffer: list[str]) -> np.array:
        """ Reads the map matrix from the buffer. """
        output_map = np.zeros((self.lines, self.columns))
        for line in range(self.lines):
            current_line = buffer[self.MATRIX_START_INDEX + line].split(" ")
            if len(current_line) != self.columns:
                raise ValueError("Wrong number of columns in given matrix")

            for column in range(self.columns):
                output_map[line][column] = int(current_line[column])

        return output_map
