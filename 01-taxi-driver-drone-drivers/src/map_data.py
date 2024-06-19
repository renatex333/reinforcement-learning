""" This module contains the MapData class, which is a dataclass that stores the map data. """

import dataclasses
import numpy as np

@dataclasses.dataclass
class MapData:
    """ Class representing the map data. """
    # Map
    lines: int
    columns: int
    # Points
    taxi_initial_position: tuple[int]
    passenger_initial_position: tuple[int]
    drop_off_point: tuple[int]
    # Map matrix
    map_matrix: np.array

    def print_data(self):
        """ Print the map data. """
        print(f"Taxi initial position: {self.taxi_initial_position}")
        print(f"Passenger initial position: {self.passenger_initial_position}")
        print(f"Drop off point: {self.drop_off_point}")
        print(f"Lines: {self.lines}")
        print(f"Columns: {self.columns}")
        print()

    def print_map(self):
        """ Print the map. """
        print("Map:")
        for line in self.map_matrix:
            print("".join(str(line)))
        print()
