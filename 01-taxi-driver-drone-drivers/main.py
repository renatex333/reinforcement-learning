""" Main module to run the taxi driver problem with A* algorithm. """

import argparse
from os import name, system
from src.a_star import a_star_search
from src.input_parser import InputParser

directions_map = {
    (0, 1): "move right",
    (0, -1): "move left",
    (1, 0): "move down",
    (-1, 0): "move up"
}

def clear_terminal():
    """ Clear the terminal screen. """
    # for windows
    if name == "nt":
        _ = system("cls")
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


def render_taxi_board(taxi_map: list, taxi_position: tuple, objective_position: tuple):
    """
    Render the taxi driver board from the map to the console.

    Args:
    - taxi_map (list of lists): The map representing the taxi board.
    - taxi_position (tuple): The position of the taxi driver on the board (row, column).

    Returns:
    - None
    """
    for i, row in enumerate(taxi_map):
        print("| ", end="")
        for j, cell in enumerate(row):
            if (i, j) == taxi_position:
                print("🚖", end="|")  # Taxi driver's position
            elif (i, j) == objective_position:
                print("🎯", end="|")
            else:
                if cell == 0:
                    print("🧱", end="|")
                else:
                    print(" ", end=" |")
        print()  # Move to the next line


def parse_arguments():
    """ Parse the command line arguments. """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="""Receives a input file
                       from user and solves the
                       taxi driver problem with A* algorithm""",
    )
    parser.add_argument("filename", type=str, help="Filename of the input")
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Enable/disable verbose output.",
    )

    return parser.parse_args()

def print_movements(algorithm_path: list):
    """ Print the movements based on the algorithm path. """
    previous = algorithm_path[0]
    for idx in range(1, len(algorithm_path)):
        prev_x, prev_y = previous
        curr_x, curr_y = algorithm_path[idx]
        diff = (curr_x - prev_x, curr_y - prev_y)
        movement_name = directions_map[diff]
        previous = algorithm_path[idx]
        print(movement_name)

def solve_map(filename: str, verbose: bool):
    """ Solve the taxi driver problem with A* algorithm. """
    parser = InputParser()
    map_data = parser.parse_from_input(filename)

    # Get the maximum row and column (grid size)
    max_row = map_data.lines
    max_col = map_data.columns

    # Run the A* search algorithm from taxi to passenger
    passenger_pickup_path = a_star_search(
        map_matrix=map_data.map_matrix,
        source=map_data.taxi_initial_position,
        destination=map_data.passenger_initial_position,
        max_row=max_row,
        max_col=max_col
    )
    # Pickup the passenger
    # Run the A* search algorithm from passenger to dropoff
    drop_off_path = a_star_search(
        map_matrix=map_data.map_matrix,
        source=map_data.passenger_initial_position,
        destination=map_data.drop_off_point,
        max_row=max_row,
        max_col=max_col
    )
    if verbose:
        print("passenger pick up path")
        for position in passenger_pickup_path:
            render_taxi_board(map_data.map_matrix, position, map_data.passenger_initial_position)
            prompt = input("Next time step (q to quit)")
            if prompt == "q":
                return None
            clear_terminal()
        print("Drop off path")
        for position in drop_off_path:
            render_taxi_board(map_data.map_matrix, position, map_data.drop_off_point)
            prompt = input("Press enter to next time step (q to quit)")
            if prompt == "q":
                return None
            clear_terminal()
    else:
        print_movements(passenger_pickup_path)
        print("pick up")
        print_movements(drop_off_path)
        print("drop off")

    return passenger_pickup_path, drop_off_path

def main():
    """ Main execution function. """
    args = parse_arguments()
    solve_map(args.filename, args.verbose)

if __name__ == "__main__":
    main()
