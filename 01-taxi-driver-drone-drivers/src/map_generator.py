""" This module contains the code for map generation. """

import os
import random
import numpy as np

def is_valid_block_position(map_grid, pos, height, width):
    """
    Check if a new block can be placed at the position considering: 
    - The adjacency rule;
    - The position does not segregate the map into two disconnected parts.
    - The position is not already blocked;
    """

    # Adjacency Rule: A block can only be placed if it has less than 2 adjacent blocks
    adjacent_blocks = 0
    # Check all eight surrounding cells
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Skip the center cell itself
            if dy == 0 and dx == 0:
                continue
            ny, nx = pos[0] + dy, pos[1] + dx
            # Check boundaries and then check for block presence
            if 0 <= ny < height and 0 <= nx < width and map_grid[ny, nx] == 0:
                adjacent_blocks += 1

    # Segregation Rule: A block can only be placed if
    # it does not segregate the map into two disconnected parts
    free_cells_in_row = 0
    free_cells_in_col = 0

    # Check the entire row
    for col in range(width):
        if map_grid[pos[0], col] == 1:
            free_cells_in_row += 1

    # Check the entire column
    for row in range(height):
        if map_grid[row, pos[1]] == 1:
            free_cells_in_col += 1

    # If the amount of free cells in the row or column is 1, then the map will be segregated
    is_segregatory = free_cells_in_row == 1 or free_cells_in_col == 1

    # Check if the position is free to place a block
    is_free = map_grid[pos[0], pos[1]] == 1

    return adjacent_blocks < 2 and (not is_segregatory) and is_free

def generate_random_map(folder_name, min_size=5, max_size=10, file_name="random_map.txt"):
    """ Generates and writes map to file. """
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Randomly select map size within given limits
    map_height = random.randint(min_size, max_size)
    map_width = random.randint(min_size, max_size)

    # Randomly place taxi, passenger, and drop-off point
    taxi_coord = (random.randint(0, map_height-1), random.randint(0, map_width-1))
    passenger_coord = taxi_coord
    dropoff_coord = taxi_coord

    # Ensure all three points are distinct
    while passenger_coord == taxi_coord:
        passenger_coord = (random.randint(0, map_height-1), random.randint(0, map_width-1))
    while dropoff_coord in (taxi_coord, passenger_coord):
        dropoff_coord = (random.randint(0, map_height-1), random.randint(0, map_width-1))

    # Initialize map with all free tiles (1s)
    map_grid = np.ones((map_height, map_width), dtype=int)
    # Randomly generate obstacles on the map
    num_obstacles = int(0.2 * map_height * map_width)  # 20% of the map as obstacles
    num_iterations = 0
    while num_obstacles > 0 and num_iterations < 1000:
        obstacle_coord = (random.randint(1, map_height-2), random.randint(1, map_width-2))
        if (
                obstacle_coord not in [taxi_coord, passenger_coord, dropoff_coord]
                and is_valid_block_position(map_grid, obstacle_coord, map_height, map_width)
        ):
            map_grid[obstacle_coord] = 0
            num_obstacles -= 1
        num_iterations += 1

    # Write map details to the file
    with open(os.path.join(folder_name, file_name), 'w', encoding="utf-8") as file:
        file.write(f"{map_height} {map_width}\n")
        file.write(f"{taxi_coord[0]} {taxi_coord[1]}\n")
        file.write(f"{passenger_coord[0]} {passenger_coord[1]}\n")
        file.write(f"{dropoff_coord[0]} {dropoff_coord[1]}\n")
        for row in map_grid:
            file.write(' '.join(map(str, row)) + '\n')
