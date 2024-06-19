# Taxi driver 🚖

## Algorithm: A*

### About the algorithm

The A* search algorithm is a popular and versatile pathfinding and graph traversal algorithm. It is widely used in many fields, such as AI for game development, to find the most efficient path between nodes (points) in a graph.

This algorithm stands out for its efficiency in finding the shortest path from a starting point to a goal. Its strength lies in its ability to offer a balanced approach that prioritizes paths leading directly towards the goal while minimizing the total cost. This is achieved through the use of heuristics, which guide the search, making A* especially powerful in applications like AI pathfinding and graph traversal problems. The algorithm's performance and accuracy heavily depend on the choice of heuristic: an admissible heuristic ensures it always finds the optimal path.

However, the effectiveness of A* also comes with its weaknesses. The primary limitation is its space complexity, as it must store all explored and unexplored nodes in memory, which can become a significant issue for large graphs. Additionally, the choice of heuristic greatly influences the algorithm's efficiency; a poorly chosen heuristic can lead to increased search times and reduced performance. 

The heuristic used in the A* search algorithm, specifically when considering an environment without diagonal movements, employs the Manhattan distance. This heuristic calculates the total number of steps required to move horizontally and vertically from any given cell to the destination cell, without considering diagonal paths. The Manhattan distance is an example of an admissible heuristic because it never overestimates the actual cost to reach the goal. It provides a lower bound on the cost, ensuring the A* algorithm remains both efficient and guarantees finding the shortest path within the constraints of the movement allowed on the grid.

Regarding the time and space complexity of the A* search algorithm in the context of grid-based pathfinding without diagonal movements, the complexities can be succinctly described using Big O notation. The time complexity is $\(O((ROW \times COL) \log (ROW \times COL))\)$, and the space complexity is $\(O(ROW \times COL)\)$. The time complexity arises from the need to potentially explore a significant portion of the grid and manage the priority queue operations for the open list, where each operation can take logarithmic time relative to the number of elements in the queue. The space complexity reflects the storage required for maintaining the open and closed lists, which scales with the size of the grid, accounting for every possible position a node can occupy.

In conclusion, while A* is highly versatile and capable, its application requires careful consideration of the heuristic function and the computational resources available, making it less suitable for problems with extremely large search spaces or limited memory constraints.

### About the implementation

The algorithm proceeds as follows:

1. **Validation and Initialization**: It starts by validating the source and destination to ensure they're within the grid and not blocked. It also checks for the case that the taxi is initialized with the passenger. The closed list (for tracking visited cells) and cell details (storing information about each cell, like cost and parent cells) are initialized.

2. **Open List**: A priority queue, represented as a heap, is used to manage cells yet to be explored, prioritizing them based on their total cost (`total_cost`), which is the sum of the cost from the start to the current cell (`cost_from_start`) and the estimated cost from the current cell to the goal (`estimated_cost_to_goal`).

3. **Pathfinding Loop**: The algorithm iterates over the open list, extracting the cell with the lowest total cost and exploring its valid, unblocked neighbors in orthogonal directions. For each neighbor, it calculates the new costs and updates the cell details if a better path is found, or if the neighbor is the destination, it traces back the path from the destination to the source.

4. **Heuristic Calculation**: The Manhattan distance is used to estimate the cost from any cell to the destination (`estimated_cost_to_goal`), suitable for grid environments where diagonal movement is not allowed.

5. **Destination Check**: If a neighbor is the destination, the algorithm sets its parent and traces the path back to the source, returning this path.

6. **Termination**: The search ends when the destination is reached or when there are no more cells to explore. If the destination cannot be found, it raises an error.

This implementation is efficient for grid-based path planning, effectively finding the shortest path by balancing between the explored path's actual cost and the estimated cost to the goal, ensuring an optimal path is found with consideration for both explored and unexplored parts of the grid.

### Expected input format

```txt
lines columns # Map size (lines,columns)
0 0 # Taxi initial coordinate (line, column)
1 5 # Passenger initial coordinate (line, column)
7 7 # Drop-off coordinate (line, column)
# Map (1 for unblocked, 0 for blocked paths)
1 1 1 0
1 1 1 0
1 1 1 0
1 0 1 1
1 0 1 1
1 0 1 1
1 0 1 1
1 0 1 1
```

More examples are present in `maps` folder.

**OBS: Remove the # commented parts for the input to work**

### Setting up

Install the python dependencies:

```sh
pip install -r requirements.txt
```

### Running the code

To run the code, use the `main.py` file, it has the following usage:
```txt
usage: python main.py [-h] [-v] filename

Receives a input file from user and solves the taxi driver problem with A* algorithm

positional arguments:
  filename       Filename of the input

options:
  -h, --help     show this help message and exit
  -v, --verbose  Enable/disable verbose output.
```

Using the -v switch, you will get a interactive map on the terminal to see the taxi moving.

Example:

```sh
python main.py -v maps/map_1.txt
```

### Running tests

Run from the root folder:
```sh
pytest tests/
```

### References

GeeksforGeeks. (2024). [A* Search Algorithm](https://www.geeksforgeeks.org/a-search-algorithm/). 

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/2z7X09GL)
