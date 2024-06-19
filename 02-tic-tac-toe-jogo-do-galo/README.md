# Tic Tac Toe

## Algorithm: Minimax

### About the algorithm

The Minimax algorithm is a decision-making tool used in artificial intelligence (AI), game theory, and computational economics, to find the optimal move for a player, assuming that the opponent also plays optimally. At its core, Minimax is applied in two-player, turn-based games, such as Tic Tac Toe, Chess, and Go, with zero-sum outcomes—meaning one player's gain is equivalent to the other's loss. The algorithm simulates all possible moves in the game tree, evaluates the outcomes, and chooses the move that maximizes the player's minimum gain or minimizes the maximum loss, hence the name "Minimax".

## Strengths

1. **Optimal Strategy**: It guarantees an optimal playing strategy if both players make the best moves available to them.
2. **Simplicity**: The algorithm is conceptually simple and can be implemented for any two-player game with a finite set of moves.
3. **Predictability**: By evaluating all possible outcomes, it provides a comprehensive strategy that accounts for every contingency.
4. **Adaptability**: While primarily used in games, its principles can be adapted for use in various decision-making processes across different fields.

## Weaknesses

1. **Computational Expense**: For games with a vast number of possible moves and game states, the algorithm can be computationally expensive and impractical.
2. **Memory Usage**: Storing the game tree for complex games can consume a significant amount of memory, making it difficult to use on memory-constrained systems.
3. **Predictability**: While predictability is a strength, it can also be a weakness, as it means the AI's moves can be anticipated if the opponent knows the algorithm is in use.
4. **Lacks Learning Ability**: Minimax does not learn from past games; it evaluates the game tree from scratch every time, which can be inefficient compared to algorithms that utilize machine learning.

## Time and Space Complexity

The time complexity of the Minimax algorithm is O(b^d), where `b` is the branching factor, or the average number of moves available from any given position, and `d` is the depth of the tree, or the number of moves ahead that the algorithm evaluates. The space complexity is O(bd), as it needs to store a stack of game states proportional to the depth of the recursion. However, these complexities can be mitigated by optimizations such as Alpha-Beta pruning, which reduces the number of nodes evaluated.

## Conclusion

The Minimax algorithm is a cornerstone of game theory and AI for two-player games, offering a straightforward yet powerful framework for making optimal decisions. Despite its computational and memory limitations, it remains a popular choice due to its predictability and adaptability. For simpler games or those with a manageable number of outcomes, Minimax provides a reliable and optimal strategy. However, for more complex games, enhancements or alternative algorithms might be necessary to manage computational resources effectively.

### Setting up

Install the python dependencies:

```sh
pip install -r requirements.txt
```

### Running the code

To run the code, use the `main.py` file, it has the following usage:
```txt
usage: python3 main.py [-r] [-n]

Receives no positional arguments.

options:
  -r, --render    Render the game interface (True or False)
  -n, --episodes  Number of episodes to play (Integer)
```

Example:

```sh
python main.py --render True --episodes 3
```

### References


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/RTuXpCvk)

The Farama Foundation. (2024). [PettingZoo Documentation](https://pettingzoo.farama.org/environments/classic/tictactoe/)

GeeksforGeeks. (2022). [Minimax Algorithm in Game Theory](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/)