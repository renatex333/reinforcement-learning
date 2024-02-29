# Taxi Driver Problem - Q-Learning

The Taxi Problem is a problem described in the "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" paper by Tom Dietterich. The problem is as follows:
    
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    
    There are six primitive actions in this environment:

        - south
        - north
        - east
        - west
        - pickup
        - dropoff

    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.

The environment is a 5x5 grid, with four locations marked as R(ed), G(reen), Y(ellow), and B(lue). The taxi operates in this environment, and the passenger is picked up and dropped off at one of the four locations. The taxi has to learn to move to the passenger's location, pick up the passenger, move to the destination, and drop off the passenger.

Using a Q-Learning algorithm, the agent learns to act in the environment and is then evaluated.

The `q_learning.py` module implements the Q-Learning algorithm for the Taxi Driver problem, while the `tax_driver.py` script is responsible for running the training and simulation.

## Basic Usage

```txt
usage: python taxi_driver.py [--filename FILENAME] [--data-dir DATA_DIR] [--results-dir RESULTS_DIR] [--train]
                      [--episodes EPISODES] [--alpha ALPHA] [--gamma GAMMA] [--epsilon EPSILON]
                      [--epsilon-min EPSILON_MIN] [--epsilon-dec EPSILON_DEC] [--random] [--only-exploit]

This program implements a Q-learning model for solving the taxi driver problem. It can operate in both training and inference modes, allowing users to specify various parameters to customize the learning process and output results.

positional arguments:
  none

options:
  --filename FILENAME   Identifier to data and results filenames. Default is "taxi-driver".
  --data-dir DATA_DIR   Directory to store the training data. Default is "data". Will be created if not exists.
  --results-dir RESULTS_DIR
                        Directory to store the resulting images (graphs). Default is "results". Will be created if not exists.
  --train               Flag to enable model training. If not set, the model will perform inference using the Q-table.
  --episodes EPISODES   Number of episodes to run during training. Default is 50000.
  --alpha ALPHA         Learning rate for the Q-learning algorithm. Default is 0.1.
  --gamma GAMMA         Discount factor for the Q-learning algorithm. Default is 0.99.
  --epsilon EPSILON     Exploration rate for the Q-learning algorithm. Default is 0.7.
  --epsilon-min EPSILON_MIN
                        Minimum exploration rate. Default is 0.05.
  --epsilon-dec EPSILON_DEC
                        Exploration decay rate. Default is 0.99.
  --random              Randomize all agent actions. Useful for exploring the action space.
  --only-exploit        Only exploit the Q-table for making decisions, without further training.
```

Use these options to configure the behavior of the taxi driver AI model, including training parameters and file paths for input and output data.

## Usage Example

    python3 taxi_driver.py --filename taxi-driver-teste --train

# References

[Assignment](https://insper.github.io/rl/classes/05_x_hyperparameters/)