import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.hyperparameters_data import Hyperparameters
from src.parameters_data import Parameters

def plot_learning_curve(
    hyperparameters: Hyperparameters,
    parameters: Parameters,
    algorithm_choice: str,
    ax: plt.Axes,
) -> None:

    df = pd.read_feather(f"{parameters.data_dir}/{parameters.filename}.feather")
    df_mov_avg = df.rolling(window=200).mean()

    plot = sns.lineplot(x="episode", y="rewards", data=df_mov_avg, ax=ax)
    plot.set_title(
        f"{algorithm_choice} - alpha: {hyperparameters.alpha} - gamma: {hyperparameters.gamma} - epsilon: {hyperparameters.epsilon_start} \n epsilon_min: {hyperparameters.epsilon_min} - epsilon_dec: {hyperparameters.epsilon_dec}"
    )
    plot.set_xlabel("Episode")
    plot.set_ylabel("Moving Average Rewards")
