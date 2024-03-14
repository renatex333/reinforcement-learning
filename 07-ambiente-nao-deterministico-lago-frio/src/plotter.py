import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.hyperparameters_data import Hyperparameters

def plotter(
        filename: str = "filename",
        data_dir: str = "data",
        results_dir: str = "results",
        algorithm: str = "q-learning",
        hyperparameters: Hyperparameters = None
    ):
    df = pd.read_feather(f"{data_dir}/{filename}.feather")
    df.index.name = "episode"
    df_mov_avg = df.rolling(window=200).mean()

    plt.figure()
    plot = sns.lineplot(x="episode", y="rewards", data=df_mov_avg)
    plot.set_title(f"{algorithm} - alpha: {hyperparameters.alpha} - gamma: {hyperparameters.gamma} - epsilon: {hyperparameters.epsilon}")
    plt.savefig(f"{results_dir}/{filename}.png")
    plt.close()
