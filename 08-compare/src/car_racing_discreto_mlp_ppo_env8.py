"""
Module to train a PPO model with a MLP policy on the CarRacing-v1 environment with discrete actions.
"""

import sys
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

BASE_PATH = "car_racing_discreto_MLP_ppo_env8"
RESULTS_PATH = f"../results/{BASE_PATH}/"
MODEL_PATH = f"../models/{BASE_PATH}"
ENV_NAME = "CarRacing-v2"

def main():
    train_model = input("Train model? (Y/n): ")
    if train_model.lower() != "n":
        train()
    sys.exit(test())

def train():
    new_logger = configure(RESULTS_PATH, ["stdout", "csv", "tensorboard"])

    vec_env = make_vec_env(
        ENV_NAME,
        n_envs=8,
        env_kwargs={
            "render_mode": "rgb_array",
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": False
        },
        wrapper_class=my_wrapper,
        wrapper_kwargs={
            "shape": (64, 64),
            "keep_dim": True
        },
    )

    # Stable-baseline PPO usage docs:
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        batch_size=1024,
        n_steps=512,
        n_epochs=10,
        verbose=1,
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=600_000)
    model.save(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    return 0

def test():
    model = PPO.load(MODEL_PATH)

    print("Trained model. Testing...")
    env = gym.make(
        ENV_NAME,
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )

    wrapped_env = my_wrapper(env, shape=(64, 64), keep_dim=True)

    (obs,_) = wrapped_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = wrapped_env.step(action)
        wrapped_env.render()
        if done:
            obs = wrapped_env.reset()
    wrapped_env.close()
    return 0

def my_wrapper(env, shape, keep_dim):
    resized_env = ResizeObservation(env, shape=shape)
    gray_env = GrayScaleObservation(resized_env, keep_dim=keep_dim)
    return gray_env

if __name__ == "__main__":
    main()
