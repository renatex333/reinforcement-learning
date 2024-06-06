from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

ENV_NAME = "CarRacing-v2"
TEMP_PATH = "../results/car_racing_continuous_cnn_ppo_env_1/"
MODEL_PATH = "../models/car_racing_continuous_cnn_ppo_env_1"

def train():
    new_logger = configure(TEMP_PATH, ["stdout", "csv", "tensorboard"])
    env_kwargs = {
        "continuous": True,
    }
    vec_env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs=env_kwargs)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        clip_range=0.2,
        ent_coef=0.0,
        gae_lambda=0.95,
        n_steps=512,
        n_epochs=10,
        vf_coef=0.5,
        learning_rate=2e-4,
        batch_size=128,
        gamma=0.99,
        tensorboard_log=None,
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=1_000_000)
    model.save(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

def test():
    model = PPO.load(MODEL_PATH)

    print("modelo treinado")
    env = gym.make(ENV_NAME, render_mode="human", continuous=True)
    (obs, _) = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        env.render()

if __name__ == "__main__":
    should_train = input("Train? (y/n): ")
    if should_train == "y":
        train()
    test()
