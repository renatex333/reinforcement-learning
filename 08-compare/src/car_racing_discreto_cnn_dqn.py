
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

MODEL_NAME = 'car_racing_discreto_cnn_dqn'
ENV_NAME = 'CarRacing-v2'
tmp_path = f"../results/{MODEL_NAME}/"


def train():
    print(f'Training model {MODEL_NAME}')
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    env_kwargs = {
        "continuous": False,
    }

    vec_env = make_vec_env(ENV_NAME,
                           n_envs=1,
                           env_kwargs=env_kwargs,
                           )

    # docs de toda documentação em https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    #
    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=1_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=0.9,
        exploration_final_eps=0.02,
        tau=1,
        tensorboard_log=None
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=600_000)
    model.save(f"../models/{MODEL_NAME}")

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    del model
    print(f'Model {MODEL_NAME} trained.')


def test():
    print(f'Loading model {MODEL_NAME}')
    model = DQN.load(f"../models/{MODEL_NAME}")

    env = gym.make(
        ENV_NAME,
        render_mode="human",
        continuous=False
    )

    print('Testing model')
    (obs, _) = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        env.render()
    env.close()
    print('Test finished')


if __name__ == '__main__':
    train()
    test()
