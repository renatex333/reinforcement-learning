from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

TMP_PATH = "../results/car_racing_continous_ppo_mlp_1_env/"
MODEL_PATH = "../models/car_racing_continous_ppo_mlp_1_env"
ENV_NAME = "CarRacing-v2"

new_logger = configure(TMP_PATH, ["stdout", "csv", "tensorboard"])

env = gym.make(ENV_NAME)
env_kwargs = {
    "continuous": True,
}
vec_env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs=env_kwargs)


model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=None,
)

model.set_logger(new_logger)
model.learn(total_timesteps=1_000_000)
model.save(MODEL_PATH)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model

model = PPO.load(MODEL_PATH)
print('modelo treinado')
env = gym.make(ENV_NAME, render_mode='human')
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()