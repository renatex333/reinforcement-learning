
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from collections import OrderedDict

tmp_path = "../results/bipedalwalker_ppo_hugging_face/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("BipedalWalker-v3")
vec_env = make_vec_env("BipedalWalker-v3", n_envs=32)
             
# valores de hiper-parâmetros obtidos em https://huggingface.co/sb3
model = PPO(
    policy="MlpPolicy",
    clip_range=0.18,
    env=vec_env,
    batch_size=64,
    learning_rate=0.0003,
    n_steps=2048,
    gamma=0.999,
    gae_lambda=0.95,
    ent_coef=0.0,
    n_epochs=10,
    tensorboard_log=None)

model.set_logger(new_logger)
model.learn(total_timesteps=5_000_000)
model.save("../models/bipedalwalker_ppo_hugging_face")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = PPO.load("../models/bipedalwalker_ppo_hugging_face")

print('modelo treinado')
env = gym.make("BipedalWalker-v3", render_mode='human')
vec_env = make_vec_env("BipedalWalker-v3", n_envs=1, monitor_dir=None, wrapper_class=None, env_kwargs={'render_mode': 'human'})

obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.render()
    if dones.any():
        obs = vec_env.reset()