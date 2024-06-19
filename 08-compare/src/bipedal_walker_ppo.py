
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/bipedalwalker_ppo/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("BipedalWalker-v3")
vec_env = make_vec_env("BipedalWalker-v3", n_envs=1)

# Documentação completa em https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    tensorboard_log=None
)

model.set_logger(new_logger)
model.learn(total_timesteps=5_000_000)
model.save("../models/bipedalwalker_ppo")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = PPO.load("../models/bipedalwalker_ppo")

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