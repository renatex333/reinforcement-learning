from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


tmp_path = "../results/car_racing_discreto_mlp_a2c/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CarRacing-v2")
vec_env = make_vec_env("CarRacing-v2", n_envs=8)

# docs de toda documentação em https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
model = A2C(
    policy="MlpPolicy",
    env=vec_env,
)

model.set_logger(new_logger)
model.learn(total_timesteps=600_000)
model.save("../models/car_racing_discreto_mlp_a2c")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

del model
model = A2C.load("../models/car_racing_discreto_mlp_a2c")

print("Modelo treinado!")

env = gym.make("CarRacing-v2", render_mode="human")

(obs, _) = env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
