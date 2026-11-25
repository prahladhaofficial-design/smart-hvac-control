"""Train the PPO agent on summer 2016 weather (June-September). Testing uses summer 2017 (never seen).

Run:  python src/train_ppo.py            (about 25-40 minutes on a laptop CPU; a trained model is already in models/)
"""
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from building_model import Building, build_schedule, load_weather
from hvac_env import HVACEnv

ROOT = Path(__file__).resolve().parent.parent
TOTAL_STEPS = 1_500_000
SEED = 0


def main():
    b = Building()
    sched = build_schedule(load_weather(year=2016), b)
    env = make_vec_env(lambda: HVACEnv(b, sched, train=True), n_envs=8, seed=SEED)
    # rewards are large and uneven (comfort violations dominate early on) -> rescale them for stable learning.
    # Observations are already hand-normalised in HVACEnv.observe, so only rewards are normalised here.
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.995)
    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=1024, n_epochs=10,
        gamma=0.995,           # 15-min steps -> looks ~2 days ahead, enough to plan pre-cooling
        gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=SEED, verbose=1,
    )
    model.learn(total_timesteps=TOTAL_STEPS)
    (ROOT / "models").mkdir(exist_ok=True)
    model.save(ROOT / "models" / "ppo_hvac")
    print("saved models/ppo_hvac.zip")


if __name__ == "__main__":
    main()
