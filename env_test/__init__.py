from gym.envs.registration import register

register(
    id='sumo_env-v0',
    entry_point='env_test.env_singlecar_gym:SUMO_ENV',
    max_episode_steps=100,
    reward_threshold=10.0,
)