import gym
import env_test.env_singlecar_gym


if __name__ == '__main__':
    env = gym.make('SUMO_ENV-v0')
    env.reset()
    for _ in range(1000):
        veh_id, obs, reward, next_obs, done = env.step(env.action_space.sample())  # take a random action
        print(veh_id, obs, reward, next_obs, done)
