import gym
import env_test.env_singlecar_gym


if __name__ == '__main__':
    env = gym.make('sumo_env-v0')
    env.reset()
    for _ in range(1000):
        reward, next_obs, done,info = env.step(env.action_space.sample())  # take a random action
        print(reward, next_obs, done,info)
