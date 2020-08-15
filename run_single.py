from env_singe_car import *
import time
import numpy as np
if __name__ == '__main__':
    env = ENV()
    env.update()

    # 864000 = 3600 * 5 * 24*2 2days
    episodes={}
    episode=[]
    for i in range(864000):
        action = np.random.randint(-5, 5, 1)
        veh_id, state, reward, state_next, terminal=env.step(action)
        if veh_id is not None:
            if veh_id not in episodes.keys():
                episodes[veh_id]=[]
            episodes[veh_id].append([veh_id, state, reward, state_next, terminal])

        time.sleep(0.02)
