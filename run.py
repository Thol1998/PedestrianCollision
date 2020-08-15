from env import *
import time
if __name__ == '__main__':
    env = ENV()
    action=2
    env.update()
    vehicle_id_list = env.curr_vehicle_dict.keys()
    # 864000 = 3600 * 5 * 24*2 2days
    episodes={}
    for i in range(864000):
        vehs_id, states, rewards, states_next, terminals=env.step()
        for veh_id in vehs_id:
            index = vehs_id.index(veh_id)
            if veh_id in episodes.keys():
                episodes[veh_id]+=[states[index], rewards[index], states_next[index], terminals[index]]
            else:
                episodes[veh_id]=[states[index], rewards[index], states_next[index], terminals[index]]

        time.sleep(0.02)
