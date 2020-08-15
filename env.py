import os
import sys
import optparse
import traci
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


class ENV():
    def __init__(self):
        options = get_options()
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", "sumomodel/exp.sumocfg"])

        self.actionStepLength = 0.2
        self.speed = 20
        # 车辆进入观测的范围  --ignore-route-errors
        self.Observe_Position_x = 16 + 30
        self.Observe_Position_y = 16 + 30
        # 记录当前交叉口车辆状态
        self.curr_vehicle_dict = {}
        # 记录当前交叉口车辆停车次数
        self.curr_vehicle_stop_num_dict = {}
        # 记录当前交叉口车辆所观测行人
        self.curr_vehicle_pedestrian_dict = {}
        # 记录当前交叉口撞车车辆
        self.curr_vehicle_broken=[]

    def update(self):
        veh_id_list = traci.vehicle.getIDList()
        new_vehicle_dict = {}
        new_vehicle_stop_num_dict = {}
        new_vehicle_pedestrian_dict = {}
        for veh_id in veh_id_list:
            # Observe_Position_x和Observe_Position_y范围内车辆进行观测
            if (traci.vehicle.getPosition(veh_id)[0] <= self.Observe_Position_x) \
                    and (traci.vehicle.getPosition(veh_id)[1] <= self.Observe_Position_y):
                # 更新vehicle_dict和vehicle_pedestrian_dict
                new_vehicle_dict[veh_id], new_vehicle_pedestrian_dict[veh_id] = self._get_state(veh_id)

                # 更新vehicle_stop_num_dict
                if traci.vehicle.getSpeed(veh_id) == 0:
                    if veh_id in self.curr_vehicle_stop_num_dict:
                        new_vehicle_stop_num_dict[veh_id] = self.curr_vehicle_stop_num_dict[veh_id] + 1
                    else:
                        new_vehicle_stop_num_dict[veh_id] = 1

        self.curr_vehicle_pedestrian_dict = new_vehicle_pedestrian_dict
        self.curr_vehicle_stop_num_dict = new_vehicle_stop_num_dict
        self.curr_vehicle_dict = new_vehicle_dict

    def _get_state(self, veh_id):
        # 最大观测行人数MAX_PERSON_COUNT
        MAX_PERSON_COUNT = 5
        START_OB_PERSON_POSITION_X = 20.0
        END_OB_PERSON_POSITION_X = 0.0
        states = np.zeros(6 + MAX_PERSON_COUNT * 3)
        # 车的位置
        states[0] = traci.vehicle.getPosition(veh_id)[0]
        states[1] = traci.vehicle.getPosition(veh_id)[1]
        # 车的速度
        states[2] = traci.vehicle.getSpeed(veh_id)
        # 车的长度和宽度
        states[3] = traci.vehicle.getLength(veh_id)
        states[4] = traci.vehicle.getWidth(veh_id)
        # 路口交通灯
        if traci.trafficlight.getRedYellowGreenState(traci.trafficlight.getIDList()[0])[6] == 'r':
            states[5] = 1
        # 行人
        person_count = 0
        person_id_list = []
        for person_id in traci.person.getIDList():
            if (person_count <= MAX_PERSON_COUNT - 1) and \
                    (traci.person.getPosition(person_id)[0] < START_OB_PERSON_POSITION_X) and \
                    (traci.person.getPosition(person_id)[0] > END_OB_PERSON_POSITION_X):
                person_id_list.append(person_id)
                # 行人的位置
                states[6 + person_count * 3] = traci.person.getPosition(person_id)[0]
                states[7 + person_count * 3] = traci.person.getPosition(person_id)[1]
                # 行人的速度
                states[8 + person_count * 3] = traci.person.getSpeed(person_id)
                person_count += 1
            if person_count > MAX_PERSON_COUNT:
                break
        return states, person_id_list

    def set_Observe_Position_x(self, x):
        self.Observe_Position_x = x

    def get_state(self, veh_id):
        return self.curr_vehicle_dict[veh_id]

    def step(self, action_list=None):
        veh_id_list, state_list, reward_list, state_next_list, terminal_list = [], [], [], [], []

        v0, v1 = 0, 0
        # 观测车辆当前状态结束的情况
        for veh_id in self.curr_vehicle_dict.keys():
            veh_id_list.append(veh_id)
            state_list.append(self.curr_vehicle_dict[veh_id])
            reward_list.append(None)
            state_next_list.append(None)
            terminal_list.append(True)

            traci.vehicle.setType(veh_id, 'vtype3')
            traci.vehicle.setSpeedMode(veh_id, sm=0)
            traci.vehicle.setSpeed(veh_id, self.speed)
            self.speed = np.random.randint(2, 5, 1)
            v0 = traci.vehicle.getSpeed(veh_id)

        # 对观测范围内的车辆添加动作
        if not action_list is None:
            for veh_id, action in zip(veh_id_list, action_list):
                self.set_action(veh_id, action)

        # 运行一步
        traci.simulationStep()
        self.update()

        # 判断是否相撞
        for veh_id in self.curr_vehicle_dict.keys():
            if self.is_brake(veh_id):
                self.del_vehicle(veh_id)
                self.curr_vehicle_broken.append(veh_id)
        self.update()

        for veh_id in self.curr_vehicle_dict.keys():

            # 观测车辆在运行过程中
            if veh_id in veh_id_list:
                reward_list[veh_id_list.index(veh_id)] = self.get_total_reward(veh_id)
                state_next_list[veh_id_list.index(veh_id)] = self.curr_vehicle_dict[veh_id]
                terminal_list[veh_id_list.index(veh_id)] = False
            # 新出现观测车辆
            else:
                veh_id_list.append(veh_id)
                state_list.append(self.curr_vehicle_dict[veh_id])

                reward_list.append(None)
                state_next_list.append(None)
                terminal_list.append(False)
            v1 = traci.vehicle.getSpeed(veh_id)

        # print("v0:", v0, "v1:", v1, " a:", (v1 - v0) / 0.2)

        return veh_id_list, state_list, reward_list, state_next_list, terminal_list

    def set_action(self, veh_id, action):
        if not action == None:
            traci.vehicle.setSpeed(veh_id, traci.vehicle.getSpeed(veh_id) + action)

    def get_total_reward(self, veh_id):
        # print('speed_reward ', self.speed_reward(veh_id),
        #       ' pedestrian_reward ', self.pedestrian_reward(veh_id),
        #       ' stop_num_reward ', self.stop_num_reward(veh_id))
        return self.speed_reward(veh_id) + self.pedestrian_reward(veh_id) + self.stop_num_reward(veh_id)

    def speed_reward(self, veh_id):
        # 路口最大速度
        MAX_SPEED = 40.0
        speed = traci.vehicle.getSpeed(veh_id)
        if speed <= MAX_SPEED:
            speed_reward = speed / MAX_SPEED
        else:
            speed_reward = -1
        return speed_reward

    def distance_vehicle_pedestrians(self, veh_id, person_id):
        position_veh = np.array(traci.vehicle.getPosition(veh_id))
        position_person = np.array(traci.person.getPosition(person_id))
        return np.linalg.norm(position_veh - position_person)

    def is_brake(self, veh_id):
        CROSS_RANGE_LEFT = 7.0  # 7.5
        CROSS_RANGE_RIGHT = 11.2  # 11.25
        CROSS_RANGE_UP = 16.0
        CROSS_RANGE_DOWN = 11.25

        position_veh = traci.vehicle.getPosition(veh_id)[1]
        action_step_length = traci.vehicle.getActionStepLength(veh_id)
        v = traci.vehicle.getSpeed(veh_id)
        length = traci.vehicle.getLength(veh_id)
        # step执行后，车头超过斑马线
        if position_veh > CROSS_RANGE_DOWN:
            # step执行前，车尾在斑马线
            if (position_veh - action_step_length * v - length) < CROSS_RANGE_UP:
                #是否有行人在人行线内
                for person_id in self.curr_vehicle_pedestrian_dict[veh_id]:
                    position_person = traci.person.getPosition(person_id)[0]
                    if (position_person >= CROSS_RANGE_LEFT) and (position_person <= CROSS_RANGE_RIGHT):

                        return True

        return False

    def del_vehicle(self,veh_id):
        traci.vehicle.moveToXY(veh_id, 'gneE_0', 2, x=0, y=100)

    def pedestrian_reward(self, veh_id):
        # 车与行人的最小安全距离
        # MAX_SAFE_DISTANCE_VEH_PERSON = 1.5
        # if len(self.curr_vehicle_pedestrian_dict[veh_id]) == 0:
        #     return 0
        # for person_id in self.curr_vehicle_pedestrian_dict[veh_id]:
        #     if self.distance_vehicle_pedestrians(veh_id, person_id) < MAX_SAFE_DISTANCE_VEH_PERSON:
        #         return -1
        # return 0

        if veh_id in self.curr_vehicle_broken:
            return -10
        else:
            return 0

    def stop_num_reward(self, veh_id):
        MAX_STOP_NUM = 3.0
        stop_num_reward = 0
        if len(self.curr_vehicle_stop_num_dict) > 0:
            if veh_id in self.curr_vehicle_stop_num_dict:
                if self.curr_vehicle_stop_num_dict[veh_id] <= MAX_STOP_NUM:
                    stop_num_reward = -self.curr_vehicle_stop_num_dict[veh_id] / MAX_STOP_NUM
                else:
                    stop_num_reward = -1

        return stop_num_reward
