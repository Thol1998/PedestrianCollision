import os
import sys
import optparse
import traci
import numpy as np
import re
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

        self.action_acc_up=5
        self.action_acc_down = -self.action_acc_up*3
        self.actionStepLength = 0.2
        self.speed_delta=0
        # self.max_step=100
        # 车辆进入观测的范围  --ignore-route-errors
        self.Observe_Position_x = 16 + 60
        self.Observe_Position_y = 16 + 30
        # 记录当前交叉口车辆状态
        self.curr_vehicle_id=None
        self.curr_vehicle_state = ''
        # 记录当前交叉口车辆停车次数
        self.curr_vehicle_stop_num = 0
        # 记录当前交叉口车辆所观测行人
        self.curr_vehicle_pedestrian = []
        # 记录当前交叉口撞车车辆
        self.curr_vehicle_broken = []

        #人行道判断相撞的范围
        self.CROSS_RANGE_LEFT = 8.2  # 7.5
        self.CROSS_RANGE_RIGHT = 10.5  # 11.25
        self.CROSS_RANGE_UP = 15.7
        self.CROSS_RANGE_DOWN = 11.7

        # 最大观测行人数MAX_PERSON_COUNT
        self.MAX_PERSON_COUNT = 2
        self.START_OB_PERSON_POSITION_X = 20.0
        self.END_OB_PERSON_POSITION_X = -15.0

        self.MAX_SPEED = 30.0

    def update(self):
        veh_id_list = traci.vehicle.getIDList()
        old_vehicle_id=self.curr_vehicle_id
        counter=0
        for veh_id in veh_id_list:
            # Observe_Position_x和Observe_Position_y范围内车辆进行观测
            if (traci.vehicle.getPosition(veh_id)[0] <= self.Observe_Position_x) \
                    and (traci.vehicle.getPosition(veh_id)[1] <= self.Observe_Position_y):
                #保持交叉口单辆车
                if counter == 1:
                    self.del_vehicle(veh_id)
                else:
                    # 更新vehicle_dict和vehicle_pedestrian_dict
                    self.curr_vehicle_id = veh_id
                    self.curr_vehicle_state, self.curr_vehicle_pedestrian = self._get_state(veh_id)
                    traci.vehicle.setSpeed(veh_id, 0)
                    for pedestrian_id in self.curr_vehicle_pedestrian:
                        traci.person.setSpeed(pedestrian_id,0.00000001)
                    counter += 1
            else:
                if (traci.vehicle.getPosition(veh_id)[1] > self.Observe_Position_y):
                    traci.vehicle.setSpeed(veh_id,30)
        if (counter==0) and (old_vehicle_id==self.curr_vehicle_id):
            self.curr_vehicle_id = None
        traci.simulationStep()
        if counter==1:
            traci.vehicle.setSpeed(self.curr_vehicle_id,self.curr_vehicle_state[2])
        for pedestrian_id in self.curr_vehicle_pedestrian:
            traci.person.setSpeed(pedestrian_id, 1.4)


    def _get_state(self, veh_id):
        states = np.zeros(6 + self.MAX_PERSON_COUNT * 4)
        # 车的位置
        states[0] = traci.vehicle.getPosition(veh_id)[0]
        states[1] = traci.vehicle.getPosition(veh_id)[1]
        # 车的速度
        states[2] = traci.vehicle.getSpeed(veh_id)
        # 车的长度和宽度
        states[3] = traci.vehicle.getLength(veh_id)
        states[4] = traci.vehicle.getWidth(veh_id)
        # states[3]=0
        # states[4]=0
        # 路口交通灯
        # if traci.trafficlight.getRedYellowGreenState(traci.trafficlight.getIDList()[0])[6] == 'r':
        #     states[5] = 1
        states[5] = 1
        # 行人

        person_list = []
        for person_id in traci.person.getIDList():
            if (traci.person.getPosition(person_id)[0] < self.START_OB_PERSON_POSITION_X) and \
                (traci.person.getPosition(person_id)[0] > self.END_OB_PERSON_POSITION_X):
                person_list.append([person_id,np.abs(traci.person.getPosition(person_id)[0]-9.5)])
                person_list.sort(key = lambda person: person[1])

        person_count = 0
        for person_id, _ in person_list[0:self.MAX_PERSON_COUNT]:
            # 行人的位置
            states[6 + person_count * 4] = traci.person.getPosition(person_id)[0]
            states[7 + person_count * 4] = traci.person.getPosition(person_id)[1]
            # states[7 + person_count * 4]=0
            # 行人的速度
            states[8 + person_count * 4] = traci.person.getSpeed(person_id)
            if int(re.findall('\d', person_id)[0]) == 1:
                states[9 + person_count * 4] = 1
            else:
                states[9 + person_count * 4] = 0
            person_count += 1

        return states, [person_list[i][0] for i in range(len(person_list))]

    def set_Observe_Position_x(self, x):
        self.Observe_Position_x = x

    def get_state(self):
        return self.curr_vehicle_state

    def reset(self):
        if self.curr_vehicle_id is not None:
            self.del_vehicle(self.curr_vehicle_id)
            self.update()
        while True:
            veh_id, state, reward, state_next, terminal=self._step()
            if veh_id is not None:
                self.curr_vehicle_stop_num = 0
                return veh_id, state, reward, state_next, terminal

    def step(self, action=None):
        # if action == 0:
        #     action = -5*3
        # elif action == 1:
        #     action = 5
        # else:
        #     action = 0
        while True:
            veh_id, state, reward, state_next, terminal=self._step(action)
            if veh_id is not None:
                return veh_id, state, reward, state_next, terminal

    def _step(self, action=None):

        if self.curr_vehicle_id is None:
            traci.simulationStep()
            self.update()
            return self.curr_vehicle_id,self.curr_vehicle_state,0,self.curr_vehicle_state,False

        # 观测车辆当前状态结束的情况
        veh_id=self.curr_vehicle_id
        state=self.curr_vehicle_state
        state_next=state
        terminal=True
        #放弃安全模式
        traci.vehicle.setType(veh_id, 'vtype3')
        traci.vehicle.setLaneChangeMode(veh_id, lcm=0)
        traci.vehicle.setSpeedMode(veh_id, sm=0)

        #测试
        # self.speed = np.random.randint(2, 20, 1)
        # traci.vehicle.setSpeed(veh_id, self.speed)
        # v0, v1 = 0, 0
        # v0 = traci.vehicle.getSpeed(veh_id)

        # 对观测范围内的车辆添加动作
        if not action is None:
            self.set_action(veh_id, action)

        # 运行一步
        traci.simulationStep()
        # if self.PedstrianCollision():
        if self.is_brake():
            self.del_vehicle(veh_id)
            self.curr_vehicle_broken.append(veh_id)
        # else:
        #     if (traci.vehicle.getPosition(veh_id)[0] > 20) and \
        #             (traci.vehicle.getSpeed(veh_id) == 0):
        #         self.del_vehicle(veh_id)
        #         self.curr_vehicle_broken.append(veh_id)

        # if self.curr_vehicle_stop_num>=self.max_step:
        #     self.del_vehicle(veh_id)

        self.update()
        if self.curr_vehicle_id is not None:
            if (traci.vehicle.getSpeed(self.curr_vehicle_id)==0) and(not self.speed_delta==0):
                self.curr_vehicle_stop_num = self.curr_vehicle_stop_num + 1
            state_next=self.curr_vehicle_state
            terminal=False

        reward = self.get_total_reward()

        # v1 = traci.vehicle.getSpeed(veh_id)
        # print("v0:", v0, "v1:", v1, " a:", (v1 - v0) / 0.2)

        return veh_id, state, reward, state_next, terminal

    def set_action(self, veh_id, action):
        if not action == None:
            old_speed=traci.vehicle.getSpeed(veh_id)
            speed=np.max([traci.vehicle.getSpeed(veh_id) + action*traci.vehicle.getActionStepLength(veh_id),0])
            traci.vehicle.setSpeed(veh_id, speed)
            self.speed_delta=speed-old_speed

    def action_space_sample(self):
        return np.random.uniform(self.action_acc_down,self.action_acc_up,1)

    def get_total_reward(self):
        # print('speed_reward ', self.speed_reward(veh_id),
        #       ' pedestrian_reward ', self.pedestrian_reward(veh_id),
        #       ' stop_num_reward ', self.stop_num_reward(veh_id))
        if self.curr_vehicle_id is None:
            return 0
        speed_reward=self.speed_reward()
        pedestrian_reward=self.pedestrian_reward()
        stop_num_reward=self.stop_num_reward()
        total_reward= speed_reward + pedestrian_reward + stop_num_reward
        if traci.vehicle.getSpeed(self.curr_vehicle_id) <= -35000000:
            print( traci.vehicle.getSpeed(self.curr_vehicle_id))
        return total_reward

        # return self.pedestrian_reward(veh_id)+1

    def speed_reward(self):
        # 路口最大速度
        speed = traci.vehicle.getSpeed(self.curr_vehicle_id)
        if speed <= self.MAX_SPEED:
            speed_reward = speed / self.MAX_SPEED
        else:
            speed_reward = -1

        return speed_reward

    def pedestrian_reward(self):
        # 车与行人的最小安全距离
        # MAX_SAFE_DISTANCE_VEH_PERSON = 1.5
        # if len(self.curr_vehicle_pedestrian_dict[veh_id]) == 0:
        #     return 0
        # for person_id in self.curr_vehicle_pedestrian_dict[veh_id]:
        #     if self.distance_vehicle_pedestrians(veh_id, person_id) < MAX_SAFE_DISTANCE_VEH_PERSON:
        #         return -1
        # return 0
        if self.curr_vehicle_id in self.curr_vehicle_broken:
            self.curr_vehicle_broken.pop(self.curr_vehicle_broken.index(self.curr_vehicle_id))
            return -100
        else:
            return 0

    def stop_num_reward(self):
        MAX_STOP_NUM = 3.0
        stop_num_reward = 0
        if traci.vehicle.getSpeed(self.curr_vehicle_id)>0:return 0
        if self.curr_vehicle_stop_num<= MAX_STOP_NUM:
            stop_num_reward = -self.curr_vehicle_stop_num / MAX_STOP_NUM
        else:
            stop_num_reward = -1

        return stop_num_reward

    def is_brake(self):
        Threshold=2
        curr_vehicle_state=self.curr_vehicle_state
        veh_pos=curr_vehicle_state[0:2]
        if len(self.curr_vehicle_pedestrian)<self.MAX_PERSON_COUNT:
            for i in range(len(self.curr_vehicle_pedestrian)):
                distance=np.linalg.norm(veh_pos - curr_vehicle_state[6+i*4:8+i*4])
                if distance<Threshold:
                    print(self.curr_vehicle_id,  'broken! ','distance:',distance)
                    return True
        else:
            for i in range(self.MAX_PERSON_COUNT):
                distance=np.linalg.norm(veh_pos - curr_vehicle_state[6+i*4:8+i*4])
                if distance<Threshold:
                    print(self.curr_vehicle_id, 'broken! ','distance:',distance)
                    return True
        return False

    def del_vehicle(self, veh_id):
        traci.vehicle.moveToXY(veh_id, 'gneE_0', 2, x=0, y=95)

    def distance_vehicle_pedestrians(self, veh_id, person_id):
        position_veh = np.array(traci.vehicle.getPosition(veh_id))
        position_person = np.array(traci.person.getPosition(person_id))
        return np.linalg.norm(position_veh - position_person)

    # def PedstrianCollision(self):
    #     PedstrianCollision=True
    #     pos_v = traci.vehicle.getPosition(self.curr_vehicle_id)[1]
    #     for person_id in traci.person.getIDList():
    #         pos = traci.person.getPosition(person_id)[0]
    #         if (pos_v > self.CROSS_RANGE_DOWN) and (
    #             pos_v < (self.CROSS_RANGE_UP + traci.vehicle.getLength(self.curr_vehicle_id))):
    #             # 行人避让车辆
    #             if (pos<self.CROSS_RANGE_LEFT and pos>(self.CROSS_RANGE_LEFT - 2.0)) or \
    #                     (pos<(self.CROSS_RANGE_RIGHT + 2.0) and pos>self.CROSS_RANGE_RIGHT):
    #                 traci.person.setSpeed(person_id, speed=0.00000001)
    #                 PedstrianCollision = False
    #         else:
    #             if traci.person.getSpeed(person_id) == 0:
    #                 traci.person.setSpeed(person_id, speed=1.4)
    #     return PedstrianCollision

    # def is_brake(self, veh_id):
    #     position_veh = traci.vehicle.getPosition(veh_id)[1]
    #     action_step_length = traci.vehicle.getActionStepLength(veh_id)
    #     v = traci.vehicle.getSpeed(veh_id)
    #     length = traci.vehicle.getLength(veh_id)
    #     # step执行后，车头超过斑马线
    #     if position_veh > self.CROSS_RANGE_DOWN and traci.vehicle.getSpeed(self.curr_vehicle_id)>1:
    #         # # step执行前，车尾在斑马线
    #         # if (position_veh - action_step_length * v - length) < self.CROSS_RANGE_UP:
    #         # step执行后，车头在斑马线
    #         if (position_veh - action_step_length * v ) < self.CROSS_RANGE_UP:
    #             # 是否有行人在人行线内
    #             for person_id in self.curr_vehicle_pedestrian:
    #                 position_person = traci.person.getPosition(person_id)[0]
    #                 if (position_person >= self.CROSS_RANGE_LEFT) and (position_person <= self.CROSS_RANGE_RIGHT):
    #                     print(self.curr_vehicle_id,'broken')
    #                     return True
    #     return False
