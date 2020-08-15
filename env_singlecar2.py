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


def get_options(nogui=True):
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=nogui, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


class ENV():
    def __init__(self,nogui=True):
        options = get_options(nogui)
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", "sumomodel/cross1.0/exp.sumocfg"])

        #new
        self.curr_broken=False
        ##
        self.action_acc_down=-15
        self.action_acc_up=15
        self.actionStepLength = 0.2
        self.speed_delta=0
        # 车辆进入观测的范围  --ignore-route-errors
        self.Observe_Position_x = 16 + 60
        self.Observe_Position_y = 16 + 30
        # 记录当前交叉口车辆状态
        self.curr_vehicle_id=None
        self.curr_vehicle_state = []
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

    def _get_state(self):
        veh_id=self.curr_vehicle_id
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
        # if traci.trafficlight.getRedYellowGreenState(traci.trafficlight.getIDList()[0])[5] == 'G':
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

        self.curr_vehicle_state=states
        self.curr_vehicle_pedestrian=[person_list[i][0] for i in range(len(person_list))]

    def set_Observe_Position_x(self, x):
        self.Observe_Position_x = x

    def get_state(self):
        return self.curr_vehicle_state

    def action_space_sample(self):

        return np.random.uniform(self.action_acc_down,self.action_acc_up,1)

    def get_total_reward(self):
        # print('speed_reward ', self.speed_reward(veh_id),
        #       ' pedestrian_reward ', self.pedestrian_reward(veh_id),
        #       ' stop_num_reward ', self.stop_num_reward(veh_id))
        if self.curr_vehicle_id is None:
            return -100
        speed_reward=self.speed_reward()
        pedestrian_reward=self.pedestrian_reward()
        stop_num_reward=self.stop_num_reward()
        total_reward= speed_reward + pedestrian_reward + stop_num_reward

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

    # def is_brake(self,veh_id):
    #     Threshold=1.2
    #     curr_vehicle_state=self.curr_vehicle_state
    #     veh_pos=curr_vehicle_state[0:2]
    #     min=np.min([len(self.curr_vehicle_pedestrian), self.MAX_PERSON_COUNT])
    #     for i in range(min):
    #         distance=np.linalg.norm(veh_pos - curr_vehicle_state[6+i*4:8+i*4])
    #         if distance<Threshold and traci.vehicle.getSpeed(veh_id)>1 :
    #             print(self.curr_vehicle_id,  'broken! ','distance:',distance)
    #             self.curr_broken=True
    #             return True
    #
    #     self.curr_broken = False
    #     return False

    def is_brake(self,veh_id):
        veh_x,veh_y2=traci.vehicle.getPosition(veh_id)
        veh_v=traci.vehicle.getSpeed(veh_id)
        veh_y1=veh_y2-veh_v*self.actionStepLength
        veh_right = veh_x + traci.vehicle.getWidth(veh_id) * 0.5
        veh_left = veh_x - traci.vehicle.getWidth(veh_id) * 0.5
        for ped_id in self.curr_vehicle_pedestrian:
            ped_x2, ped_y = traci.person.getPosition(ped_id)
            ped_v = traci.person.getSpeed(ped_id)
            if veh_y2>=ped_y and veh_y1<ped_y:
                if int(re.findall('\d', ped_id)[0]) == 1:
                    ped_x1 = ped_x2 + ped_v * self.actionStepLength
                    if (ped_x2<=veh_right and ped_x1>veh_right) or (ped_x1<=veh_right and ped_x1>=(veh_left+ped_v*self.actionStepLength)):
                        print(self.curr_vehicle_id, 'broken! ')
                        self.curr_broken = True
                        return True
                else:
                    ped_x1 = ped_x2 - ped_v * self.actionStepLength
                    if (ped_x2 >= veh_left and ped_x1 < veh_left) or (ped_x1>=veh_left and ped_x1<=(veh_right-ped_v*self.actionStepLength)):
                        print(self.curr_vehicle_id, 'broken! ')
                        self.curr_broken = True
                        return True
        self.curr_broken = False
        return False

    def del_vehicle(self, veh_id):
        traci.vehicle.moveToXY(veh_id, 'gneE_0', 2, x=-0, y=95)

    def distance_vehicle_pedestrians(self, veh_id, person_id):
        position_veh = np.array(traci.vehicle.getPosition(veh_id))
        position_person = np.array(traci.person.getPosition(person_id))
        return np.linalg.norm(position_veh - position_person)

    # def is_brake(self):
    #     veh_id=self.curr_vehicle_id
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
    #                     print(self.curr_vehicle_id, 'broken! ')
    #                     return True
    #     return False

    def set_action(self,action):
        if self.curr_vehicle_id is None:
            print('There is no veh!')
        else:
            if not action == None:
                old_speed = traci.vehicle.getSpeed(self.curr_vehicle_id)
                speed = np.max([traci.vehicle.getSpeed(self.curr_vehicle_id) + action * traci.vehicle.getActionStepLength(self.curr_vehicle_id), 0])
                traci.vehicle.setSpeed(self.curr_vehicle_id, speed)
                self.speed_delta = speed - old_speed

    def step(self,action=None):
        # 观测车辆当前状态结束的情况
        veh_id=self.curr_vehicle_id
        state=self.curr_vehicle_state
        state_next=state
        terminal=True
        #放弃安全模式
        traci.vehicle.setType(veh_id, 'vtype3')
        traci.vehicle.setLaneChangeMode(veh_id, lcm=0)
        traci.vehicle.setSpeedMode(veh_id, sm=0)
        # 对观测范围内的车辆添加动作
        if not action is None:
            self.set_action(action)
        # 运行一步
        traci.simulationStep()
        self._get_state()
        for v_id in traci.vehicle.getIDList():
            if self.veh_is_observe(v_id):
                if not v_id==veh_id:
                    self.del_vehicle(v_id)
        #判断是否碰撞
        if self.is_brake(veh_id):
            #碰撞情况下移除veh,结束此episode
            self.del_vehicle(veh_id)
            self.curr_vehicle_broken.append(veh_id)
            traci.simulationStep()
        else:
            #非碰撞情况下，判断veh是否离开观测区域
            if self.veh_is_observe(veh_id):
                #未离开,计算state_next;否则结束此episode
                if (traci.vehicle.getSpeed(self.curr_vehicle_id) == 0) and (not self.speed_delta == 0):
                    self.curr_vehicle_stop_num = self.curr_vehicle_stop_num + 1
                state_next = self.curr_vehicle_state
                terminal = False
        if self.veh_is_observe(veh_id):
            # 未离开,计算state_next;否则结束此episode
            if (traci.vehicle.getSpeed(self.curr_vehicle_id) == 0) and (not self.speed_delta == 0):
                self.curr_vehicle_stop_num = self.curr_vehicle_stop_num + 1
            state_next = self.curr_vehicle_state
            terminal = False

        reward = self.get_total_reward()
        return veh_id, state, reward, state_next, terminal

    def reset(self):
        if self.curr_vehicle_id is not None:
            self.del_vehicle(self.curr_vehicle_id)
            self.curr_vehicle_id=None
        while self.curr_vehicle_id is None:
            traci.simulationStep()
            for veh_id in traci.vehicle.getIDList():
                if self.veh_is_observe(veh_id):
                    self.curr_vehicle_id=veh_id
                    return self.step()

    def veh_is_observe(self,veh_id):
        # Observe_Position_x和Observe_Position_y范围内车辆进行观测
        if (traci.vehicle.getPosition(veh_id)[0] <= self.Observe_Position_x) \
                and (traci.vehicle.getPosition(veh_id)[1] <= self.Observe_Position_y):
            return True
        return False


# class Car():
#     def __init__(self,length,width,direction_angle=1.5*np.pi,pos_x=0,pos_y=0):
#         self.length=length
#         self.width=width
#         self.direction_angle=direction_angle
#         self.pos_x=pos_x
#         self.pos_y=pos_y
#         self.k=np.zeros([4,2])
#         self.b = np.zeros(4)
#         self.set_position(direction_angle,pos_x,pos_y)
#
#     def set_position(self,direction_angle,pos_x,pos_y):
#         self.k[0]=np.tan(2*np.pi-np.radians(direction_angle))
#         if self.k
#         self.b[0]=pos_y-self.k[0]*pos_x
#
#         left_up_x=pos_x+self.width/(2*np.sqrt(self.k[0]*self.k[0]+1))
#         left_up_y=self.k[0]*left_up_x+self.b[0]
#         self.k[1] = -1/(self.k[0])
#         self.b[1] = left_up_y - self.k[1] * left_up_x
#
#         left_down_x = pos_x - self.width / (2 * np.sqrt(self.k[0] * self.k[0] + 1))
#         left_down_y = self.k[0] * left_down_x + self.b[0]
#         self.k[2] = self.k[0]
#         self.b[2] = left_down_y - self.k[2] * left_down_x
#
#         right_middle_x = pos_x + self.length *np.cos(1.5*np.pi-np.radians(direction_angle))
#         right_middle_y = pos_y + self.length *np.sin(1.5*np.pi-np.radians(direction_angle))
#         self.k[3] = self.k[1]
#         self.b[3] = right_middle_y - self.k[3] * right_middle_x
#
#     def distance_from_car(self,x,y):
#         pass


# if __name__ == '__main__':
#     c=Car(4.8,2,270,100,0)
#     print(c.k,c.b)

