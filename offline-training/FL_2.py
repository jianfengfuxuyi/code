
#import the necessary packages#
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.optimizers import Adam, SGD  # adopt ADAM to optimize neural networks
from keras.models import load_model


class UE(object):
    def __init__(self):
        self.name = "UE1"
        self.x = 0.0
        self.y = 0.0
        self.txP = 0.0
        self.channelgain = 0.0
        self.sinr = 0.0
        self.sinr_dB = 0.0
        self.data_rate = 0.0
        self.t_upload = 0.0
        self.task_b = []  # size array of tasks, in bits
        self.task_d = []  # computation resource required for completing tasks, in CPU cycles
        self.cur_task_b = 0  # the size of the requested task, in bits
        self.cur_task_d = 0  # the computation resource required for completing the task, in CPU cycles
        self.u2f_cq = np.zeros((3, 4))   # channel gain matrix  
        self.task_MP = np.zeros((5, 5))  # transmission probability matrix for tasks
        self.task_MP1 = np.zeros((5, 5))
        self.channel_MP1 = np.zeros((4, 4)) # transmission probability matrix for channel gain
        self.channel_MP2 = np.zeros((4, 4))
        self.channel_MP3 = np.zeros((4, 4))
        self.calchannel_MP1 = np.zeros((4, 4))
        self.calchannel_MP2 = np.zeros((4, 4))
        self.calchannel_MP3 = np.zeros((4, 4))


class FAP(object):
    def __init__(self):
        self.name = "FAP"
        self.x = 0.0
        self.y = 0.0
        self.cap = 0
        self.cf = 10 ** (-10)  # the computation constant factor for F-AP
        self.comp_limit = 0



class FranEnv(object):
    def __init__(self, UE_nm, FAP_nm):
        self.UE_nm = UE_nm
        self.FAP_nm = FAP_nm
        self.fap = []
        self.ue = []
        self.ff = np.ones(1000)
        self.u2f_distance = np.zeros((UE_nm, FAP_nm))  # distance
        self.link_cf = 10 ** (-5)                     #the computation constant factor for fronthaul links
        self.cloud_cf =  1.5 * (10 ** (-8))            #the computation constant factor for cloud
        self.cloud_cap = 6 * (10 ** 6)
        self.para_init()


    def para_init(self):
        # initilization
        for ue_index in range(0, self.UE_nm):
            self.ue.append(UE())
            # 5 tasks
            self.ue[ue_index].task_b = [3, 2, 6, 4, 5]  # the size of tasks (in bits) 
            self.ue[ue_index].task_d = [21, 12, 48, 32, 45] # the computation resource required for computing tasks (in CPU cycles)

            '''the transition probability matrix for the channel gain between each user to F-AP 1
            for example, the channel gain value for UE 1 to F-AP 1 can be randomly chosen from 4 values
            thus, the dimension for this matrix is 4*4
            each decimal value represents the probability that the channel gain between UE to F-AP 1 is the mth (column number) value in offloading duration t
            if the channel gain between UE to F-AP 1 is the nth (row number) value in offloading duration t-1
            '''
            self.ue[ue_index].channel_MP1[0] = [0.34, 0.15, 0.32, 0.19]
            self.ue[ue_index].channel_MP1[1] = [0.28, 0.01, 0.13, 0.58]
            self.ue[ue_index].channel_MP1[2] = [0.23, 0.15, 0.39, 0.23]
            self.ue[ue_index].channel_MP1[3] = [0.14, 0.40, 0.20, 0.26]

            
            self.ue[ue_index].calchannel_MP1[0] = [0.34, 0.49, 0.81, 1]
            self.ue[ue_index].calchannel_MP1[1] = [0.28, 0.29, 0.42, 1]
            self.ue[ue_index].calchannel_MP1[2] = [0.23, 0.38, 0.77, 1]
            self.ue[ue_index].calchannel_MP1[3] = [0.14, 0.54, 0.74, 1]

            # the transition probability matrix for the channel gain between each user to F-AP 2
            self.ue[ue_index].channel_MP2[0] = [0.48, 0.29, 0.02, 0.21]
            self.ue[ue_index].channel_MP2[1] = [0.28, 0.12, 0.40, 0.20]
            self.ue[ue_index].channel_MP2[2] = [0.32, 0.22, 0.31, 0.15]
            self.ue[ue_index].channel_MP2[3] = [0.26, 0.10, 0.34, 0.30]


            self.ue[ue_index].calchannel_MP2[0] = [0.48, 0.77, 0.79, 1]
            self.ue[ue_index].calchannel_MP2[1] = [0.28, 0.40, 0.80, 1]
            self.ue[ue_index].calchannel_MP2[2] = [0.32, 0.54, 0.85, 1]
            self.ue[ue_index].calchannel_MP2[3] = [0.26, 0.36, 0.70, 1]

            # the transition probability matrix for the channel gain between each user to F-AP 3
            self.ue[ue_index].channel_MP3[0] = [0.42, 0.13, 0.14, 0.31]
            self.ue[ue_index].channel_MP3[1] = [0.36, 0.22, 0.37, 0.05]
            self.ue[ue_index].channel_MP3[2] = [0.32, 0.14, 0.35, 0.19]
            self.ue[ue_index].channel_MP3[3] = [0.23, 0.10, 0.35, 0.32]


            self.ue[ue_index].calchannel_MP3[0] = [0.42, 0.55, 0.69, 1]
            self.ue[ue_index].calchannel_MP3[1] = [0.36, 0.58, 0.95, 1]
            self.ue[ue_index].calchannel_MP3[2] = [0.32, 0.46, 0.81, 1]
            self.ue[ue_index].calchannel_MP3[3] = [0.23, 0.33, 0.68, 1]

        ''' the transition probability matrix for the task requested by UE 1
            for example, the task requested by UE 1 can be randomly chosen from 5 different tasks
            thus, the dimension for this matrix is 5*5
            each decimal value represents the probability that UE 1 requests mth (column number) task in offloading duration t
            if UE 1 requests nth (row number) task in offloading duration t-1
        '''
        self.ue[0].task_MP[0] = [0.32, 0.44, 0.13, 0.10, 0.01]
        self.ue[0].task_MP[1] = [0.27, 0.34, 0.14, 0.15, 0.1]
        self.ue[0].task_MP[2] = [0.17, 0.50, 0.17, 0.08, 0.08]
        self.ue[0].task_MP[3] = [0.03, 0.31, 0.28, 0.14, 0.24]
        self.ue[0].task_MP[4] = [0.13, 0.21, 0.16, 0.16, 0.34]


        self.ue[0].task_MP1[0] = [0.32, 0.76, 0.89, 0.99, 1]
        self.ue[0].task_MP1[1] = [0.27, 0.61, 0.75, 0.90, 1]
        self.ue[0].task_MP1[2] = [0.17, 0.67, 0.84, 0.92, 1]
        self.ue[0].task_MP1[3] = [0.03, 0.34, 0.62, 0.76, 1]
        self.ue[0].task_MP1[4] = [0.13, 0.34, 0.50, 0.66, 1]

        # the transition probability matrix for the task requested by UE 2, dimension is in 5*5
        self.ue[1].task_MP[0] = [0.2, 0.26, 0.26, 0.22, 0.06]
        self.ue[1].task_MP[1] = [0.26, 0.11, 0.07, 0.29, 0.28]
        self.ue[1].task_MP[2] = [0.24, 0.09, 0.24, 0.29, 0.13]
        self.ue[1].task_MP[3] = [0.18, 0.15, 0.26, 0.28, 0.13]
        self.ue[1].task_MP[4] = [0.13, 0.08, 0.19, 0.45, 0.15]


        self.ue[1].task_MP1[0] = [0.2, 0.46, 0.72, 0.94, 1]
        self.ue[1].task_MP1[1] = [0.26, 0.37, 0.44, 0.73, 1]
        self.ue[1].task_MP1[2] = [0.24, 0.33, 0.57, 0.86, 1]
        self.ue[1].task_MP1[3] = [0.18, 0.33, 0.59, 0.87, 1]
        self.ue[1].task_MP1[4] = [0.13, 0.21, 0.4, 0.85, 1]

        # the transition probability matrix for the task requested by UE 3, dimension is in 5*5
        self.ue[2].task_MP[0] = [0.03, 0.22, 0.11, 0.28, 0.35]
        self.ue[2].task_MP[1] = [0.3, 0.09, 0.29, 0.06, 0.27]
        self.ue[2].task_MP[2] = [0.27, 0.16, 0.29, 0.22, 0.06]
        self.ue[2].task_MP[3] = [0.12, 0.28, 0.24, 0.09, 0.27]
        self.ue[2].task_MP[4] = [0.17, 0.2, 0.23, 0.14, 0.25]


        self.ue[2].task_MP1[0] = [0.03, 0.25, 0.36, 0.64, 1]
        self.ue[2].task_MP1[1] = [0.3, 0.39, 0.68, 0.74, 1]
        self.ue[2].task_MP1[2] = [0.27, 0.43, 0.72, 0.94, 1]
        self.ue[2].task_MP1[3] = [0.12, 0.4, 0.64, 0.73, 1]
        self.ue[2].task_MP1[4] = [0.17, 0.37, 0.6, 0.74, 1]

        # the transition probability matrix for the task requested by UE 4, dimension is in 5*5
        self.ue[3].task_MP[0] = [0.13, 0.27, 0.04, 0.24, 0.32]
        self.ue[3].task_MP[1] = [0.25, 0.06, 0.23, 0.21, 0.25]
        self.ue[3].task_MP[2] = [0.23, 0.28, 0.08, 0.36, 0.05]
        self.ue[3].task_MP[3] = [0.03, 0.12, 0.01, 0.44, 0.4]
        self.ue[3].task_MP[4] = [0.29, 0.21, 0.1, 0.3, 0.1]


        self.ue[3].task_MP1[0] = [0.13, 0.4, 0.44, 0.68, 1]
        self.ue[3].task_MP1[1] = [0.25, 0.31, 0.54, 0.75, 1]
        self.ue[3].task_MP1[2] = [0.23, 0.51, 0.59, 0.95, 1]
        self.ue[3].task_MP1[3] = [0.03, 0.15, 0.16, 0.6, 1]
        self.ue[3].task_MP1[4] = [0.29, 0.5, 0.6, 0.9, 1]

        # the transition probability matrix for the task requested by UE 5, dimension is in 5*5
        self.ue[4].task_MP[0] = [0.12, 0.06, 0.42, 0.23, 0.17]
        self.ue[4].task_MP[1] = [0.36, 0.02, 0.25, 0.19, 0.18]
        self.ue[4].task_MP[2] = [0.02, 0.28, 0.22, 0.34, 0.14]
        self.ue[4].task_MP[3] = [0.07, 0.25, 0.15, 0.32, 0.21]
        self.ue[4].task_MP[4] = [0.34, 0.31, 0.07, 0.2, 0.08]


        self.ue[4].task_MP1[0] = [0.12, 0.18, 0.6, 0.83, 1]
        self.ue[4].task_MP1[1] = [0.36, 0.38, 0.63, 0.82, 1]
        self.ue[4].task_MP1[2] = [0.02, 0.3, 0.52, 0.86, 1]
        self.ue[4].task_MP1[3] = [0.07, 0.32, 0.47, 0.79, 1]
        self.ue[4].task_MP1[4] = [0.34, 0.65, 0.72, 0.92, 1]

        # the transition probability matrix for the task requested by UE 6, dimension is in 5*5
        self.ue[5].task_MP[0] = [0.04, 0.12, 0.39, 0.19, 0.26]
        self.ue[5].task_MP[1] = [0.1, 0.05, 0.01, 0.34, 0.5]
        self.ue[5].task_MP[2] = [0.26, 0.18, 0.03, 0.46, 0.07]
        self.ue[5].task_MP[3] = [0.1, 0.33, 0.14, 0.27, 0.16]
        self.ue[5].task_MP[4] = [0.09, 0.27, 0.1, 0.23, 0.31]


        self.ue[5].task_MP1[0] = [0.04, 0.16, 0.55, 0.74, 1]
        self.ue[5].task_MP1[1] = [0.1, 0.15, 0.16, 0.5, 1]
        self.ue[5].task_MP1[2] = [0.26, 0.44, 0.47, 0.93, 1]
        self.ue[5].task_MP1[3] = [0.1, 0.43, 0.57, 0.84, 1]
        self.ue[5].task_MP1[4] = [0.09, 0.36, 0.46, 0.69, 1]

        ''' the channel gain value matrix between UE 1 to F-AP 1-3
        row - UE 1 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[0].u2f_cq[0] = [1, 1.5, 3, 3.5]
        self.ue[0].u2f_cq[1] = [2, 2.5, 4, 4.5]
        self.ue[0].u2f_cq[2] = [1.1, 2.3, 3.5, 5.6]

        ''' the channel gain value matrix between UE 2 to F-AP 1-3
        row - UE 2 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[1].u2f_cq[0] = [3, 6, 4, 5]
        self.ue[1].u2f_cq[1] = [2.8, 5, 3.7, 1]
        self.ue[1].u2f_cq[2] = [5.3, 2.4, 3.9, 6.7]

        ''' the channel gain value matrix between UE 3 to F-AP 1-3
        row - UE 3 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[2].u2f_cq[0] = [1.4, 5, 2, 1.6]
        self.ue[2].u2f_cq[1] = [4.1, 3, 3.6, 2.1]
        self.ue[2].u2f_cq[2] = [2.2, 4.3, 5.6, 7]

        ''' the channel gain value matrix between UE 4 to F-AP 1-3
        row - UE 4 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[3].u2f_cq[0] = [4.5, 3.6, 5.7, 6]
        self.ue[3].u2f_cq[1] = [6.2, 7, 4, 5.5]
        self.ue[3].u2f_cq[2] = [3.6, 5.1, 5.2, 4]

        ''' the channel gain value matrix between UE 5 to F-AP 1-3
        row - UE 5 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[4].u2f_cq[0] = [5.5, 6, 4.9, 7]
        self.ue[4].u2f_cq[1] = [1.9, 2.7, 4.3, 4.9]
        self.ue[4].u2f_cq[2] = [2.1, 3.9, 4.2, 3.3]

        ''' the channel gain value matrix between UE 6 to F-AP 1-3
        row - UE 6 to F-AP x, x = 1,2,3
        column - 4 values can be chosen for each channel gain 
        '''
        self.ue[5].u2f_cq[0] = [6.6, 5.3, 4.2, 4.5]
        self.ue[5].u2f_cq[1] = [5.7, 3, 6.2, 4.9]
        self.ue[5].u2f_cq[2] = [4.5, 5.3, 6.6, 3.1]

        # initialization for the computation constraint of each F-AP
        for i in range(0, self.FAP_nm):
            self.fap.append(FAP())

        self.fap[0].cap = 23
        self.fap[1].cap = 25
        self.fap[2].cap = 21
        
        self.fap[0].comp_limit = 23
        self.fap[1].comp_limit = 25
        self.fap[2].comp_limit = 21



    def sample(self):
        return random.randint(0, self.nb_actions - 1)

    # initialization for the beginning state
    def start_state(self, len1):
        new_state = np.zeros((self.UE_nm, len1))
        for ue_index in range(self.UE_nm):
            new_state[ue_index][len1 - 2] = max(self.ue[ue_index].task_b)
            new_state[ue_index][len1 - 1] = max(self.ue[ue_index].task_d)
            for i in range(self.UE_nm):
                new_state[ue_index][i * self.FAP_nm] = 1      # choose to connect F-AP 1
                new_state[ue_index][i * self.FAP_nm + 1] = 0  # choose not to connect F-AP 2
                new_state[ue_index][i * self.FAP_nm + 2] = 0  # choose not to connect F-AP 3
            for fap_index in range(0, self.FAP_nm):
            # the channel gain for the next state
                if fap_index == 0:
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = min(self.ue[ue_index].u2f_cq[fap_index])
                else:
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = max(self.ue[ue_index].u2f_cq[fap_index])
        return new_state

    # new state computation
    def dqn_state_compute(self, action_index, len1, index1, gain1):
        new_state = np.zeros((self.UE_nm, len1))
        nex_index = [5, 5, 5, 5, 5, 5]


        for ue_index in range(self.UE_nm):
            p1 = random.random()
            for j in range(0, 5):
                if p1 < self.ue[ue_index].task_MP1[index1[ue_index]][j]:
                    break
            nex_index[ue_index] = j


        for ue_index in range(self.UE_nm):
            new_state[ue_index][len1 - 2] = self.ue[ue_index].task_b[nex_index[ue_index]]
            new_state[ue_index][len1 - 1] = self.ue[ue_index].task_d[nex_index[ue_index]]
            for i in range(self.UE_nm):
                if action_index[i] == 0:  # the current user choose to connect F-AP 1
                    new_state[ue_index][i * self.FAP_nm] = 1
                    new_state[ue_index][i * self.FAP_nm + 1] = 0
                    new_state[ue_index][i * self.FAP_nm + 2] = 0
                elif action_index[i] == 1:  # the current user choose to connect F-AP 2
                    new_state[ue_index][i * self.FAP_nm] = 0
                    new_state[ue_index][i * self.FAP_nm + 1] = 1
                    new_state[ue_index][i * self.FAP_nm + 2] = 0
                elif action_index[i] == 2:  # the current user choose to connect F-AP 3
                    new_state[ue_index][i * self.FAP_nm] = 0
                    new_state[ue_index][i * self.FAP_nm + 1] = 0
                    new_state[ue_index][i * self.FAP_nm + 2] = 1

            # the transition for the channel gain
            for fap_index in range(0, self.FAP_nm):
                if fap_index == 0:
                    c = random.random()
                    for j in range(0, 4):
                        if c < self.ue[ue_index].calchannel_MP1[gain1[ue_index * self.FAP_nm]][j]:
                            break


                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][j]

                elif fap_index == 1:  # the channel gain between F-AP 2 to each user
                    c = random.random()
                    for a in range(0, 4):
                        if c < self.ue[ue_index].calchannel_MP2[gain1[ue_index * self.FAP_nm + fap_index]][a]:
                            break


                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]

                elif fap_index == 2:  # the channel gain between F-AP 3 to each user
                    c = random.random()
                    for b in range(0, 4):
                        if c < self.ue[ue_index].calchannel_MP3[gain1[ue_index * self.FAP_nm + fap_index]][b]:
                            break


                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]

        return new_state

    # calculate the next state and reward
    def dqn_step(self, cur_state, action_number, state_len):
        W = 20 * (10 ** 7)  # channel bandwidth, in Hz
        T_upload = 0.04  # uploading time, in s
        N1 = 0
        N2 = 0
        N3 = 0
        cur_index = [5, 5, 5, 5, 5, 5]  # the serial number for requested tasks
        gain2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # the serial number for channel gain
        reward = np.zeros(self.UE_nm)

        P_noise = 1  # the noise power
        wireless_consumption = np.zeros(self.UE_nm)
        fap_consumption = np.zeros(self.UE_nm)
        total_consumption = np.zeros(self.UE_nm)
    
        # the power for transmitted signal, SINR ......
        for ue_index in range(self.UE_nm):
            for i in range(0, 5):
                if self.ue[ue_index].task_b[i] == cur_state[ue_index][state_len - 2]:
                    cur_index[ue_index] = i      # the serial number for the next requested task
                    break

            for p in range(0, 4):
                if self.ue[ue_index].u2f_cq[0][p] == cur_state[ue_index][self.UE_nm * self.FAP_nm]:  # the channel gain between users to F-AP 1
                    break
            # the channel gain value between users to F-AP 1 for the next state
            gain2[ue_index * self.FAP_nm] = p


            for b in range(0, 4):
                if self.ue[ue_index].u2f_cq[1][b] == cur_state[ue_index][self.UE_nm * self.FAP_nm + 1]: # the channel gain between users to F-AP 2
                    break
            # the channel gain value between users to F-AP 2 for the next state
            gain2[ue_index * self.FAP_nm + 1] = b


            for q in range(0, 4):
                if self.ue[ue_index].u2f_cq[2][q] == cur_state[ue_index][self.UE_nm * self.FAP_nm + 2]: # the channel gain between users to F-AP 3
                    break
             # the channel gain value between users to F-AP 3 for the next state
            gain2[ue_index * self.FAP_nm + 2] = q

        # the next state
        new_state = self.dqn_state_compute(action_number, state_len, cur_index, gain2)


        # store the serial number of connected UEs
        FAP1_ue=[]
        FAP2_ue=[]
        FAP3_ue=[]
        # TDMA
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                N1 += 1
                FAP1_ue.append(ue_index)
            elif action_number[ue_index] == 1:
                N2 += 1  # the number of users connect to F-AP 2
                FAP2_ue.append(ue_index)
            elif action_number[ue_index] == 2:
                N3 += 1
                FAP3_ue.append(ue_index)

        #calculate the transmission time
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:  # connect to F-AP 1
                self.ue[ue_index].t_upload = (1 / N1) * T_upload
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm] * (10 ** 3)
            elif action_number[ue_index] == 1:  # connect to F-AP 2
                self.ue[ue_index].t_upload = (1 / N2) * T_upload
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 1] * (10 ** 3)
            elif action_number[ue_index] == 2:  # connect to F-AP 3
                self.ue[ue_index].t_upload = (1 / N3) * T_upload
                #channel gain
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 2] * (10 ** 3)
            # the size of the task, the required computation CPU cycles
            self.ue[ue_index].cur_task_b = cur_state[ue_index][state_len - 2] * (10 ** 5)
            self.ue[ue_index].cur_task_d = cur_state[ue_index][state_len - 1] * (10 ** 8)
            # transmission rate
            self.ue[ue_index].data_rate = self.ue[ue_index].cur_task_b / self.ue[ue_index].t_upload
            self.ue[ue_index].txP = [(2 ** (self.ue[ue_index].data_rate / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            # calculate the energy consumption
            wireless_consumption[ue_index] = self.ue[ue_index].txP * self.ue[ue_index].t_upload
            if action_number[ue_index] == 0:    # the current user connect to F-AP 1
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 0, FAP1_ue, cur_state)
            elif action_number[ue_index] == 1:  # the current user connect to F-AP 2
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 1, FAP2_ue, cur_state)
            elif action_number[ue_index] == 2:  # the current user connect to F-AP 3
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 2, FAP3_ue, cur_state)
            total_consumption[ue_index] = wireless_consumption[ue_index] + fap_consumption[ue_index]
            # calculate the reward
            reward[ue_index] = round((total_consumption[ue_index] * (-1)), 5)

        reward_total = reward[0]+ reward[1]+ reward[2]+ reward[3]+ reward[4]+ reward[5]

        for i in range(self.UE_nm):
            reward[i] = reward_total

        return new_state, reward

    # greedy algorithm for F-APs' decisions on request forwarding
    def FAP_offdecision(self, ue_index1, fap_index1, FAP_ue, cur_state1):
        fap_consumption = np.zeros(len(FAP_ue))
        link_consumption = np.zeros(len(FAP_ue))
        cloud_consumption = np.zeros(len(FAP_ue))
        cal_consumption = np.zeros(len(FAP_ue))
        consumption = 0
        fap_ue = []
        sort_comp = []  # sort by the calculation amount


        if fap_index1 == 0:
            self.fap[fap_index1].comp_limit = 23
        elif fap_index1 == 1:
            self.fap[fap_index1].comp_limit = 25
        elif fap_index1 == 2:
            self.fap[fap_index1].comp_limit = 21

        if len(FAP_ue) > 0:  # there exists user who connects to F-AP
            for i in range(len(FAP_ue)):
                # F-AP energy consumption
                fap_consumption[i] = cur_state1[FAP_ue[i]][-1] * (10 ** 8) * ((self.fap[fap_index1].cap * (10 ** 8)) ** 2) * self.fap[fap_index1].cf  # 此任务在云端计算，功耗为
                # fronthaul energy consumption
                link_consumption[i] = cur_state1[FAP_ue[i]][-2] * (10 ** 5) * self.link_cf
                # cloud energy consumption
                cloud_consumption[i] = link_consumption[i] + (cur_state1[FAP_ue[i]][-1] * (10 ** 8)) * ((self.cloud_cap) ** 2) * self.cloud_cf
                if fap_consumption[i] > cloud_consumption[i]:
                    cal_consumption[i] = cloud_consumption[i]
                # offload to the cloud
                elif fap_consumption[i] < cloud_consumption[i]:
                    fap_ue.append(i)  # remember the current iteration number
                    sort_comp.append(cur_state1[FAP_ue[i]][-1])


            for k in range(1, len(fap_ue)):
                for j in range(i, 0, -1):
                    if sort_comp[j] < sort_comp[j - 1]:
                        sort_comp[j], sort_comp[j - 1] = sort_comp[j - 1], sort_comp[j]
                    else:
                        break
            sort_comp.sort(reverse=True)


            for a in range(len(sort_comp)):
                for b in range(len(fap_ue)):
                    if sort_comp[a] == cur_state1[FAP_ue[fap_ue[b]]][-1]:
                        break

                number = fap_ue[b]

                if sort_comp[a] < self.fap[fap_index1].comp_limit:  # the computation burden does not exceed the computation limitation
                    # local F-AP energy consumption
                    cal_consumption[number] = fap_consumption[number]
                    self.fap[fap_index1].comp_limit -= sort_comp[a]
                elif sort_comp[a] > self.fap[fap_index1].comp_limit:
                    cal_consumption[number] = cloud_consumption[number]  # cloud energy consumption
            for ue_index in range(self.UE_nm):
                if ue_index1 == (FAP_ue[ue_index]):
                    break

            consumption = cal_consumption[ue_index]

        return consumption


# ************DQN architecture**************************#
class DQN(object):
    def __init__(self, env, gamma, lr, value_max, value_min, nb_steps_ex, nb_actions, observation_shape, batch_size, memory):
        self.env = env
        self.memory = memory
        self.gamma = gamma
        self.value_max = value_max
        self.value_min = value_min
        self.learning_rate = lr
        self.tau = .125
        self.dec = -float(value_max - value_min) / float(nb_steps_ex)
        self.nb_actions = nb_actions
        self.observation_shape = observation_shape
        self.batch_size = batch_size
        self.model = self.create_model(nb_actions, observation_shape)
        self.target_model = self.create_model(nb_actions, observation_shape)


    # architecture building #
    def create_model(self, nb_actions, observation_shape):
        model = Sequential()
        model.add(Dense(64, input_dim=observation_shape, activation="relu")) # input layer, 64 neurons
        model.add(Dense(32, activation="relu"))  # the second layer, 32 neurons
        model.add(Dense(nb_actions))  # the ouput layer
        model.add(Activation('linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, step):  # epsilon greedy action selection
        epsilon = self.dec * float(step) + float(self.value_max)
        epsilon = max(self.value_min, epsilon)


        if np.random.random() < epsilon:
            return random.randint(0, self.nb_actions - 1)
        else:
            return np.argmax(self.model.predict(state.reshape(1, len(state)))[0])

    def Q(self, state):
        return (self.model.predict(state.reshape(1, len(state)))[0])

    # memory replay function
    def replay(self, done, memory):
        if len(memory) < self.batch_size:
            return

        samples = random.sample(memory, self.batch_size-1)
        # 2 variables to store the estimated q values
        x = np.zeros((self.batch_size, self.observation_shape))
        y = np.zeros((self.batch_size, self.nb_actions))

        memory_size_t = len(memory)


        for i, sample in enumerate(samples):
            state, action, reward, new_state = sample

            target = self.target_model.predict(state.reshape(1, len(state)))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape(1, len(new_state)))[0])
                target[0][action] = reward + Q_future * self.gamma
            x[i] = state
            y[i] = target[0]

        cur_state, cur_action, cur_reward, cur_new_state = memory[memory_size_t - 1]  # experience
        target = self.target_model.predict(cur_state.reshape(1, len(cur_state)))
        if done:
            target[0][cur_action] = cur_reward
        else:

            Q_future = max(self.target_model.predict(cur_new_state.reshape(1, len(cur_new_state)))[0])
            target[0][cur_action] = cur_reward + Q_future * self.gamma

        x[self.batch_size - 1] = cur_state
        y[self.batch_size - 1] = target[0]

        return x, y


    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def save_model(self):
        self.model.save('D:\\Yijing Ren CODE\\Single Agent-2 hidden layers\\Single DQN Model-16\\dqn_model.h5')

