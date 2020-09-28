# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:14:09 2019

@author: Administrator
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from keras.models import Sequential  #
from keras.layers import Dense, Activation  #
from keras.optimizers import Adam, SGD  # 采用adam对神经网络进行优化
from keras.models import load_model


class UE(object):
    def __init__(self):
        self.name = "UE1"
        self.x = 0.0
        self.y = 0.0
        self.channelgain = 0.0
        self.sinr = 0.0
        self.task_b = []  # in bits
        self.task_d = []  # in CPU computation cycles
        self.cur_task_b = 0
        self.cur_task_d = 0
        self.u2f_cq = np.zeros((3, 4))
        self.task_MP = np.zeros((5, 5))  # markov matrix
        self.task_MP1 = np.zeros((5, 5))  # markov matrix
        self.channel_MP1 = np.zeros((4, 4))
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
        self.cap = 0  # the computation capability of F-AP Hz
        self.cf = 10 ** (-30)
        self.comp_limit = 0



class FranEnv(object):
    def __init__(self, UE_nm, FAP_nm):
        self.UE_nm = UE_nm
        self.FAP_nm = FAP_nm
        self.fap = []  # F-AP object
        self.ue = []  # UE object
        self.ff = np.ones(1000)
        self.u2f_distance = np.zeros((UE_nm, FAP_nm))
        self.link_cf = 10 ** (-8)
        self.cloud_cf =  1.5 * (10 ** (-30))
        self.cloud_cap = 6 * (10 ** 9)
        self.para_init()


    def para_init(self):
        for ue_index in range(0, self.UE_nm):
            self.ue.append(UE())

            self.ue[ue_index].task_b = [3, 2, 1, 4, 5]     # in bits，10^5
            self.ue[ue_index].task_d = [21, 12, 5, 32, 45]  # in CPU-cycles，10^8

            self.ue[ue_index].channel_MP1[0] = [0.34, 0.15, 0.32, 0.19]
            self.ue[ue_index].channel_MP1[1] = [0.28, 0.01, 0.13, 0.58]
            self.ue[ue_index].channel_MP1[2] = [0.23, 0.15, 0.39, 0.23]
            self.ue[ue_index].channel_MP1[3] = [0.14, 0.40, 0.20, 0.26]

            # transfer probability matrix 1
            self.ue[ue_index].calchannel_MP1[0] = [0.34, 0.49, 0.81, 1]
            self.ue[ue_index].calchannel_MP1[1] = [0.28, 0.29, 0.42, 1]
            self.ue[ue_index].calchannel_MP1[2] = [0.23, 0.38, 0.77, 1]
            self.ue[ue_index].calchannel_MP1[3] = [0.14, 0.54, 0.74, 1]

            # transfer probability matrix 2
            self.ue[ue_index].channel_MP2[0] = [0.48, 0.29, 0.02, 0.21]
            self.ue[ue_index].channel_MP2[1] = [0.28, 0.12, 0.40, 0.20]
            self.ue[ue_index].channel_MP2[2] = [0.32, 0.22, 0.31, 0.15]
            self.ue[ue_index].channel_MP2[3] = [0.26, 0.10, 0.34, 0.30]


            self.ue[ue_index].calchannel_MP2[0] = [0.48, 0.77, 0.79, 1]
            self.ue[ue_index].calchannel_MP2[1] = [0.28, 0.40, 0.80, 1]
            self.ue[ue_index].calchannel_MP2[2] = [0.32, 0.54, 0.85, 1]
            self.ue[ue_index].calchannel_MP2[3] = [0.26, 0.36, 0.70, 1]

            # transfer probability matrix 3
            self.ue[ue_index].channel_MP3[0] = [0.42, 0.13, 0.14, 0.31]
            self.ue[ue_index].channel_MP3[1] = [0.36, 0.22, 0.37, 0.05]
            self.ue[ue_index].channel_MP3[2] = [0.32, 0.14, 0.35, 0.19]
            self.ue[ue_index].channel_MP3[3] = [0.23, 0.10, 0.35, 0.32]

            self.ue[ue_index].calchannel_MP3[0] = [0.42, 0.55, 0.69, 1]
            self.ue[ue_index].calchannel_MP3[1] = [0.36, 0.58, 0.95, 1]
            self.ue[ue_index].calchannel_MP3[2] = [0.32, 0.46, 0.81, 1]
            self.ue[ue_index].calchannel_MP3[3] = [0.23, 0.33, 0.68, 1]

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

        # task transfer probability matrix for UE 6
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

        # Channel gain for UE 1
        self.ue[0].u2f_cq[0] = [1, 1.5, 3, 3.5]
        self.ue[0].u2f_cq[1] = [2, 2.5, 4, 4.5]
        self.ue[0].u2f_cq[2] = [1.1, 2.3, 3.5, 5.6]

        # Channel gain for UE 2
        self.ue[1].u2f_cq[0] = [3, 6, 4, 5]
        self.ue[1].u2f_cq[1] = [2.8, 5, 3.7, 1]
        self.ue[1].u2f_cq[2] = [5.3, 2.4, 3.9, 6.7]

        # Channel gain for UE 3
        self.ue[2].u2f_cq[0] = [1.4, 5, 2, 1.6]
        self.ue[2].u2f_cq[1] = [4.1, 3, 3.6, 2.1]
        self.ue[2].u2f_cq[2] = [2.2, 4.3, 5.6, 7]

        # Channel gain for UE 4
        self.ue[3].u2f_cq[0] = [4.5, 3.6, 5.7, 6]
        self.ue[3].u2f_cq[1] = [6.2, 7, 4, 5.5]
        self.ue[3].u2f_cq[2] = [3.6, 5.1, 5.2, 4]

        # Channel gain for UE 5
        self.ue[4].u2f_cq[0] = [5.5, 6, 4.9, 7]
        self.ue[4].u2f_cq[1] = [1.9, 2.7, 4.3, 4.9]
        self.ue[4].u2f_cq[2] = [2.1, 3.9, 4.2, 3.3]

        # Channel gain for UE 6
        self.ue[5].u2f_cq[0] = [6.6, 5.3, 4.2, 4.5]
        self.ue[5].u2f_cq[1] = [5.7, 3, 6.2, 4.9]
        self.ue[5].u2f_cq[2] = [4.5, 5.3, 6.6, 3.1]

        # Initialization for the computation capability of F-APs
        for i in range(0, self.FAP_nm):
            self.fap.append(FAP())

        self.fap[0].cap = 23
        self.fap[1].cap = 25
        self.fap[2].cap = 21

        self.fap[0].comp_limit = 23
        self.fap[1].comp_limit = 25
        self.fap[2].comp_limit = 21

        self.fap[0].rank = 2
        self.fap[1].rank = 3
        self.fap[2].rank = 1

        self.fap[0].channelrank = 0
        self.fap[1].channelrank = 0
        self.fap[2].channelrank = 0


    def sample(self):  # randomly generate an action
        return random.randint(0, self.nb_actions - 1)


    def start_state(self, len1):
        new_state = np.zeros((self.UE_nm, len1))
        for ue_index in range(self.UE_nm):
            new_state[ue_index][len1 - 2] = max(self.ue[ue_index].task_b)
            new_state[ue_index][len1 - 1] = max(self.ue[ue_index].task_d)
            for i in range(self.UE_nm):
                new_state[ue_index][i * self.FAP_nm] = 1
                new_state[ue_index][i * self.FAP_nm + 1] = 0
                new_state[ue_index][i * self.FAP_nm + 2] = 0
            for fap_index in range(0, self.FAP_nm):
                if fap_index == 0:
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = min(self.ue[ue_index].u2f_cq[fap_index])
                else:
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = max(self.ue[ue_index].u2f_cq[fap_index])
        return new_state


    # new state computation
    def dqn_state_compute(self, action_index, len1, index1, gain1):
        new_state = np.zeros((self.UE_nm, len1))
        nex_index = [5, 5, 5, 5, 5, 5]  # the next state

        # the requested task for the next state
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
                if action_index[i] == 0:  # the current user select F-AP 1
                    new_state[ue_index][i * self.FAP_nm] = 1
                    new_state[ue_index][i * self.FAP_nm + 1] = 0
                    new_state[ue_index][i * self.FAP_nm + 2] = 0
                elif action_index[i] == 1:  # the current user select F-AP 2
                    new_state[ue_index][i * self.FAP_nm] = 0
                    new_state[ue_index][i * self.FAP_nm + 1] = 1
                    new_state[ue_index][i * self.FAP_nm + 2] = 0
                elif action_index[i] == 2:  # the current user select F-AP 3
                    new_state[ue_index][i * self.FAP_nm] = 0
                    new_state[ue_index][i * self.FAP_nm + 1] = 0
                    new_state[ue_index][i * self.FAP_nm + 2] = 1


            for fap_index in range(0, self.FAP_nm):
                if fap_index == 0:  # the channel gain between F-AP 1 and other users
                    x = self.next_channelgain(ue_index, fap_index, gain1)
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][x]

                elif fap_index == 1:  # the channel gain between F-AP 2 and other users
                    a = self.next_channelgain(ue_index, fap_index, gain1)
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]

                elif fap_index == 2:  # the channel gain between F-AP 3 and other users
                    b = self.next_channelgain(ue_index, fap_index, gain1)
                    new_state[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]

        return new_state


    def next_channelgain(self, ue_index, fap_index, gain1):
        c = random.random()  # generate a random number
        for j in range(0, 4):
            if c < self.ue[ue_index].calchannel_MP1[gain1[ue_index * self.FAP_nm + fap_index]][j]:
                break

        return j

    def search_channelgain(self, ue_index, fap_index, cur_state):
        for p in range(0, 4):
            if self.ue[ue_index].u2f_cq[fap_index][p] == cur_state[ue_index][self.UE_nm * self.FAP_nm + fap_index]:  # the channel gain between user and F-AP 1
                break

        return p

    def comp_initial(self):     # reinitialization for the computation capability
        self.fap[0].comp_limit = 23
        self.fap[1].comp_limit = 25
        self.fap[2].comp_limit = 21

    # calculate the next state and reward
    def dqn_step(self, cur_state, action_number, state_len):
        W = 20 * (10 ** 6)  # channel bandwidth，单位Hz
        T_upload = 0.04
        N1 = 0  # the number of users connect to F-AP 1
        N2 = 0  # the number of users connect to F-AP 2
        N3 = 0  # the number of users connect to F-AP 3
        cur_index = [5, 5, 5, 5, 5, 5]
        gain2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        reward = np.zeros(self.UE_nm)

        P_noise = 1  # noise power
        wireless_consumption = np.zeros(self.UE_nm)  # users' wireless transmission energy consumption
        fap_consumption = np.zeros(self.UE_nm)  # computing energy consumption
        total_consumption = np.zeros(self.UE_nm)  # total energy consumption for each user


        for ue_index in range(self.UE_nm):
            for i in range(0, 5):
                if self.ue[ue_index].task_b[i] == cur_state[ue_index][state_len - 2]:
                    cur_index[ue_index] = i
                    break

            for fap_index in range(self.FAP_nm):
                gain2[ue_index * self.FAP_nm + fap_index] = self.search_channelgain(ue_index, fap_index, cur_state)

        # calculate the next state
        new_state = self.dqn_state_compute(action_number, state_len, cur_index, gain2)


        FAP1_ue = []
        FAP2_ue = []
        FAP3_ue = []
        # TDMA
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                N1 += 1
                FAP1_ue.append(ue_index)
            elif action_number[ue_index] == 1:
                N2 += 1
                FAP2_ue.append(ue_index)
            elif action_number[ue_index] == 2:
                N3 += 1
                FAP3_ue.append(ue_index)

        # calculate the task uploading time for each user
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                self.ue[ue_index].t_upload = (1 / N1) * T_upload
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm] * (10 ** 3)
            elif action_number[ue_index] == 1:
                self.ue[ue_index].t_upload = (1 / N2) * T_upload
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 1] * (10 ** 3)
            elif action_number[ue_index] == 2:
                self.ue[ue_index].t_upload = (1 / N3) * T_upload
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 2] * (10 ** 3)
            # get the sizes of tasks, computation CPU cycles for tasks
            self.ue[ue_index].cur_task_b = cur_state[ue_index][state_len - 2] * (10 ** 5)  # in bits
            self.ue[ue_index].cur_task_d = cur_state[ue_index][state_len - 1] * (10 ** 8)  # in CPU cycles
            # task transmission rate
            self.ue[ue_index].data_rate = self.ue[ue_index].cur_task_b / self.ue[ue_index].t_upload
            # transmission power
            self.ue[ue_index].txP = [(2 ** (self.ue[ue_index].data_rate / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            # calculate the reward
            wireless_consumption[ue_index] = self.ue[ue_index].txP * self.ue[ue_index].t_upload
            if action_number[ue_index] == 0:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 0, FAP1_ue, cur_state)
            elif action_number[ue_index] == 1:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 1, FAP2_ue, cur_state)
            elif action_number[ue_index] == 2:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 2, FAP3_ue, cur_state)
            # total computing energy consumption
            total_consumption[ue_index] = wireless_consumption[ue_index] + fap_consumption[ue_index]
            # reward
            reward[ue_index] = round((total_consumption[ue_index] * (-1)), 5)

        reward_total = reward[0] + reward[1] + reward[2] + reward[3] + reward[4] + reward[5]

        for i in range(self.UE_nm):
            reward[i] = reward_total

        return new_state, reward

    # the energy consumption for DRL training, exhaustive search, priority-based search, random selection
    def dqn_step1(self, cur_state, action_number, state_len):
        W = 20 * (10 ** 6)
        T_upload = 0.04
        t_upload = np.zeros(self.UE_nm)
        data_rate = np.zeros(self.UE_nm)
        txP = np.zeros(self.UE_nm)
        N1 = 0
        N2 = 0
        N3 = 0
        reward = np.zeros(self.UE_nm)  # reward for 6 users

        P_noise = 1  # noise power
        wireless_consumption = np.zeros(self.UE_nm)  # wireless transmission energy consumption for users
        fap_consumption = np.zeros(self.UE_nm)
        total_consumption = np.zeros(self.UE_nm)

        FAP1_ue = []
        FAP2_ue = []
        FAP3_ue = []
        # TDMA
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                N1 += 1
                FAP1_ue.append(ue_index)
            elif action_number[ue_index] == 1:
                N2 += 1
                FAP2_ue.append(ue_index)
            elif action_number[ue_index] == 2:
                N3 += 1
                FAP3_ue.append(ue_index)

        self.ue[ue_index].cur_task_b = cur_state[ue_index][state_len - 2] * (10 ** 5)  # in bits
        self.ue[ue_index].cur_task_d = cur_state[ue_index][state_len - 1] * (10 ** 8)  # in CPU cycles
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:  # the current user connect to F-AP 1
                t_upload[ue_index] = (1 / N1) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            elif action_number[ue_index] == 1:  # the current user connect to F-AP 2
                t_upload[ue_index] = (1 / N2) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 1] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            elif action_number[ue_index] == 2:  # the current user connect to F-AP 3
                t_upload[ue_index] = (1 / N3) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 2] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain

            # calculate the energy consumption and reward
            wireless_consumption[ue_index] = txP[ue_index] * t_upload[ue_index]
            if action_number[ue_index] == 0:  # the current user connect to F-AP 1
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 0, FAP1_ue, cur_state)
            elif action_number[ue_index] == 1:  # the current user connect to F-AP 2
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 1, FAP2_ue, cur_state)
            elif action_number[ue_index] == 2:  # the current user connect to F-AP 3
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 2, FAP3_ue, cur_state)
            # the total computation energy consumption
            total_consumption[ue_index] = wireless_consumption[ue_index] + fap_consumption[ue_index]
            # reward
            reward[ue_index] = round((total_consumption[ue_index] * (-1)), 5)

        reward_total = reward[0] + reward[1] + reward[2] + reward[3] + reward[4] + reward[5]

        for i in range(self.UE_nm):
            reward[i] = reward_total

        return reward

    def dqn_state_compute1(self, action_index1, action_index4, action_index5, action_index6, len1, index1, gain1):
            new_state1 = np.zeros((self.UE_nm, len1))  # DRL-new state
            new_state4 = np.zeros((self.UE_nm, len1))  # random selection-new state
            new_state5 = np.zeros((self.UE_nm, len1))  # greedy algorithm-new state
            new_state6 = np.zeros((self.UE_nm, len1))  # genetic algorithm-new state
            nex_index = [5, 5, 5, 5, 5, 5]  # the next state

            # the task requested for the next state
            for ue_index in range(self.UE_nm):
                p1 = random.random()
                for j in range(0, 5):
                    if p1 < self.ue[ue_index].task_MP1[index1[ue_index]][j]:
                        break
                nex_index[ue_index] = j

            # calculate the next state for 6 users
            for ue_index in range(self.UE_nm):
                new_state1[ue_index][len1 - 2] = self.ue[ue_index].task_b[nex_index[ue_index]]      # DRL
                new_state1[ue_index][len1 - 1] = self.ue[ue_index].task_d[nex_index[ue_index]]
                new_state4[ue_index][len1 - 2] = self.ue[ue_index].task_b[nex_index[ue_index]]      # random selection
                new_state4[ue_index][len1 - 1] = self.ue[ue_index].task_d[nex_index[ue_index]]
                new_state5[ue_index][len1 - 2] = self.ue[ue_index].task_b[nex_index[ue_index]]      # greedy algorithm-based selection
                new_state5[ue_index][len1 - 1] = self.ue[ue_index].task_d[nex_index[ue_index]]
                new_state6[ue_index][len1 - 2] = self.ue[ue_index].task_b[nex_index[ue_index]]      # genetic algorithm-based selection
                new_state6[ue_index][len1 - 1] = self.ue[ue_index].task_d[nex_index[ue_index]]
                for i in range(self.UE_nm):
                    if action_index1[i] == 0:
                        new_state1[ue_index][i * self.FAP_nm] = 1
                        new_state1[ue_index][i * self.FAP_nm + 1] = 0
                        new_state1[ue_index][i * self.FAP_nm + 2] = 0
                    elif action_index1[i] == 1:
                        new_state1[ue_index][i * self.FAP_nm] = 0
                        new_state1[ue_index][i * self.FAP_nm + 1] = 1
                        new_state1[ue_index][i * self.FAP_nm + 2] = 0
                    elif action_index1[i] == 2:
                        new_state1[ue_index][i * self.FAP_nm] = 0
                        new_state1[ue_index][i * self.FAP_nm + 1] = 0
                        new_state1[ue_index][i * self.FAP_nm + 2] = 1
                for l in range(self.UE_nm):     # random selection
                    if action_index4[l] == 0:
                        new_state4[ue_index][l * self.FAP_nm] = 1
                        new_state4[ue_index][l * self.FAP_nm + 1] = 0
                        new_state4[ue_index][l * self.FAP_nm + 2] = 0
                    elif action_index4[l] == 1:
                        new_state4[ue_index][l * self.FAP_nm] = 0
                        new_state4[ue_index][l * self.FAP_nm + 1] = 1
                        new_state4[ue_index][l * self.FAP_nm + 2] = 0
                    elif action_index4[l] == 2:
                        new_state4[ue_index][l * self.FAP_nm] = 0
                        new_state4[ue_index][l * self.FAP_nm + 1] = 0
                        new_state4[ue_index][l * self.FAP_nm + 2] = 1
                for m in range(self.UE_nm):     # greedy algorithm-based selection
                    if action_index5[m] == 0:
                        new_state5[ue_index][m * self.FAP_nm] = 1
                        new_state5[ue_index][m * self.FAP_nm + 1] = 0
                        new_state5[ue_index][m * self.FAP_nm + 2] = 0
                    elif action_index5[m] == 1:
                        new_state5[ue_index][m * self.FAP_nm] = 0
                        new_state5[ue_index][m * self.FAP_nm + 1] = 1
                        new_state5[ue_index][m * self.FAP_nm + 2] = 0
                    elif action_index5[m] == 2:
                        new_state5[ue_index][m * self.FAP_nm] = 0
                        new_state5[ue_index][m * self.FAP_nm + 1] = 0
                        new_state5[ue_index][m * self.FAP_nm + 2] = 1
                for n in range(self.UE_nm):    # genetic-algorithm based selection
                    if action_index6[n] == 0:
                        new_state6[ue_index][n * self.FAP_nm] = 1
                        new_state6[ue_index][n * self.FAP_nm + 1] = 0
                        new_state6[ue_index][n * self.FAP_nm + 2] = 0
                    elif action_index6[n] == 1:
                        new_state6[ue_index][n * self.FAP_nm] = 0
                        new_state6[ue_index][n * self.FAP_nm + 1] = 1
                        new_state6[ue_index][n * self.FAP_nm + 2] = 0
                    elif action_index6[n] == 2:
                        new_state6[ue_index][n * self.FAP_nm] = 0
                        new_state6[ue_index][n * self.FAP_nm + 1] = 0
                        new_state6[ue_index][n * self.FAP_nm + 2] = 1

                # the channel gain between the current user and other F-APs
                for fap_index in range(0, self.FAP_nm):
                    if fap_index == 0:
                        x = self.next_channelgain(ue_index, fap_index, gain1)

                        new_state1[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][x]     # DRL
                        new_state4[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][x]     # random selection
                        new_state5[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][x]     # greedy-algorithm based selection
                        new_state6[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][x]     # genetic-algorithm based selection
                    elif fap_index == 1:
                        a = self.next_channelgain(ue_index, fap_index, gain1)
                        new_state1[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]
                        new_state4[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]
                        new_state5[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]
                        new_state6[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][a]
                    elif fap_index == 2:
                        b = self.next_channelgain(ue_index, fap_index, gain1)
                        new_state1[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]
                        new_state4[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]
                        new_state5[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]
                        new_state6[ue_index][self.UE_nm * self.FAP_nm + fap_index] = self.ue[ue_index].u2f_cq[fap_index][b]

            return new_state1, new_state4, new_state5, new_state6

    def dqn_step_state(self, cur_state1, action_number1, action_number4, action_number5, action_number6, state_len):
        cur_index = [5, 5, 5, 5, 5, 5]      # the current state
        gain2 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        for ue_index in range(self.UE_nm):
            for i in range(0, 5):
                if self.ue[ue_index].task_b[i] == cur_state1[ue_index][state_len - 2]:
                    cur_index[ue_index] = i
                    break

            for fap_index in range(self.FAP_nm):
                gain2[ue_index * self.FAP_nm + fap_index] = self.search_channelgain(ue_index, fap_index, cur_state1)

        new_state1, new_state4, new_state5, new_state6= self.dqn_state_compute1(action_number1, action_number4, action_number5, action_number6, state_len, cur_index, gain2)

        return new_state1, new_state4, new_state5, new_state6



    def FAP_offdecision(self, ue_index1, fap_index1, FAP_ue, cur_state1):
        fap_consumption = np.zeros(len(FAP_ue))
        link_consumption = np.zeros(len(FAP_ue))
        cloud_consumption = np.zeros(len(FAP_ue))
        cal_consumption = np.zeros(len(FAP_ue))
        consumption = 0
        fap_ue = []
        sort_comp = []

        self.comp_initial()

        if len(FAP_ue) > 0:
            for i in range(len(FAP_ue)):
                fap_consumption[i] = cur_state1[FAP_ue[i]][-1] * (10 ** 8) * ((self.fap[fap_index1].cap * (10 ** 8)) ** 2) * self.fap[fap_index1].cf  # 此任务在云端计算，功耗为
                # the link transmission energy consumption
                link_consumption[i] = cur_state1[FAP_ue[i]][-2] * (10 ** 5) * self.link_cf
                cloud_consumption[i] = link_consumption[i] + (cur_state1[FAP_ue[i]][-1] * (10 ** 8)) * ((self.cloud_cap) ** 2) * self.cloud_cf
                if fap_consumption[i] > cloud_consumption[i]:
                    cal_consumption[i] = cloud_consumption[i]
                elif fap_consumption[i] < cloud_consumption[i]:
                    fap_ue.append(i)
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

                if sort_comp[a] < self.fap[fap_index1].comp_limit:
                    # 功耗为在FAP本地计算功耗
                    cal_consumption[number] = fap_consumption[number]
                    self.fap[fap_index1].comp_limit -= sort_comp[a]
                elif sort_comp[a] > self.fap[fap_index1].comp_limit:
                    cal_consumption[number] = cloud_consumption[number]
                    # print(1)
            for ue_index in range(self.UE_nm):
                if ue_index1 == (FAP_ue[ue_index]):
                    break

            consumption = cal_consumption[ue_index]

        return consumption

    # greedy algorithm,
    def greedy(self, cur_state5):
        self.fap[0].comp_limit = 23
        self.fap[1].comp_limit = 25
        self.fap[2].comp_limit = 21
        fap = [3, 3, 3, 3, 3, 3]
        for ue_index in range(self.UE_nm):
            fap_consump = []
            fap1_consump = cur_state5[ue_index][-1] * (10 ** 8) * ((self.fap[0].cap * (10 ** 8)) ** 2) * self.fap[0].cf
            fap2_consump = cur_state5[ue_index][-1] * (10 ** 8) * ((self.fap[1].cap * (10 ** 8)) ** 2) * self.fap[1].cf
            fap3_consump = cur_state5[ue_index][-1] * (10 ** 8) * ((self.fap[2].cap * (10 ** 8)) ** 2) * self.fap[2].cf
            fap_consump.append(fap1_consump);fap_consump.append(fap2_consump);fap_consump.append(fap3_consump)
            index1 = fap_consump.index(min(fap_consump))
            if cur_state5[ue_index][-1] > self.fap[index1].comp_limit:
                if ue_index != 0:
                    for index2 in range(ue_index):
                        if index2 == index1:
                            fap_consump[index1]= float("inf")
                            index1 = fap_consump.index(min(fap_consump))
                            break;
                fap[ue_index] = index1
            else:
                self.fap[index1].comp_limit -= cur_state5[ue_index][-1]
                fap[ue_index] = index1

        return fap


# ************DQN network architecture **************************#
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


    def create_model(self, nb_actions, observation_shape):
        model = Sequential()
        model.add(Dense(64, input_dim=observation_shape, activation="relu"))  # input layer
        model.add(Dense(32, activation="relu"))  # hidden layer #1
        model.add(Dense(nb_actions))  # output layer
        model.add(Activation('linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, step):  # epsilon greedy action selection
        epsilon = self.dec * float(step) + float(self.value_max)
        epsilon = max(self.value_min, epsilon)


        if np.random.random() < epsilon:
            return random.randint(0, self.nb_actions - 1)  # return an action
        else:
            return np.argmax(self.model.predict(state.reshape(1, len(state)))[0])

    def Q(self, state):
        return (self.model.predict(state.reshape(1, len(state)))[0])

    def replay(self, done, memory):
        if len(memory) < self.batch_size:
            return

        samples = random.sample(memory, self.batch_size-1)
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

        cur_state, cur_action, cur_reward, cur_new_state = memory[memory_size_t - 1]
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

#****************** genetic algorithm*******************#
class UE1(object):
    def __init__(self):
        self.name = "UE1"
        self.channelgain = 0.0
        self.task_b = []
        self.task_d = []
        self.cur_task_b = 0
        self.cur_task_d = 0

class FAP1(object):
    def __init__(self):
        self.name = "FAP"
        self.cap = 0
        self.cf = 10 ** (-30)
        self.comp_limit = 0

class GA(object):
    def __init__(self, population_size, chromosome_length, max_value, pc, pm, cur_state6):
        self.UE_nm = 6
        self.FAP_nm = 3
        self.ue = []
        self.fap = []
        self.population_size = population_size
        self.choromosome_length = chromosome_length
        self.max_value = max_value
        self.pc = pc
        self.pm = pm
        self.link_cf = 10 ** (-8)
        self.cloud_cf = 1.5 * (10 ** (-30))
        self.cloud_cap = 6 * (10 ** 9)
        self.cur_state6 = cur_state6
        self.para_init()

    def para_init(self):
        for ue_index in range(0, self.UE_nm):
            self.ue.append(UE1())
            self.ue[ue_index].task_b = [3, 2, 1, 4, 5]
            self.ue[ue_index].task_d = [21, 12, 5, 32, 45]

        for i in range(0, self.FAP_nm):
            self.fap.append(FAP1())

        self.fap[0].cap = 23
        self.fap[1].cap = 25
        self.fap[2].cap = 21

        self.fap[0].comp_limit = 23
        self.fap[1].comp_limit = 25
        self.fap[2].comp_limit = 21


    def species_origin(self):
        population = [[]]
        for i in range(self.population_size):
            temporary = []

            for j in range(self.choromosome_length):

                a = random.randint(0,1)
                b = random.randint(0,1)

                while (a+b)==2:
                    a = random.randint(0,1)
                    b = random.randint(0,1)
                temporary.append(a)
                temporary.append(b)


            population.append(temporary)

        return population[1:]


    def translation(self, population):
        temporary = [[]]
        for i in range(len(population)):
            total = []
            for j in range(self.choromosome_length):
                total1 = population[i][j*2]*(2**1)+population[i][j*2+1]*(2**0)
                total.append(total1)

            temporary.append(total)

        return temporary[1:]


    def function(self, population):
        function1 = []
        temporary = self.translation(population)
        state_len = 23


        for i in range(len(temporary)):

            consumption = self.step1(self.cur_state6, temporary[i], state_len)

            consumption1 = 1/consumption
            function1.append(consumption1)

        return function1

    def step1(self, cur_state, action_number, state_len):
        W = 20 * (10 ** 6)
        T_upload = 0.04
        t_upload = np.zeros(self.UE_nm)
        data_rate = np.zeros(self.UE_nm)
        txP = np.zeros(self.UE_nm)
        N1 = 0
        N2 = 0
        N3 = 0
        reward = np.zeros(self.UE_nm)

        P_noise = 1  # noise power
        wireless_consumption = np.zeros(self.UE_nm)  # the wireless transmission energy consumption for user
        fap_consumption = np.zeros(self.UE_nm)
        total_consumption = np.zeros(self.UE_nm)


        FAP1_ue = []
        FAP2_ue = []
        FAP3_ue = []
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                N1 += 1
                FAP1_ue.append(ue_index)
            elif action_number[ue_index] == 1:
                N2 += 1
                FAP2_ue.append(ue_index)
            elif action_number[ue_index] == 2:
                N3 += 1
                FAP3_ue.append(ue_index)

        self.ue[ue_index].cur_task_b = cur_state[ue_index][state_len - 2] * (10 ** 5)
        self.ue[ue_index].cur_task_d = cur_state[ue_index][state_len - 1] * (10 ** 8)
        for ue_index in range(self.UE_nm):
            if action_number[ue_index] == 0:
                t_upload[ue_index] = (1 / N1) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            elif action_number[ue_index] == 1:
                t_upload[ue_index] = (1 / N2) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 1] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            elif action_number[ue_index] == 2:
                t_upload[ue_index] = (1 / N3) * T_upload
                data_rate[ue_index] = self.ue[ue_index].cur_task_b / t_upload[ue_index]
                self.ue[ue_index].channelgain = cur_state[ue_index][self.UE_nm * self.FAP_nm + 2] * (10 ** 3)
                txP[ue_index] = [(2 ** (data_rate[ue_index] / W)) - 1] * (P_noise) / self.ue[ue_index].channelgain
            elif action_number[ue_index] == 3:  # exception
                txP[ue_index] = 100000

            # calculate the energy consumption, reward
            # calculate the wireless transmission energy consumption
            wireless_consumption[ue_index] = txP[ue_index] * t_upload[ue_index]
            if action_number[ue_index] == 0:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 0, FAP1_ue, cur_state)
            elif action_number[ue_index] == 1:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 1, FAP2_ue, cur_state)
            elif action_number[ue_index] == 2:
                fap_consumption[ue_index] = self.FAP_offdecision(ue_index, 2, FAP3_ue, cur_state)
            elif action_number[ue_index] == 3:  # exception
                fap_consumption[ue_index] = 100000
            total_consumption[ue_index] = wireless_consumption[ue_index] + fap_consumption[ue_index]
            reward[ue_index] = (total_consumption[ue_index])

        reward_total = reward[0] + reward[1] + reward[2] + reward[3] + reward[4] + reward[5]

        for i in range(self.UE_nm):
            reward[i] = reward_total

        return reward[0]

    def FAP_offdecision(self, ue_index1, fap_index1, FAP_ue, cur_state1):
        fap_consumption = np.zeros(len(FAP_ue))
        link_consumption = np.zeros(len(FAP_ue))
        cloud_consumption = np.zeros(len(FAP_ue))
        cal_consumption = np.zeros(len(FAP_ue))
        consumption = 0
        fap_ue = []
        sort_comp = []
        if fap_index1 == 0:
            self.fap[fap_index1].comp_limit = 23
        elif fap_index1 == 1:
            self.fap[fap_index1].comp_limit = 25
        elif fap_index1 == 2:
            self.fap[fap_index1].comp_limit = 21

        if len(FAP_ue) > 0:
            for i in range(len(FAP_ue)):
                fap_consumption[i] = cur_state1[FAP_ue[i]][-1] * (10 ** 8) * ((self.fap[fap_index1].cap * (10 ** 8)) ** 2) * self.fap[fap_index1].cf  # 此任务在云端计算，功耗为
                link_consumption[i] = cur_state1[FAP_ue[i]][-2] * (10 ** 5) * self.link_cf
                cloud_consumption[i] = link_consumption[i] + (cur_state1[FAP_ue[i]][-1] * (10 ** 8)) * ((self.cloud_cap) ** 2) * self.cloud_cf
                if fap_consumption[i] > cloud_consumption[i]:
                    cal_consumption[i] = cloud_consumption[i]
                elif fap_consumption[i] < cloud_consumption[i]:
                    fap_ue.append(i)
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

                if sort_comp[a] < self.fap[fap_index1].comp_limit:
                    cal_consumption[number] = fap_consumption[number]
                    self.fap[fap_index1].comp_limit -= sort_comp[a]
                elif sort_comp[a] > self.fap[fap_index1].comp_limit:
                    cal_consumption[number] = cloud_consumption[number]

            for ue_index in range(self.UE_nm):
                if ue_index1 == (FAP_ue[ue_index]):
                    break

            consumption = cal_consumption[ue_index]

        return consumption


    def fitness(self, function1):
        fitness_value = []
        num = len(function1)
        Cmin = 0
        for i in range(num):
            temporary = Cmin + function1[i]
            fitness_value.append(temporary)
        return fitness_value


    def sum(self, fitness_value):
        total = 0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total


    def cumsum(self, fitness1):
        for i in range(len(fitness1)):
            total = 0
            j = 0

            while (j <= i):
                total += fitness1[j]
                j += 1
            fitness1[i] = total

    def selection(self, population, fitness_value):
        new_fitness = []
        total_fitness = self.sum(fitness_value)
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)
        self.cumsum(new_fitness)
        ms = []
        population_length = pop_len = len(population)
        for i in range(pop_len):
            ms.append(random.random())
        fitin = 0
        newin = 0
        new_population = new_pop = population

        while newin < pop_len:
            if (ms[newin] < new_fitness[fitin]):
                new_pop[newin] = population[fitin]
                newin += 1
            else:
                fitin += 1
        population = new_pop


    def crossover(self, population):
        pop_len = len(population)
        for i in range(pop_len - 1):
            if (random.random() < self.pc):
                cpoint1 = random.randint(0, len(population[0])/2)
                cpoint = cpoint1*2
                temporary1 = []
                temporary2 = []

                temporary1.extend(population[i][0:cpoint])
                temporary1.extend(population[i + 1][cpoint:len(population[i])])
                temporary2.extend(population[i + 1][0:cpoint])
                temporary2.extend(population[i][cpoint:len(population[i])])
                population[i] = temporary1
                population[i + 1] = temporary2


# gene mutation
    def mutation(self, population):
        px = len(population)
        py = len(population[0])
        for i in range(px):
            if (random.random() < self.pm):
                mpoint = random.randint(0, py - 1)
                if (population[i][mpoint] == 1):
                    population[i][mpoint] = 0
                else:
                    population[i][mpoint] = 1

    # transform the binary to decimalism
    def b2d(self, best_individual):
        total = []
        b = len(best_individual)
        limit = int(b/2)
        for i in range(limit):
            total1 = best_individual[i*2]*(2**1)+best_individual[i*2+1]*(2**0)
            total.append(total1)

        return total

    # find the best individual
    def best(self, population, fitness_value):
        px = len(population)
        bestindividual = population[0]
        bestfitness = fitness_value[0]
        for i in range(1, px):
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                bestindividual = population[i]
        return bestindividual, bestfitness

    def main(self):
        results = [[]]
        population = pop = self.species_origin()
        for i in range(100):
            function_value = self.function(population)
            fitness_value = self.fitness(function_value)
            best_individual, best_fitness = self.best(population, fitness_value)
            results.append([best_fitness, self.b2d(best_individual)])
            self.selection(population, fitness_value)
            self.crossover(population)
            self.mutation(population)
        results = results[1:]
        results.sort()
        return results[-1][1]



