# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:12:59 2019

@author: Betty Ren
FL_1. py - main program
FL_2. py - store the required classes and functions
run FL_1.py, start training
python version: Python 3.5 (Tensorflow)
keras._version_: 1.2.0
"""

# import the necessary packages#
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from sklearn.cluster import SpectralClustering
from FL_2 import FranEnv, DQN
#from FL_2 import federated_average, federated_target_average
import time

ue_nm = 6  # the number of users
fap_nm = 3  # the number of F-APs
test_flag = True
qos_sinr_cue = 3  # in dB    3dB = 2bps/Hz


gamma = 1  # discount factor  
lr = 0.0001  # adam learning rate
value_max = 1.0  # the maximum value of epsilon
value_min = 0.01  # the minimum value of epsilon
batch_size = 16  #
nb_steps = 30  # the number of steps contained in one epoch
trials = 60000  # the number of epoch
nb_steps_ex = nb_steps * trials / 3  # explore controlling
nb_steps_update = 3  # the update interval of model parameters
updateTargetNetwork = 900  # the update interval of target model parameters

ini_len = 1000  # for initialization of replay memory
memory = []
memory_size = 3000

time_start = time.time()

# environment building

env = FranEnv(ue_nm, fap_nm)  # network environment settings




state_len = env.FAP_nm * env.UE_nm + env.FAP_nm + 2
nb_actions = 1 * env.FAP_nm  # the number of action space



# initialization
initial_state = env.start_state(state_len)      # initial state
print(initial_state)



# memory initialization
memory = deque(maxlen=memory_size)

ini_action_index = [3, 3, 3, 3, 3, 3]  # the access point selection for 6 users
ini_reward = []
ini_new_state = []
ini_cur_state = initial_state
for ini_index in range(ini_len):
    for ue_index in range(env.UE_nm):
        ini_action_index[ue_index] = random.randint(0, nb_actions - 1)  # the action chosen for each user
    ini_new_state, ini_reward = env.dqn_step(ini_cur_state, ini_action_index, state_len)  # the next state and reward
    for ue_index in range(env.UE_nm):
        memory.append([ini_cur_state[ue_index], ini_action_index[ue_index], ini_reward[ue_index], ini_new_state[ue_index]])

    ini_cur_state = ini_new_state

# a drl_agent is built ***********#
drl_agent = DQN(env, gamma, lr, value_max, value_min, nb_steps_ex, nb_actions, state_len, batch_size, memory)




drl_agent.target_train()  # update the target model


steps = []
loss_per_epoch = []
ave_reward_epoch = []
q_mean = []
total_step = 0
consumption_UE = []  # consumption array
loss = []
ave_consumption_UE = []  # average energy consumption array
ave_loss_step = []
ave_q1=[]
ave_q2=[]
ave_q3=[]

with open("avg_reward.txt", "w")as f:
    with open("loss.txt", "w")as f7:
        with open("q1.txt", "w")as f13:
            with open("q2.txt", "w")as f14:
                with open("q3.txt", "w")as f15:
                    for trial in range(trials):
                        cur_state = initial_state
                        action = [3, 3, 3, 3, 3, 3]
                        done = False  # boolean variable
                        reward_record = []  # reward
                        loss_total = 0
                        UE1_Q1 = []
                        UE1_Q2 = []
                        UE1_Q3 = []

                        for step in range(nb_steps):
                            env.fap[0].comp_limit = 23  # the limited CPU cycles for F-AP's computation
                            env.fap[1].comp_limit = 25
                            env.fap[2].comp_limit = 21
                            if step == nb_steps - 1:
                                done = True
                            total_step += 1
                            #iteration loop
                            for ue_index in range(env.UE_nm):
                                action[ue_index] = drl_agent.act(cur_state[ue_index],total_step)

                            # generate the next state and reward
                            new_state, reward = env.dqn_step(cur_state, action, state_len)

                            q = drl_agent.Q(cur_state[0])

                            UE1_Q1.append(q[0])
                            UE1_Q2.append(q[1])
                            UE1_Q3.append(q[2])

                            # store the current state, action, reward, next state into each individual's memory
                            for ue_index in range(env.UE_nm):
                                memory.append([cur_state[ue_index], action[ue_index], reward[ue_index], new_state[ue_index]])


                            # The memory size is limited. When the memory capacity is reached, a record will be deleted every time a record is added.
                            # Through statement memory.append(), the far left record can be automatically deleted


                            # Update Q Network using mini-batch : Experience Replay
                            if total_step % nb_steps_update == 0:
                                for ue_index in range(env.UE_nm):
                                    x_train, y_train = drl_agent.replay(done, memory)
                            # mini-batch training
                                    loss_temp = drl_agent.model.train_on_batch(x_train, y_train)
                                    loss_total += loss_temp
                                    loss.append(loss_temp)

                            # update the current state
                            cur_state = new_state


                            # update the Target Q Network
                            if (total_step) % updateTargetNetwork == 0:
                                drl_agent.target_train()
                            if done:
                                break

                            consumption_UE.append(reward[0] * (-1))

                        ave_q1.append(np.mean(np.array(UE1_Q1)))
                        ave_q2.append(np.mean(np.array(UE1_Q2)))
                        ave_q3.append(np.mean(np.array(UE1_Q3)))


                        ave_consumption_UE.append(np.mean(np.array(consumption_UE[-1])))
                        ave_loss_step.append(np.mean(np.array(loss[-1])))


                        f.write("EPOCH=%03d,avg_reward= %.5f" % (trial + 1, ave_consumption_UE[-1]))
                        f.write('\n')
                        f.flush()

                        f7.write("EPOCH=%03d,loss= %.5f" % (trial + 1, ave_loss_step[-1]))
                        f7.write('\n')
                        f7.flush()

                        f13.write("EPOCH=%03d,q= %.5f" % (trial + 1, ave_q1[-1]))
                        f13.write('\n')
                        f13.flush()
                        f14.write("EPOCH=%03d,q= %.5f" % (trial + 1, ave_q2[-1]))
                        f14.write('\n')
                        f14.flush()
                        f15.write("EPOCH=%03d,q= %.5f" % (trial + 1, ave_q3[-1]))
                        f15.write('\n')
                        f15.flush()

                        print("Trial {}".format(trial))  # The output is the number of trial




drl_agent.save_model()      #save drl training model

time_end = time.time()
