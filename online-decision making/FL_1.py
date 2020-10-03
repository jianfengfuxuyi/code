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

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from sklearn.cluster import SpectralClustering
from FL_2 import FranEnv, DQN, GA
from keras.models import load_model
#from FL_2 import federated_average, federated_target_average
import time
import scipy.io as sio

ue_nm = 6  # the number of UEs
fap_nm = 3  # the number of F-APs
test_flag = True


gamma = 1  # discount factor
lr = 0.0001  # the learning rate of adam
value_max = 1.0  # the maximum value of epsilon
value_min = 0.01  # the minimum value of epsilon
batch_size = 16  #
nb_steps = 30  # # the number of steps contained in one epoch
trials = 100 # the number of epoch
nb_steps_ex = nb_steps * trials / 3 # explore controlling
nb_steps_update = 3 # the update interval of model parameters
updateTargetNetwork = 900 # the update interval of target model parameters

ini_len = 1000  # for initialization of replay memory
memory = []
memory_size = 3000



# environment building
env = FranEnv(ue_nm, fap_nm)  # network environment settings

state_len = env.FAP_nm * env.UE_nm + env.FAP_nm + 2  # network connection state, channel gain, the size of the task (in bits), CPU cycles required for computation
nb_actions = 1 * env.FAP_nm  # the number of action space


# initialization #
initial_state = env.start_state(state_len)      # initial state
print(initial_state)


# memory initialization #
memory = deque(maxlen=memory_size)

# ini_action = np.ones(env.UE_nm, dtype='intp') * (nb_actions - 1)  # each action represents the serial number
ini_action_index = [3, 3, 3, 3, 3, 3]  # the F-AP selection for 6 users
ini_reward = []
ini_new_state = []
ini_cur_state = initial_state
for ini_index in range(ini_len):  # ini_index indicates the current experience number
    for ue_index in range(env.UE_nm):
        ini_action_index[ue_index] = random.randint(0, nb_actions - 1)
    ini_new_state, ini_reward = env.dqn_step(ini_cur_state, ini_action_index, state_len)

    for ue_index in range(env.UE_nm):
        memory.append([ini_cur_state[ue_index], ini_action_index[ue_index], ini_reward[ue_index], ini_new_state[ue_index]])

    ini_cur_state = ini_new_state

# load model #
model = load_model("D:\\Yijing Ren CODE\\Single DQN Model-Comparing Schemes\\dqn_model.h5")

# drl model #
drl_agent = DQN(env, gamma, lr, value_max, value_min, nb_steps_ex, nb_actions, state_len, batch_size, memory)

drl_agent.model = model     # single-agent DQN model

steps = []
loss_per_epoch = []
ave_reward_epoch = []
q_mean = []
total_step = 0

dqn_epoch_step = np.zeros((trials, nb_steps))     # dqn；store the reward for each step, each epoch
prio_epoch_step = np.zeros((trials, nb_steps))    # priority-based selection
random_epoch_step = np.zeros((trials, nb_steps))  # random-based F-AP selection
greedy_epoch_step = np.zeros((trials, nb_steps))  # greedy algorithm-based F-AP selection
genetic_epoch_step = np.zeros((trials, nb_steps)) # genetic algorithm-based F-AP selection
dqn_consumption_UE = []                           # DQN
prio_consumption_UE = []                          # priority-based selection, store the total reward in each epoch
random_consumption_UE = []                        # random-based selection, store the total reward in each epoch
greedy_consumption_UE = []                        # greedy algorithm-based F-AP selection
genetic_consumption_UE = []                       # genetic algorithm-based F-AP selection
avestep_dqn_consumption = []                      # the average energy consumption in step for DRL
avestep_prio_consumption = []                     # the average energy consumption in step for priority-based F-AP selection
avestep_random_consumption = []                   # the average energy consumption in step for random-based F-AP selection
avestep_greedy_consumption = []                   # the average energy consumption in step for greedy algorithm-based F-AP selection
avestep_genetic_consumption = []                  # the average energy consumption in step for genetic algorithm-based F-AP selection

loss = []


dqn_ave_consumption_UE = []      # DRL-based method
prio_ave_consumption_UE = []     # priority-based selection
random_ave_consumption_UE = []   # random-based selection
greedy_ave_consumption_UE = []   # greedy algorithm
genetic_ave_consumption_UE = []  # genetic algorithm
dqn_time = np.zeros((trials, nb_steps))
random_time = np.zeros((trials, nb_steps))
greedy_time = np.zeros((trials, nb_steps))
genetic_time = np.zeros((trials, nb_steps))


ave_loss_step = []
discounted_reward = []

ave_q1=[]
ave_q2=[]
ave_q3=[]


with open("dqn.txt", "w")as a:
    with open('state_save.txt', 'w')as x1:
        with open("priority.txt", "w")as h:
            with open("random.txt", "w")as z:
                with open("greedy.txt", "w")as u:
                    with open("dqn_state.txt", "w")as a1:
                        with open("check.txt", "w")as c:
                            with open("dqn_epoch.txt", "w")as f:
                                with open("priority_epoch.txt", "w")as f1:
                                    with open("random_epoch.txt", "w")as f2:
                                        with open("genetic.txt", "w")as f3:
                                            with open("ave_loss.txt", "w")as f7:
                                                for trial in range(trials):  # loop for the iteration of epochs
                                                    cur_state1 = initial_state  # DRL initial state
                                                    cur_state4 = initial_state  # the initial state for random selection
                                                    cur_state5 = initial_state  # the initial state for greedy algorithm
                                                    cur_state6 = initial_state  # the initial state for genetic algorithm
                                                    action1 = [3, 3, 3, 3, 3, 3]  # drl action
                                                    action4 = [3, 3, 3, 3, 3, 3]  # random-selection action
                                                    action5 = [3, 3, 3, 3, 3, 3]  # greedy algorithm-based action
                                                    action6 = [3, 3, 3, 3, 3, 3]  # genetic algorithm-based action
                                                    done = False  # boolean variable
                                                    loss_total = 0
                                                    reward1_UE = []  # store each step, drl reward
                                                    reward4_UE = []
                                                    reward5_UE = []
                                                    reward6_UE = []
                                                    reward1_total = []
                                                    reward4_total = []
                                                    reward5_total = []
                                                    reward6_total = []

                                                    for step in range(nb_steps):
                                                        # F-AP computation capability
                                                        env.fap[0].comp_limit = 23
                                                        env.fap[1].comp_limit = 25
                                                        env.fap[2].comp_limit = 21
                                                        ARRS = []
                                                        # save the state for each step
                                                        if step == nb_steps - 1:
                                                            done = True
                                                        total_step += 1

                                                        # DRL action selection
                                                        start1 = time.time()
                                                        for ue_index in range(env.UE_nm):
                                                            action1[ue_index] = drl_agent.act(cur_state1[ue_index],total_step)  # 2. generate action
                                                        reward1 = env.dqn_step1(cur_state1, action1, state_len)
                                                        end1 = time.time()
                                                        time1 = end1 - start1

                                                        for i in range(env.UE_nm):
                                                            jointsFrame = cur_state1[i]
                                                            ARRS.append(jointsFrame)
                                                            for Ji in range(env.FAP_nm*env.UE_nm, state_len):
                                                                strNum = str(jointsFrame[Ji])
                                                                x1.write(strNum)
                                                                x1.write(' ')
                                                            x1.write('\n')
                                                            x1.flush()

                                                        # random-based action selection
                                                        start2 = time.time()
                                                        for ue_index in range(env.UE_nm):
                                                            action4[ue_index] = random.randint(0, nb_actions - 1)
                                                        reward4 = env.dqn_step1(cur_state4, action4, state_len)
                                                        end2 = time.time()
                                                        time2 = end2 - start2

                                                        # greedy-algorithm based action selection
                                                        start3 = time.time()
                                                        action5 = env.greedy(cur_state5)
                                                        reward5 = env.dqn_step1(cur_state5, action5, state_len)
                                                        end3 = time.time()
                                                        time3 = end3 - start3

                                                        # genetic-algorithm based action selection
                                                        start4 = time.time()
                                                        population_size = 100
                                                        max_value = 10
                                                        chromosome_length = 6
                                                        pc = 0.6
                                                        pm = 0.01
                                                        ga = GA(population_size, chromosome_length, max_value, pc, pm, cur_state6)
                                                        action6 = ga.main()
                                                        reward6 = env.dqn_step1(cur_state6, action6, state_len)
                                                        end4 = time.time()
                                                        time4 = end4 - start4

                                                        dqn_time[trial][step] = time1
                                                        random_time[trial][step] = time2
                                                        greedy_time[trial][step] = time3
                                                        genetic_time[trial][step] = time4

                                                        new_state1, new_state4, new_state5, new_state6 = env.dqn_step_state(cur_state1, action1, action4, action5, action6, state_len)

                                                        # store the reward
                                                        reward1_UE.append(reward1[0] * (-1))  # drl
                                                        reward4_UE.append(reward4[0] * (-1))  # random selection
                                                        reward5_UE.append(reward5[0] * (-1))  # greedy algorithm
                                                        reward6_UE.append(reward6[0] * (-1))  # genetic algorithm

                                                        # the stored reward for each epoch
                                                        reward1_total.append(np.sum(reward1_UE))
                                                        reward4_total.append(np.sum(reward4_UE))
                                                        reward5_total.append(np.sum(reward5_UE))
                                                        reward6_total.append(np.sum(reward6_UE))

                                                        a.write("EPOCH=%03d, step=%03d, reward= %.5f" % (trial + 1, step + 1, reward1_UE[-1]))
                                                        a.write('\n')
                                                        a.flush()

                                                        z.write("EPOCH=%03d, step=%03d, reward= %.5f" % (trial + 1, step + 1, reward4_UE[-1]))
                                                        z.write('\n')
                                                        z.flush()

                                                        u.write("EPOCH=%03d, step=%03d, reward= %.5f" % (trial + 1, step + 1, reward5_UE[-1]))
                                                        u.write('\n')
                                                        u.flush()

                                                        f3.write("EPOCH=%03d, step=%03d, reward= %.5f" % (trial + 1, step + 1, reward6_UE[-1]))
                                                        f3.write('\n')
                                                        f3.flush()
                                                        # store the experience into each individual's memory
                                                        for ue_index in range(env.UE_nm):
                                                            memory.append([cur_state1[ue_index], action1[ue_index],reward1[ue_index], new_state1[ue_index]])

                                                        # The memory size is limited. When the memory capacity is reached, a record will be deleted every time a record is added.
                                                        # Through statement memory.append(), the far left record can be automatically deleted

                                                        # Update Q Network using mini-batch : Experience Replay

                                                        # Update the current state
                                                        # if trial >= (trials-81):
                                                        cur_state1 = new_state1  # DRL new state
                                                        cur_state4 = new_state4  # random selection new state
                                                        cur_state5 = new_state5  # greedy algorithm new state
                                                        cur_state6 = new_state6  # genetic algorithm new state

                                                        # update the Target Q Network
                                                        if (total_step) % updateTargetNetwork == 0:
                                                            drl_agent.target_train()  # update the target model
                                                        if done:
                                                            break

                                                        # total energy consumption
                                                        dqn_consumption_UE.append(reward1_UE[-1])     # drl: add the energy consumption
                                                        random_consumption_UE.append(reward4_UE[-1])  # random selection: add the energy consumption
                                                        greedy_consumption_UE.append(reward5_UE[-1])  # greedy algorithm-based selection
                                                        genetic_consumption_UE.append(reward6_UE[-1])  # genetic algorithm-based selection


                                                    for i in range(nb_steps):
                                                        dqn_epoch_step[trial][i] = reward1_total[i]
                                                        random_epoch_step[trial][i] = reward4_total[i]
                                                        greedy_epoch_step[trial][i] = reward5_total[i]
                                                        genetic_epoch_step[trial][i] = reward6_total[i]


                                                    dqn_ave_consumption_UE.append(np.mean(np.array(dqn_consumption_UE[-1])))  # dqn
                                                    random_ave_consumption_UE.append(np.mean(np.array(random_consumption_UE[-1])))  # random selection
                                                    greedy_ave_consumption_UE.append(np.mean(np.array(greedy_consumption_UE[-1])))  # greedy algorithm-based selection
                                                    genetic_ave_consumption_UE.append(np.mean(np.array(genetic_consumption_UE[-1])))  # genetic algortihm-based selection

                                                    f.write("EPOCH=%03d,avg_reward= %.5f" % (trial + 1, dqn_ave_consumption_UE[-1]))
                                                    f.write('\n')
                                                    f.flush()

                                                    f2.write("EPOCH=%03d,avg_reward= %.5f" % (trial + 1, random_ave_consumption_UE[-1]))
                                                    f2.write('\n')
                                                    f2.flush()

                                                    f3.write("EPOCH=%03d,avg_reward= %.5f" % (trial + 1, greedy_ave_consumption_UE[-1]))
                                                    f3.write('\n')
                                                    f3.flush()

                                                    print("Trial {}".format(trial))  # The output is the number of trial



total_t = np.zeros((nb_steps, 4))
with open("dqn_addup.txt", "w")as o:                # record the reward of DRL training
    with open("random_addup.txt", "w")as r:         # record the reward of random selection
        with open("greedy_addup.txt", "w")as t:     # record the reward of greedy algorithm-based approach
            with open("genetic_addup.txt", "w")as s: # record the reward of genetic algorithm-based approach
                for step_index in range(nb_steps):
                    sum1 = 0  # DQN
                    sum4 = 0  # random selection
                    sum5 = 0  # greedy algorithm
                    sum6 = 0  # genetic algorithm
                    for trial_index in range(trials):
                        sum1 += dqn_epoch_step[trial_index][step_index]
                        sum4 += random_epoch_step[trial_index][step_index]
                        sum5 += greedy_epoch_step[trial_index][step_index]
                        sum6 += genetic_epoch_step[trial_index][step_index]
                        total_t[step_index][0] += dqn_time[trial_index][step_index]
                        total_t[step_index][1] += random_time[trial_index][step_index]
                        total_t[step_index][2] += greedy_time[trial_index][step_index]
                        total_t[step_index][3] += genetic_time[trial_index][step_index]
                        if trial_index == (trials - 1):
                            avestep_dqn_consumption.append(sum1 / trials)
                            avestep_random_consumption.append(sum4 / trials)
                            avestep_greedy_consumption.append(sum5 / trials)
                            avestep_genetic_consumption.append(sum6 / trials)
                            total_t[step_index][0] = total_t[step_index][0] / trials
                            total_t[step_index][1] = total_t[step_index][1] / trials
                            total_t[step_index][2] = total_t[step_index][2] / trials
                            total_t[step_index][3] = total_t[step_index][3] / trials


                            o.write("STEP=%03d, reward_addup= %.5f" % (step_index + 1, avestep_dqn_consumption[-1]))
                            o.write('\n')
                            o.flush()
                            r.write("STEP=%03d, reward_addup= %.5f" % (step_index + 1, avestep_random_consumption[-1]))
                            r.write('\n')
                            r.flush()
                            t.write("STEP=%03d, reward_addup= %.5f" % (step_index + 1, avestep_greedy_consumption[-1]))
                            t.write('\n')
                            t.flush()
                            s.write("STEP=%03d, reward_addup= %.5f" % (step_index + 1, avestep_genetic_consumption[-1]))
                            s.write('\n')
                            s.flush()

                            print(avestep_dqn_consumption)

avg_time = np.zeros(4)      #4 comparing schemes
for iter in range(nb_steps):
    avg_time[0] += total_t[iter][0]
    avg_time[1] += total_t[iter][1]
    avg_time[2] += total_t[iter][2]
    avg_time[3] += total_t[iter][3]

avg_time[0] = avg_time[0] / nb_steps
avg_time[1] = avg_time[1] / nb_steps
avg_time[2] = avg_time[2] / nb_steps
avg_time[3] = avg_time[3] / nb_steps

with open("avg_time.txt", "w")as h:
    h.write("dqn = %.5f" % (avg_time[0]))
    h.write('\n')
    h.write("random = %.5f" % (avg_time[1]))
    h.write('\n')
    h.write("greedy = %.5f" % (avg_time[2]))
    h.write('\n')
    h.write("genetic = %.5f" % (avg_time[3]))
    h.flush()
print(avg_time[0], avg_time[1], avg_time[2], avg_time[3])

plt.figure(1)
plt.plot(range(1, trials + 1), dqn_ave_consumption_UE)
plt.xlabel('Epochs')
plt.ylabel('Average Energy Consumption - UE')
plt.savefig('DQN avg_reward - UE.png')

plt.figure(2)
plt.plot(range(1, nb_steps + 1), avestep_dqn_consumption, color = '#FF0000', label = 'dqn')
plt.plot(range(1, nb_steps + 1), avestep_random_consumption, color = '#FFD700', linestyle='--', label = 'random')
plt.plot(range(1, nb_steps + 1), avestep_greedy_consumption, color = '#008000', linestyle='-.', label = 'greedy')
plt.plot(range(1, nb_steps + 1), avestep_genetic_consumption, color = '#FF69B4', linestyle=':', label = 'genetic')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average Energy Consumption - UE')
plt.savefig('Comparing Schemes .png')

