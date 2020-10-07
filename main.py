import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import SchedulingEnv
from model import RL_model, baseline_DQN, baselines


lamda = 25
EPISODE = 10
policyName = ['my_dqn', 'round-robin', 'earliest', 'DQN', 'best-fit', 'sensible']
policyNum = len(policyName)
start_learn = 500  # DQN parameter
learn_interval = 1  # DQN parameter
my_learn_interval = 1

my_start_learn = 500
my_target_update_freq = 1200
my_greedy = 8
dis_learn_freq = 4

'''
store
'''
performance_lamda = np.zeros((policyNum))
performance_success = np.zeros((policyNum))
performance_util = np.zeros((policyNum))
performance_finishT = np.zeros((policyNum))


#gen env
env = SchedulingEnv(lamda)
#build model
brainRL = baseline_DQN(env.actionNum, env.s_features)
mod = RL_model(env.s_features, env.actionNum)
brainOthers = baselines(env.actionNum, env.VMtypes)

global_step = 0  # DQN parameter
my_learn_step = 0
my_learn_interval = 1
for episode in range(EPISODE):
    print('----------------------------Episode', episode, '----------------------------')
    my_greedy += 0.04
    #my_target_update_freq = max(500, my_target_update_freq - 50)
    job_c = 1  # job counter
    performance_c = 0
    env.reset(lamda)  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    brainOthers.sensible_reset()
    performance_respTs = []

    while True:
        #baseline DQN
        global_step += 1
        finish, job_attrs = env.workload(job_c)
        DQN_state = env.getState(job_attrs, 4)
        if global_step != 1:
                brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        reward_DQN = env.feedback(job_attrs, action_DQN, 4)
        if (global_step > start_learn) and (global_step % learn_interval == 0):  # learn
            brainRL.learn()
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

        # my AI DDQN
        my_DQN_state = env.getState(job_attrs, 1)
        if global_step > start_learn or episode>1:
            p = np.random.randint(10)
            if p < my_greedy or episode>2:
                if episode>2:
                    dev = np.random.rand()
                else:
                    dev = 0
                my_act = mod(torch.FloatTensor(my_DQN_state))

                if dev>0.90:
                    my_act = my_act.detach().numpy()
                    my_act = my_act.argsort()
                    my_act = my_act[-4]
                else:

                    my_act = np.argmax(my_act.detach().numpy())
            else:
                my_act = np.random.randint(10)
        else:
            my_act = action = np.random.randint(10)
        my_reward_DQN = env.feedback(job_attrs, my_act, 1)
        if global_step != 1:
            mod.store_buffer(my_last_state, my_last_action, my_last_reward, my_DQN_state)
        if (global_step > my_start_learn) and (global_step % my_learn_interval) == 0:
            mod.learn(episode)
            if episode>=3:
                mod.ad_learn_dis()
            my_learn_step += 1
        if my_learn_step % my_target_update_freq == 0:
            mod.update_target()
        my_last_state = my_DQN_state
        my_last_action = my_act
        my_last_reward = my_reward_DQN
        # round robin policy
        action_RR = brainOthers.RR_choose_action(job_c)
        reward_RR = env.feedback(job_attrs, action_RR, 2)
        # earliest policy
        idleTimes = env.get_VM_idleT(3)  # get VM state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = env.feedback(job_attrs, action_early, 3)
        # suitable policy
        suit_state = env.getState(job_attrs, 5)  # job type, VM wait time
        action_suit = brainOthers.suit_choose_action(suit_state)  # best
        reward_suit = env.feedback(job_attrs, action_suit, 5)
        #dis buffer
        if global_step != 1:
            mod.store_dis_buffer(last_dis_state, last_dis_action, last_dis_reward, suit_state)
        last_dis_state = suit_state
        last_dis_action = action_suit
        last_dis_reward = reward_suit
        #training dis
        if global_step > 100:
            mod.dis_learn()
        # sensible routing policy
        action_sensible = brainOthers.sensible_choose_action(job_attrs[1])  # job_attrs[1]: job arrivalT
        reward_sensible = env.feedback(job_attrs, action_sensible, 6)
        state_sensible = env.getStateP(job_attrs[0])
        brainOthers.sensible_counter(state_sensible, action_sensible)

        # get performance data
        # choice 2: get performance according to JobNum
        if job_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(policyNum, performance_c, job_c)
            finishTs = env.get_FinishTimes(policyNum, performance_c, job_c)
            avg_exeTs = env.get_executeTs(policyNum, performance_c, job_c)
            avg_waitTs = env.get_waitTs(policyNum, performance_c, job_c)
            avg_respTs = env.get_responseTs(policyNum, performance_c, job_c)
            performance_respTs.append(avg_respTs)
            successTs = env.get_successTimes(policyNum, performance_c, job_c)
            performance_c = job_c

        job_c += 1
        if episode>2:
            my_learn_interval += 2
        if finish:
            break

    # episode performance
    startP = 2000
    total_Rewards = env.get_totalRewards(policyNum, startP)
    avg_allRespTs = env.get_total_responseTs(policyNum, startP)
    total_success = env.get_totalSuccess(policyNum, startP)
    avg_util = env.get_avgUtilitizationRate(policyNum, startP)
    total_Ts = env.get_totalTimes(policyNum, startP)
    # JobDistribution = env.get_JobDistribution(policyNum)

    print('total performance (after 2000 jobs):')
    print('[AI-DDQN] reward:', total_Rewards[0], ' avg_responseT:', avg_allRespTs[0],
          'success_rate:', total_success[0], ' utilizationRate:', avg_util[0], ' finishT:', total_Ts[0])
    print('[RR policy] reward:', total_Rewards[1], ' avg_responseT:', avg_allRespTs[1], '',
          'success_rate:', total_success[1], ' utilizationRate:', avg_util[1], ' finishT:', total_Ts[1])
    print('[earliest policy] reward:', total_Rewards[2], ' avg_responseT:', avg_allRespTs[2], '',
          'success_rate:', total_success[2], ' utilizationRate:', avg_util[2], ' finishT:', total_Ts[2])
    print('[DQN policy] reward:', total_Rewards[3], ' avg_responseT:', avg_allRespTs[3], '',
          'success_rate:', total_success[3], ' utilizationRate:', avg_util[3], ' finishT:', total_Ts[3])
    print('[suitable policy] reward:', total_Rewards[4], ' avg_responseT:', avg_allRespTs[4], '',
          'success_rate:', total_success[4], ' utilizationRate:', avg_util[4], ' finishT:', total_Ts[4])
    print('[sensible policy] reward:', total_Rewards[5], ' avg_responseT:', avg_allRespTs[5], '',
          'success_rate:', total_success[5], ' utilizationRate:', avg_util[5], ' finishT:', total_Ts[5])

    if episode != 0:
        performance_lamda[:] += env.get_total_responseTs(policyNum, 0)
        performance_success[:] += env.get_totalSuccess(policyNum, 0)
        performance_util[:] += env.get_avgUtilitizationRate(policyNum, 0)
        performance_finishT[:] += env.get_totalTimes(policyNum, 0)
print('')

print('---------------------------- avg results ----------------------------')
performance_lamda = np.around(performance_lamda/(EPISODE-1), 3)
performance_success = np.around(performance_success/(EPISODE-1), 3)
performance_util = np.around(performance_util/(EPISODE-1), 3)
performance_finishT = np.around(performance_finishT/(EPISODE-1), 3)
print('avg_responseT:')
print(performance_lamda)
print('success_rate:')
print(performance_success)
print('utilizationRate:')
print(performance_util)
print('finishT:')
print(performance_finishT)


# draw pics
# pic 1: the change of avg respT in one episode
draw_respT = np.array(performance_respTs) * 1000  # ms
x = range(draw_respT.shape[0])
lables = ['s-', '^-', 'o-', 'd-', '*-', 'p-', 'o-']
plt.figure()
for i in range(policyNum):
    y = draw_respT[:, i]
    pn = policyName[i]
    la = lables[i]
    plt.plot(x, y, la, label=pn)
# define axis label
plt.xlabel('time')
plt.ylabel('avg response time (ms)')
plt.legend(loc='upper left')  # add legend

# x sticks
x_sticks = np.linspace(0, draw_respT.shape[0] - 1, draw_respT.shape[0])
# get performance according to jobNum (if job_c % N == 0: , N >=300)
x_sticks_names = np.linspace(1, draw_respT.shape[0], draw_respT.shape[0])
# get performance according to time
# x_sticks_names = np.linspace(1 * performance_showT, draw_respT.shape[0] * performance_showT, draw_respT.shape[0])
x_sticks_names = x_sticks_names.astype(int)
plt.xticks(x_sticks, x_sticks_names)

# plt.title('job request time')
plt.grid(True, linestyle="-.", linewidth=1)
plt.show()


# pic 2: the change of respT along with different lamda values
draw_lamda = np.array(performance_lamda) * 1000  # ms
x = range(draw_lamda.shape[0])
plt.figure()
for i in range(policyNum):
    y = draw_lamda[:, i]
    pn = policyName[i]
    la = lables[i]
    plt.plot(x, y, 'o-', label=pn)
# define axis label
plt.xlabel('Arrival rate (requests/s)')
plt.ylabel('avg response time (ms)')
plt.legend(loc='upper left')  # add legend
# x sticks
x_sticks = np.linspace(0, draw_lamda.shape[0] - 1, draw_lamda.shape[0])
plt.xticks(x_sticks, lamda)
plt.grid(True, linestyle="-.", linewidth=1)
plt.show()


# pic 3: the avg successRate in N episodes (bar pic)
draw_success = np.array(performance_success) * 100
draw_success = np.around(draw_success, 1)
x = list(range(draw_success.shape[0]))
width = 0.8 / policyNum

for i in range(policyNum):
    if i != 0:
        for j in range(len(x)):
            x[j] = x[j] + width
    bars = plt.bar(x, draw_success[:, i], width=width, label=policyName[i])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')
        bar.set_edgecolor('white')

plt.xlabel('Arrival rate (requests/s)')
plt.ylabel('success rate(%)')
x_sticks = np.linspace(0, draw_success.shape[0] - 1, draw_success.shape[0])
plt.xticks(x_sticks + 2 * width, lamda)
plt.legend(loc='best')
plt.grid(True, linestyle="-.", linewidth=1)
plt.show()


# pic 4: the avg utilizationRate in N episodes (bar pic)
draw_util = np.array(performance_util) * 100
draw_util = np.around(draw_util, 1)
x = list(range(draw_util.shape[0]))
width = 0.8 / policyNum

for i in range(policyNum):
    if i != 0:
        for j in range(len(x)):
            x[j] = x[j] + width
    bars = plt.bar(x, draw_util[:, i], width=width, label=policyName[i])
    '''
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')
        bar.set_edgecolor('white')
    '''
plt.xlabel('Arrival rate (requests/s)')
plt.ylabel('VM utilization rate(%)')
x_sticks = np.linspace(0, draw_util.shape[0] - 1, draw_util.shape[0])
plt.xticks(x_sticks + 2.5 * width, lamda)
plt.legend(loc='best')
plt.grid(True, linestyle="-.", linewidth=1)
plt.show()