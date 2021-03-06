import numpy as np
from scipy import stats


np.random.seed(3)
class SchedulingEnv:
    def __init__(self, lamda):
        # -------------------------------- VM info -------------------------------------
        self.VMtypes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.VMnum = len(self.VMtypes)
        self.VMcapacity = 1000  # mips
        # -------------------------- state & action space -------------------------------
        self.actionNum = self.VMnum
        # state features: 1-job attributes(length, type)
        #                 2-each VM (wait_T)
        self.s_features = 1 + self.VMnum   # consider waitT
        # ----------------------------------- workload -----------------------------------
        self.jobMI = 200  # short job
        self.jobMI_std = 20

        # generate workload (jobs' arrivalT, length, type, ddl)
        # (choice 1: according to jobNum)
        self.jobNum = 8000
        self.lamda = lamda
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * 0.3  # 250ms = waitT + exeT
        self.gen_workload(self.lamda)

        # -------------------------- scheduling-------------------------------
        # >>>>>>>>>>>>>random policy<<<<<<<<<<<<
        # 1-VM id  2-start time  3-wait time  4-waitT+exeT  5-leave time  6-reward  7-actual_exeT  8- success  9-reject
        self.RAN_events = np.zeros((9, self.jobNum))
        # 1-idle time  2-jobNum on it
        self.RAN_VM_events = np.zeros((2, self.VMnum))
        # >>>>>>>>>>>>>round robin policy<<<<<<<<<<<<
        self.RR_events = np.zeros((9, self.jobNum))
        self.RR_VM_events = np.zeros((2, self.VMnum))
        # >>>>>>>>>>>>>earliest policy<<<<<<<<<<<<
        self.early_events = np.zeros((9, self.jobNum))
        self.early_VM_events = np.zeros((2, self.VMnum))
        # >>>>>>>>>>>>>DQN policy<<<<<<<<<<<<
        self.DQN_events = np.zeros((9, self.jobNum))
        self.DQN_VM_events = np.zeros((2, self.VMnum))
        # >>>>>>>>>>>>>suitable policy<<<<<<<<<<<<
        self.suit_events = np.zeros((9, self.jobNum))
        self.suit_VM_events = np.zeros((2, self.VMnum))
        # >>>>>>>>>>>>>sensible routing policy<<<<<<<<<<<<
        self.sensible_events = np.zeros((9, self.jobNum))
        self.sensible_VM_events = np.zeros((2, self.VMnum))

    def gen_workload(self, lamda):
        # generate arrival time of jobs (poisson distribution)
        intervalT = stats.expon.rvs(scale=1 / lamda, size=self.jobNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[- 1]
        print('last job arrivalT:', round(last_arrivalT, 3))

        # generate jobs' length
        # (according to MI & MIPS, Normal/Gaussian distribution)
        self.jobsMI = np.random.normal(self.jobMI, self.jobMI_std, self.jobNum)
        self.jobsMI = self.jobsMI.astype(int)
        print("MI mean: ", round(np.mean(self.jobsMI), 3), '  MI SD:', round(np.std(self.jobsMI, ddof=1), 3))
        self.lengths = self.jobsMI / self.VMcapacity
        print("length mean: ", round(np.mean(self.lengths), 3), '  length SD:', round(np.std(self.lengths, ddof=1), 3))


        # generate jobs' type
        types = np.zeros(self.jobNum)
        # pro = np.random.uniform(0.2, 0.8)
        pro = 0.5
        for i in range(self.jobNum):
            if np.random.uniform() < pro:
                types[i] = 0
            else:
                types[i] = 1
        self.types = types

    def reset(self, lamda):
        # if each episode generates new workload
        self.arrival_Times = np.zeros(self.jobNum)
        self.jobsMI = np.zeros(self.jobNum)
        self.lengths = np.zeros(self.jobNum)
        self.types = np.zeros(self.jobNum)
        self.ddl = np.ones(self.jobNum) * 0.25  # 250ms = waitT + exeT
        self.gen_workload(lamda)

        # reset all records
        self.RAN_events = np.zeros((9, self.jobNum))  # random policy
        self.RAN_VM_events = np.zeros((2, self.VMnum))
        self.RR_events = np.zeros((9, self.jobNum))  # round robin policy
        self.RR_VM_events = np.zeros((2, self.VMnum))
        self.early_events = np.zeros((9, self.jobNum))  # earliest policy
        self.early_VM_events = np.zeros((2, self.VMnum))
        self.DQN_events = np.zeros((9, self.jobNum))  # DQN policy
        self.DQN_VM_events = np.zeros((2, self.VMnum))
        self.suit_events = np.zeros((9, self.jobNum))  # suitable policy
        self.suit_VM_events = np.zeros((2, self.VMnum))
        self.sensible_events = np.zeros((9, self.jobNum))  # sensible routing policy
        self.sensible_VM_events = np.zeros((2, self.VMnum))
        self.ad_events = np.zeros((9, self.jobNum))
        self.ad_VM_events = np.zeros((2, self.VMnum))

    def workload(self, job_count):
        arrival_time = self.arrival_Times[job_count-1]
        length = self.lengths[job_count-1]
        jobType = self.types[job_count-1]
        ddl = self.ddl[job_count-1]
        if job_count == self.jobNum:
            finish = True
        else:
            finish = False
        job_attributes = [job_count-1, arrival_time, length, jobType, ddl]
        return finish, job_attributes

    def feedback(self, job_attrs, action, policyID):
        job_id = job_attrs[0]
        arrival_time = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]
        ddl = job_attrs[4]
        # (VMs are not same, speed = 1:2) get actual job length for the specified VM
        if job_type == self.VMtypes[action]:
            real_length = length
        else:
            real_length = length * 2

        if policyID == 1:
            idleT = self.RAN_VM_events[0, action]
        elif policyID == 2:
            idleT = self.RR_VM_events[0, action]
        elif policyID == 3:
            idleT = self.early_VM_events[0, action]
        elif policyID == 4:
            idleT = self.DQN_VM_events[0, action]
        elif policyID == 5:
            idleT = self.suit_VM_events[0, action]
        elif policyID == 6:
            idleT = self.sensible_VM_events[0, action]
        elif policyID == 7:
            idleT = self.ad_VM_events[0, action]

        # waitT & start exeT
        if idleT <= arrival_time:  # if no waitT
            waitT = 0
            startT = arrival_time
        else:
            waitT = idleT - arrival_time
            startT = idleT

        durationT = waitT + real_length  # waitT+exeT
        leaveT = startT + real_length  # leave T
        new_idleT = leaveT  # update VM idle time
        # reward
        reward = - durationT / length
        #my_reward = 2 * length/durationT
        my_reward = length/durationT
        # whether success
        success = 1 if durationT <= ddl else 0

        if policyID == 1:
            self.RAN_events[0, job_id] = action
            self.RAN_events[2, job_id] = waitT
            self.RAN_events[1, job_id] = startT
            self.RAN_events[3, job_id] = durationT
            self.RAN_events[4, job_id] = leaveT
            self.RAN_events[5, job_id] = reward
            self.RAN_events[6, job_id] = real_length
            self.RAN_events[7, job_id] = success
            # update VM info
            self.RAN_VM_events[1, action] += 1
            self.RAN_VM_events[0, action] = new_idleT
            # print('VMC_after:', self.RAN_VM_events[0, action])
        elif policyID == 2:
            self.RR_events[0, job_id] = action
            self.RR_events[2, job_id] = waitT
            self.RR_events[1, job_id] = startT
            self.RR_events[3, job_id] = durationT
            self.RR_events[4, job_id] = leaveT
            self.RR_events[5, job_id] = reward
            self.RR_events[6, job_id] = real_length
            self.RR_events[7, job_id] = success
            # update VM info
            self.RR_VM_events[1, action] += 1
            self.RR_VM_events[0, action] = new_idleT
        elif policyID == 3:
            self.early_events[0, job_id] = action
            self.early_events[2, job_id] = waitT
            self.early_events[1, job_id] = startT
            self.early_events[3, job_id] = durationT
            self.early_events[4, job_id] = leaveT
            self.early_events[5, job_id] = reward
            self.early_events[6, job_id] = real_length
            self.early_events[7, job_id] = success
            # update VM info
            self.early_VM_events[1, action] += 1
            self.early_VM_events[0, action] = new_idleT
        elif policyID == 4:
            self.DQN_events[0, job_id] = action
            self.DQN_events[2, job_id] = waitT
            self.DQN_events[1, job_id] = startT
            self.DQN_events[3, job_id] = durationT
            self.DQN_events[4, job_id] = leaveT
            self.DQN_events[5, job_id] = reward
            self.DQN_events[6, job_id] = real_length
            self.DQN_events[7, job_id] = success
            # update VM info
            self.DQN_VM_events[1, action] += 1
            self.DQN_VM_events[0, action] = new_idleT
        elif policyID == 5:
            self.suit_events[0, job_id] = action
            self.suit_events[2, job_id] = waitT
            self.suit_events[1, job_id] = startT
            self.suit_events[3, job_id] = durationT
            self.suit_events[4, job_id] = leaveT
            self.suit_events[5, job_id] = reward
            self.suit_events[6, job_id] = real_length
            self.suit_events[7, job_id] = success
            # update VM info
            self.suit_VM_events[1, action] += 1
            self.suit_VM_events[0, action] = new_idleT
        elif policyID == 6:
            self.sensible_events[0, job_id] = action
            self.sensible_events[2, job_id] = waitT
            self.sensible_events[1, job_id] = startT
            self.sensible_events[3, job_id] = durationT
            self.sensible_events[4, job_id] = leaveT
            self.sensible_events[5, job_id] = reward
            self.sensible_events[6, job_id] = real_length
            self.sensible_events[7, job_id] = success
            # update VM info
            self.sensible_VM_events[1, action] += 1
            self.sensible_VM_events[0, action] = new_idleT
        elif policyID == 7:
            self.ad_events[0, job_id] = action
            self.ad_events[2, job_id] = waitT
            self.ad_events[1, job_id] = startT
            self.ad_events[3, job_id] = durationT
            self.ad_events[4, job_id] = leaveT
            self.ad_events[5, job_id] = reward
            self.ad_events[6, job_id] = real_length
            self.ad_events[7, job_id] = success
            # update VM info
            self.ad_VM_events[1, action] += 1
            self.ad_VM_events[0, action] = new_idleT
        if policyID == 1:
            return my_reward
        if policyID == 7:
            return my_reward
        return reward

    def feedbackD(self, job_attrs, action):  # reject job
        job_id = job_attrs[0]
        self.DQN_events[2, job_id] = 0  # waitT
        self.DQN_events[1, job_id] = 0  # startT
        self.DQN_events[3, job_id] = 0  # durationT
        self.DQN_events[4, job_id] = 0  # leaveT
        self.DQN_events[5, job_id] = 0  # reward
        self.DQN_events[6, job_id] = 0  # real_length
        self.DQN_events[7, job_id] = 0  # success
        self.DQN_events[8, job_id] = 1  # whether reject
        # update VM info
        self.DQN_events[0, job_id] = action
        self.DQN_VM_events[1, action] += 1

        return self.DQN_events[5, job_id]

    def get_VM_idleT(self, policyID):
        if policyID == 3:
            idleTimes = self.early_VM_events[0, :]
        elif policyID == 4:
            idleTimes = self.DQN_VM_events[0, :]
        elif policyID == 5:
            idleTimes = self.suit_VM_events[0, :]
        elif policyID == 1:
            idleTimes = self.RAN_VM_events[0, :]
        return idleTimes

    def getState(self, job_attrs, policyID):
        arrivalT = job_attrs[1]
        length = job_attrs[2]
        job_type = job_attrs[3]

        # state_job = [length, job_type]
        state_job = [job_type]
        # wait Time
        if policyID ==4:  # DQN
            idleTimes = self.get_VM_idleT(4)
        elif policyID ==5:  # suitable
            idleTimes = self.get_VM_idleT(5)
        elif policyID ==1:
            idleTimes = self.get_VM_idleT(1)
        waitTimes = [t - arrivalT for t in idleTimes]
        waitTimes = np.maximum(waitTimes, 0)
        # print('waitTimes:', waitTimes, '  ', end='')

        state = np.hstack((state_job, waitTimes))  # only consider job length & waitT
        return state

    def getStateP(self, job_id):
        duration = self.sensible_events[3, job_id]
        return duration

    def get_accumulateRewards(self, policies, start, end):
        rewards = np.zeros(policies)

        rewards[0] = sum(self.RAN_events[5, start:end])
        rewards[1] = sum(self.RR_events[5, start:end])
        rewards[2] = sum(self.early_events[5, start:end])
        rewards[3] = sum(self.DQN_events[5, start:end])
        rewards[4] = sum(self.suit_events[5, start:end])
        rewards[5] = sum(self.sensible_events[5, start:end])
        return np.around(rewards, 2)

    def get_FinishTimes(self, policies, start, end):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, start:end])
        finishT[1] = max(self.RR_events[4, start:end])
        finishT[2] = max(self.early_events[4, start:end])
        finishT[3] = max(self.DQN_events[4, start:end])
        finishT[4] = max(self.suit_events[4, start:end])
        finishT[5] = max(self.sensible_events[4, start:end])
        return np.around(finishT, 2)

    def get_executeTs(self, policies, start, end):
        executeTs = np.zeros(policies)
        executeTs[0] = np.mean(self.RAN_events[6, start:end])
        executeTs[1] = np.mean(self.RR_events[6, start:end])
        executeTs[2] = np.mean(self.early_events[6, start:end])
        executeTs[3] = np.mean(self.DQN_events[6, start:end])
        executeTs[4] = np.mean(self.suit_events[6, start:end])
        executeTs[5] = np.mean(self.sensible_events[6, start:end])
        return np.around(executeTs, 3)

    def get_waitTs(self, policies, start, end):
        waitTs = np.zeros(policies)
        waitTs[0] = np.mean(self.RAN_events[2, start:end])
        waitTs[1] = np.mean(self.RR_events[2, start:end])
        waitTs[2] = np.mean(self.early_events[2, start:end])
        waitTs[3] = np.mean(self.DQN_events[2, start:end])
        waitTs[4] = np.mean(self.suit_events[2, start:end])
        waitTs[5] = np.mean(self.sensible_events[2, start:end])
        return np.around(waitTs, 3)

    def get_responseTs(self, policies, start, end):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:end])
        respTs[1] = np.mean(self.RR_events[3, start:end])
        respTs[2] = np.mean(self.early_events[3, start:end])
        respTs[3] = np.mean(self.DQN_events[3, start:end])
        respTs[4] = np.mean(self.suit_events[3, start:end])
        respTs[5] = np.mean(self.sensible_events[3, start:end])
        return np.around(respTs, 3)

    def get_successTimes(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[7, start:end])/(end - start)
        successT[1] = sum(self.RR_events[7, start:end])/(end - start)
        successT[2] = sum(self.early_events[7, start:end])/(end - start)
        successT[3] = sum(self.DQN_events[7, start:end])/(end - start)
        successT[4] = sum(self.suit_events[7, start:end]) / (end - start)
        successT[5] = sum(self.sensible_events[7, start:end]) / (end - start)
        # successT = successT.astype(int)
        successT = np.around(successT, 3)
        return successT

    def get_rejectTimes(self, policies, start, end):
        reject = np.zeros(policies)
        reject[0] = sum(self.RAN_events[8, start:end])
        reject[1] = sum(self.RR_events[8, start:end])
        reject[2] = sum(self.early_events[8, start:end])
        reject[3] = sum(self.DQN_events[8, start:end])
        reject[4] = sum(self.suit_events[8, start:end])
        reject[5] = sum(self.sensible_events[8, start:end])
        return np.around(reject, 2)

    def get_JobDistribution(self, policies):
        distributions = np.zeros((policies, self.VMnum))
        distributions[0, :] = self.RAN_VM_events[1, :]/self.jobNum
        distributions[1, :] = self.RR_VM_events[1, :]/self.jobNum
        distributions[2, :] = self.early_VM_events[1, :]/self.jobNum
        distributions[3, :] = self.DQN_VM_events[1, :]/self.jobNum
        distributions[4, :] = self.suit_VM_events[1, :] / self.jobNum
        distributions[5, :] = self.sensible_VM_events[1, :] / self.jobNum
        return np.around(distributions, 3)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:self.jobNum])
        rewards[1] = sum(self.RR_events[5, start:self.jobNum])
        rewards[2] = sum(self.early_events[5, start:self.jobNum])
        rewards[3] = sum(self.DQN_events[5, start:self.jobNum])
        rewards[4] = sum(self.suit_events[5, start:self.jobNum])
        rewards[5] = sum(self.sensible_events[5, start:self.jobNum])
        return np.around(rewards, 2)

    def get_totalTimes(self, policies, start):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, :]) - self.arrival_Times[start]
        finishT[1] = max(self.RR_events[4, :]) - self.arrival_Times[start]
        finishT[2] = max(self.early_events[4, :]) - self.arrival_Times[start]
        finishT[3] = max(self.DQN_events[4, :]) - self.arrival_Times[start]
        finishT[4] = max(self.suit_events[4, :]) - self.arrival_Times[start]
        finishT[5] = max(self.sensible_events[4, :]) - self.arrival_Times[start]
        return np.around(finishT, 2)

    def get_avgUtilitizationRate(self, policies, start):
        avgRate = np.zeros(policies)  # sum(real_length)/ totalT*VMnum
        avgRate[0] = sum(self.RAN_events[6, start:self.jobNum]) / ((max(self.RAN_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[1] = sum(self.RR_events[6, start:self.jobNum]) / ((max(self.RR_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[2] = sum(self.early_events[6, start:self.jobNum]) / ((max(self.early_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[3] = sum(self.DQN_events[6, start:self.jobNum]) / ((max(self.DQN_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[4] = sum(self.suit_events[6, start:self.jobNum]) / ((max(self.suit_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        avgRate[5] = sum(self.sensible_events[6, start:self.jobNum]) / ((max(self.sensible_events[4, :]) - self.arrival_Times[start]) * self.VMnum)
        return np.around(avgRate, 3)

    def get_all_responseTs(self, policies):
        respTs = np.zeros((policies, self.jobNum))
        respTs[0, :] = self.RAN_events[3, :]
        respTs[1, :] = self.RR_events[3, :]
        respTs[2, :] = self.early_events[3, :]
        respTs[3, :] = self.DQN_events[3, :]
        respTs[4, :] = self.suit_events[3, :]
        respTs[5, :] = self.sensible_events[3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:self.jobNum])
        respTs[1] = np.mean(self.RR_events[3, start:self.jobNum])
        respTs[2] = np.mean(self.early_events[3, start:self.jobNum])
        respTs[3] = np.mean(self.DQN_events[3, start:self.jobNum])
        respTs[4] = np.mean(self.suit_events[3, start:self.jobNum])
        respTs[5] = np.mean(self.sensible_events[3, start:self.jobNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(policies)  # sum(self.RAN_events[7, 3000:-1])/(self.jobNum - 3000)
        successT[0] = sum(self.RAN_events[7, start:self.jobNum])/(self.jobNum - start + 1)
        successT[1] = sum(self.RR_events[7, start:self.jobNum])/(self.jobNum - start + 1)
        successT[2] = sum(self.early_events[7, start:self.jobNum])/(self.jobNum - start + 1)
        successT[3] = sum(self.DQN_events[7, start:self.jobNum])/(self.jobNum - start + 1)
        successT[4] = sum(self.suit_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        successT[5] = sum(self.sensible_events[7, start:self.jobNum]) / (self.jobNum - start + 1)
        return np.around(successT, 3)

 