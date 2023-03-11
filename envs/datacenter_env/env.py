import random
import numpy as np

from typing import List
from envs.datacenter_env.buffer import Buffer
from envs.datacenter_env.job import Job
from envs.datacenter_env.job_generator import JobGenerator
from envs.datacenter_env.machine import Machine


class DatacenterEnv:
    def __init__(self, args) -> None:
        self.args = args
        self.seed = self.args.seed
        self.set_seed()
        self.job_generator = JobGenerator(args)
        self.seq_index = 0  # which example sequence
        self.timeline_job_data = None

    # set random seed
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    # reset datacenter environment
    def reset(self):
        self.pre_time = -1
        self.curr_time = -1
        self.job_id = 0

        self.jobs_record = {}

        self.machines_state_record = []
        self.machines_occupancy_rate_record = []
        self.machines_max_rate_record = []
        self.machines_finish_time_record = []
        self.machines_job_num_record = []
        self.machines_power_record = []
        self.machines_var_record = []
        self.action_record = []

        self.early_stop = False
        self.curr_allocated = False
        self.buffer_full = False

        self.job_num = None
        self.average_workload = None

        # done
        self.done = False

        # reest mode
        # 循环使用一定数量的数据
        if self.args.reset_type == "cycle":
            if self.timeline_job_data is None:
                self.generate_sequence_jobs()
            else:
                self.seq_index = (self.seq_index + 1) % self.args.job_seq_num
        # 每次生成新的一个数据
        elif self.args.reset_type == "new":
            self.generate_sequence_jobs()
            self.seq_index = 0
        # 重复使用一个数据
        elif self.args.reset_type == "repeat":
            if self.timeline_job_data is None:
                self.generate_sequence_jobs()
            self.seq_index = 0
        elif self.args.reset_type == "index":
            if self.timeline_job_data is None:
                self.generate_sequence_jobs()

        # set current jobs
        self.set_curr_timeline_jobs()

        # set block list
        self.block_jobs: List[Job] = []

        # set machines
        self.machine_list: List[Machine] = [
            Machine(self.args, machine_id=machine_id) for machine_id in range(self.args.machine_num)
        ]

        # weekup all container
        for machine in self.machine_list:
            machine.wakeup()

        # set allocate buffer
        self.buffer = Buffer(self.args)

        # 缓冲为空且未终止时候 时间走到下一步
        # put curr time job to buffer
        while self.buffer.empty() and not self.done:
            self.time_proceed_step()

        # curr_job
        self.curr_allocate_jobs: List[Job] = [None] * self.args.job_allocate_num

        for index in range(self.args.job_allocate_num):
            if not self.buffer.empty():
                self.curr_allocate_jobs[index] = self.buffer.get()

        obs = self.observe()

        return obs

    # 生成链接
    def generate_sequence_jobs(self, max_time=None):
        # generate new jobs
        if max_time is not None:
            self.args.max_time = max_time

        self.timeline_job_data = self.job_generator.get_new_job()

    # 设置当前的任务队列
    def set_curr_timeline_jobs(self):
        (
            self.curr_actual_timeline_job_len,
            self.curr_predict_timeline_job_len,
            self.curr_timeline_job_res_req,
        ) = self.timeline_job_data[self.seq_index]

        self.calculate_job_num()
        self.calculate_average_workload()

    # 计算当前任务数量
    def calculate_job_num(self):
        job_num = 0
        for time_len in self.curr_actual_timeline_job_len:
            job_num += len(time_len)

        self.job_num = job_num
        print("job num: ", job_num)

    # 计算当前的平均负载
    def calculate_average_workload(self):
        t_all_res_req = np.zeros(self.args.res_num)
        time_len = len(self.curr_actual_timeline_job_len)
        for t in range(time_len):
            t_job_len = self.curr_actual_timeline_job_len[t]  # n*1
            if t_job_len.shape[0] == 0:
                continue
            t_job_res_req = self.curr_timeline_job_res_req[t]  # n*res_num
            t_res_req = t_job_len.reshape((-1, 1)) * t_job_res_req
            t_all_res_req += np.sum(t_res_req, axis=0)

        average_workload = (
            t_all_res_req
            / float(self.args.res_capacity)
            / float(self.args.machine_num)
            / float(time_len)
        )

        self.average_workload = average_workload

        print("average work load: ", average_workload)

    # 上轮阻塞队列入队
    def put_pre_block_job_to_buffer(self):
        for job in self.block_jobs:
            if not self.buffer.full():
                self.buffer.put(job)
            else:
                print("buffer full !")
                self.buffer_full = True
                self.done = True
        self.block_jobs = []

    # 当前时刻任务入队
    def put_curr_time_job_to_buffer(self):
        # 当前时间到来的任务入队
        curr_actual_time_jobs_len = self.curr_actual_timeline_job_len[self.curr_time]
        curr_predict_time_jobs_len = self.curr_predict_timeline_job_len[self.curr_time]
        curr_time_jobs_res_req = self.curr_timeline_job_res_req[self.curr_time]

        for job_index in range(len(curr_actual_time_jobs_len)):
            job_actual_len = curr_actual_time_jobs_len[job_index]
            job_predict_len = curr_predict_time_jobs_len[job_index]
            job_res_req = curr_time_jobs_res_req[job_index]
            new_job = Job(
                args=self.args,
                job_id=self.job_id,
                res_req=job_res_req,
                job_len=job_actual_len,
                job_predict_len=job_predict_len,
                enter_time=self.curr_time,
            )

            if not self.buffer.full():
                self.buffer.put(new_job)
                self.job_id += 1

                # 记录进入系统内的任务
                self.jobs_record[self.job_id] = new_job
            else:
                print("buffer full !")
                self.buffer_full = True
                self.done = True
                break

        # 记录缓冲占用状态
        self.buffer.record_rate()

    # 获得所有运行完成的任务
    def get_finished_job(self):
        finished_jobs: List[Job] = []
        for vm in self.machine_list:
            finished_jobs.extend(vm.finished_job)
        return finished_jobs

    def get_max_finish_time_by_occupy(self, occupy):
        occupy = np.array(occupy)
        occupy = np.sum(occupy, axis=1)
        state = occupy > 0
        max_finish_time = np.sum(state == True)
        return max_finish_time

    # 获取放置mask
    def get_machines_allocate_mask(self, job):
        machines_allocate_mask = []  # 0表示不放置
        for machine in self.machine_list:
            machines_allocate_mask.append(machine.check_allocate_feasible(job))
        machines_allocate_mask = np.array(machines_allocate_mask, dtype=np.bool8)
        return machines_allocate_mask

    # 观察当前系统状态
    def observe(self):

        if self.args.observe_type == 1:

            job = self.curr_allocate_jobs[0]
            if job == None:
                return None
            job_res_req_rate = job.res_req_rate()
            job_run_time = job.len

            machines_occupancy_rate = []
            machines_run_time = []
            action_mask = []
            for machine in self.machine_list:
                machines_occupancy_rate.append(machine.get_curr_occupancy_rate())
                machines_run_time.append(machine.get_max_finish_time())
                action_mask.append(machine.check_allocate_feasible(job))

            return (
                np.array(job_res_req_rate, dtype=np.float32),
                np.array(job_run_time, dtype=np.int32),
                np.array(machines_occupancy_rate, dtype=np.float32),
                np.array(machines_run_time, dtype=np.int32),
                np.array(action_mask, dtype=np.bool8),
            )
        elif self.args.observe_type == 2:

            job = self.curr_allocate_jobs[0]
            if job == None:
                return None
            job_res_req_rate = job.res_req_rate()
            job_run_time = job.len / self.args.max_job_len  # 归一化

            machines_sample_occupancy_rate = []
            machines_run_time = []
            action_mask = []
            for machine in self.machine_list:
                # 采样5个点
                # t*4
                machine_all_occupancy_rate = machine.get_all_occupancy_rate()
                index = np.linspace(0, len(machine_all_occupancy_rate) - 1, 5, dtype=np.int32)
                machine_sample_occupancy_rate = machine_all_occupancy_rate[index, :]
                machines_sample_occupancy_rate.append(machine_sample_occupancy_rate)
                # 记录时长
                machines_run_time.append(
                    machine.get_max_finish_time() / self.args.max_job_len
                )  # 归一化
                action_mask.append(machine.check_allocate_feasible(job))

            return (
                np.array(job_res_req_rate, dtype=np.float32),
                np.array(job_run_time, dtype=np.float32),
                np.array(machines_sample_occupancy_rate, dtype=np.float32),  # 10*5*4
                np.array(machines_run_time, dtype=np.float32),
                np.array(action_mask, dtype=np.bool8),
            )

        elif self.args.observe_type == 3:
            # job state
            jobs_state = []
            for job in self.curr_allocate_jobs:
                if job == None:
                    jobs_state.append(np.zeros((self.args.timeline_size, self.args.res_num)))
                else:
                    # [time_horizon, num_res]
                    jobs_state.append(job.observe())

            # machines state
            machines_state = []
            action_mask = []
            for machine in self.machine_list:
                # [time_horizon, num_res]
                machines_state.append(machine.observe())
                # get allocate mask
                action_mask.append(machine.check_allocate_feasible(self.curr_allocate_jobs[0]))

            # buffer state
            buffer_state = self.buffer.observe()

            return (
                np.array(jobs_state, dtype=np.float32),
                np.array(machines_state, dtype=np.float32),
                np.array(action_mask, dtype=np.bool8),
                np.array(buffer_state, dtype=np.float32),
            )
        elif self.args.observe_type == 4:
            job = self.curr_allocate_jobs[0]
            if job == None:
                return None
            job_res_req_rate = job.res_req_rate()
            job_run_time = job.len / self.args.max_job_len  # 归一化

            # todo actual or not actual
            machines_all_occupancy_rate = []
            machines_run_time = []
            jobs_num = []
            action_mask = []
            for machine in self.machine_list:
                # 采样10个点
                # t*4
                # machine_all_occupancy_rate = machine.get_all_occupancy_rate()
                if self.args.actual:
                    machine_all_occupancy_rate = machine.get_all_actual_occupancy_rate()
                else:
                    machine_all_occupancy_rate = machine.get_all_predict_occupancy_rate()

                # print(np.sum(machine_all_occupancy_rate - _machine_all_occupancy_rate))
                # print((machine_all_occupancy_rate == _machine_all_occupancy_rate).all())
                machines_all_occupancy_rate.append(machine_all_occupancy_rate)
                # 记录时长
                # machines_run_time.append(
                #     machine.get_max_finish_time() / self.args.max_job_len
                # )  # 归一化
                machines_run_time.append(
                    self.get_max_finish_time_by_occupy(machine_all_occupancy_rate)
                    / self.args.max_job_len
                )  # 归一化
                jobs_num.append(machine.get_running_job_num())
                action_mask.append(machine.check_allocate_feasible(job))
            machines_all_occupancy_rate = np.array(machines_all_occupancy_rate)
            index = np.linspace(0, len(machines_all_occupancy_rate[0]) - 1, 10, dtype=np.int32)
            machines_all_occupancy_rate = machines_all_occupancy_rate[:, index, :]
            return (
                np.array(job_res_req_rate, dtype=np.float32),
                np.array(job_run_time, dtype=np.float32),
                np.array(machines_all_occupancy_rate, dtype=np.float32),  # 10*10*4
                np.array(machines_run_time, dtype=np.float32),
                np.array(jobs_num, dtype=np.float32),
                np.array(action_mask, dtype=np.bool8),
            )
        else:
            NotImplementedError()

    # 获得当前reward
    def get_reward(self):
        reward = 0

        if self.args.reward_type == "machine_run_num":
            run_num = 0
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    run_num += 1
            reward = -run_num
        elif self.args.reward_type == "res_std":
            occupancy_rate = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    occupancy_rate.append(machine.get_curr_occupancy_rate())
            occupancy_rate = np.array(occupancy_rate)
            occupancy_std = np.std(occupancy_rate, axis=0)
            std = np.sum(occupancy_std * np.array(self.args.res_std_weight))
            reward = -std
        elif self.args.reward_type == "res_var":
            occupancy_rate = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    occupancy_rate.append(machine.get_curr_occupancy_rate())
            occupancy_rate = np.array(occupancy_rate)
            occupancy_var = np.var(occupancy_rate, axis=0)
            var = np.sum(occupancy_var * np.array(self.args.res_var_weight))
            reward = 1 / (var + 5e-1)
            if self.curr_allocated:
                reward += 0.5
            else:
                reward -= 0.5
        elif self.args.reward_type == "curr_res_rate":
            occupancy_rate = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    occupancy_rate.append(machine.get_curr_occupancy_rate())
            occupancy_rate = 1 - np.array(occupancy_rate)
            reward = -np.sum(occupancy_rate)
        elif self.args.reward_type == "all_res_rate":
            occupancy_rate = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    occupancy_rate.append(machine.get_all_occupancy_rate())
            occupancy_rate = 1 - np.array(occupancy_rate)
            reward = -np.sum(occupancy_rate)
        elif self.args.reward_type == "job_slowdown":
            for machine in self.machine_list:
                for job in machine.running_jobs:
                    reward -= 1 / float(job.len)
            for job in self.block_jobs:
                reward -= 1 / float(job.len)
        elif self.args.reward_type == "run_time_and_var":
            run_num = 0
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    run_num += 1
            reward = -run_num

            occupancy_rate = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    occupancy_rate.append(machine.get_curr_occupancy_rate())
            occupancy_rate = np.array(occupancy_rate)
            occupancy_std = np.std(occupancy_rate, axis=0)
            std = np.sum(occupancy_std * np.array(self.args.res_std_weight))
            reward = -std

            # 如何合理遍历队列？
            buffer_job_num = self.buffer.qsize()
            for _ in range(buffer_job_num):
                job = self.buffer.get()
                reward -= 1 / float(job.len)
                self.buffer.put(job)
        elif self.args.reward_type == "machine_finish_time":
            finish_time = []
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    finish_time.append(machine.get_max_finish_time())
            reward = -np.sum(finish_time)
        elif self.args.reward_type == "utilization_and_std":
            machine_future_utilization = []
            for machine in self.machine_list:
                machine_future_utilization.append(machine.get_all_occupancy_rate())

            # m*t*res_num
            machine_future_utilization = np.array(machine_future_utilization)[:, :, :]

            # t*res_num
            use_rate = np.mean(machine_future_utilization, axis=0)

            # res_num
            use_rate = np.mean(use_rate, axis=0)

            # t*res_num
            use_std = np.std(machine_future_utilization, axis=0)

            # res_num
            use_std = np.mean(use_std, axis=0)

            # 占用率越大越好，方差越小越好
            # if self.curr_allocated:
            #     reward = np.sum(use_rate - use_std)
            # else:
            #     reward = -np.sum(use_std)
            reward = -np.mean(use_std)
        elif self.args.reward_type == "runtime_and_std":
            runtime_list = []
            machines_occupancy_rate = []
            for machine in self.machine_list:
                runtime_list.append(machine.get_max_finish_time())
                machines_occupancy_rate.append(machine.get_curr_occupancy_rate())
            machines_occupancy_rate = np.array(machines_occupancy_rate)
            machines_occupancy_std = np.std(machines_occupancy_rate, axis=0)
            mean_std = np.mean(machines_occupancy_std)

            runtime = np.mean(runtime_list)
            # runtime std 都是越小越好 负号变为越大越好 表示累积reward越大越好
            reward = -np.array([runtime, mean_std * self.args.res_capacity])
        elif self.args.reward_type == "zero":
            reward = 0
        else:
            return NotImplemented

        # 如果缓冲满了导致done或者早停增加一个很大的负奖励信号
        # if self.buffer_full or self.early_stop:
        #     self.done = True
        #     reward -= 10000

        # reward 缩放
        reward = reward * self.args.reward_scale

        return reward

    # 系统整体时间向前一步
    def time_proceed_step(self):
        # 更新当前时间
        self.curr_time += 1

        # 执行任务，并更新任务状态
        for machine in self.machine_list:
            machine.time_proceed(self.curr_time)

        # 阻塞任务入队
        self.put_pre_block_job_to_buffer()

        if self.curr_time < self.args.max_time:
            # 当前时间任务入队
            self.put_curr_time_job_to_buffer()
        elif self.args.end_mode == "max_time":
            self.done = True
        elif self.args.end_mode == "all_allocate" and self.buffer.empty():
            self.done = True
        elif self.curr_time > self.args.max_end_time:
            print(f"Early Stop ! {self.args.end_mode} {self.buffer.empty()}")
            self.early_stop = True
            self.done = True

        # 记录当前machines状态
        self.record()

    def record(self):
        machines_state = []
        machines_max_rate = []
        machines_finish_time = []
        machines_power = []
        machines_occupancy_rate = []
        machines_job_num = []

        for machine in self.machine_list:
            machines_state.append(machine.state)
            machines_max_rate.append(machine.get_max_occupancy_rate())
            machines_finish_time.append(machine.get_max_finish_time())
            machines_power.append(machine.get_current_power())
            machines_occupancy_rate.append(machine.get_curr_occupancy_rate())
            machines_job_num.append(len(machine.running_jobs))

        self.machines_job_num_record.append(machines_job_num)
        self.machines_state_record.append(np.array(machines_state))
        self.machines_max_rate_record.append(np.array(machines_max_rate))
        self.machines_finish_time_record.append(np.array(machines_finish_time))
        self.machines_power_record.append(np.array(machines_power))
        machines_occupancy_rate = np.array(machines_occupancy_rate)
        self.machines_occupancy_rate_record.append(machines_occupancy_rate)
        # machines_state = np.array(machines_state)
        # machines_occupancy_rate = machines_occupancy_rate[machines_state == 1]
        # 全0 方差也很小
        machines_var = np.sum(
            np.var(machines_occupancy_rate, axis=0) * np.array(self.args.res_var_weight)
        )
        self.machines_var_record.append(machines_var)

    # def step_action(self, action):
    #     reward = 0
    #     for job_index, curr_job in enumerate(self.curr_allocate_jobs):
    #         if curr_job is not None:

    def step(self, action):
        self.action_record.append(action)
        reward = 0
        info = None

        # if self.curr_time == 1250:
        #     print("debug")

        curr_job = self.curr_allocate_jobs[0]
        self.curr_allocate_jobs[0] = None

        # not allocate job
        if action == self.args.machine_num:
            self.block_jobs.append(curr_job)
            self.curr_allocated = False

        # allocate job
        else:
            self.curr_allocated = self.machine_list[action].allocation_job(
                curr_job,
                self.curr_time,
            )
            if not self.curr_allocated:
                self.block_jobs.append(curr_job)

        # 计算reward
        reward = self.get_reward()

        # 缓冲为空且未终止时候 时间走到下一步
        while self.buffer.empty() and not self.done:
            self.time_proceed_step()

        # 缓冲不为空而且插槽为空时候任务出队
        for jop_index, job in enumerate(self.curr_allocate_jobs):
            if job is None and not self.buffer.empty():
                self.curr_allocate_jobs[jop_index] = self.buffer.get()

        # 观察新状态
        obs = self.observe()

        # 获取记录
        info = self.jobs_record
        done = self.done

        return obs, reward, done, info

    # def step_probs(self, jobs_action_prob, greedy=False):
    #     reward = 0
    #     info = None
    #     choose_actions = []
    #     for job_index, curr_job in enumerate(self.curr_allocate_jobs):
    #         if curr_job is not None:
    #             machines_allocate_mask = self.get_machines_allocate_mask(curr_job)
    #             job_action_prob = jobs_action_prob[job_index]
    #             if np.all(machines_allocate_mask == False):
    #                 print("ok")
    #                 pass
    #             job_action_prob[machines_allocate_mask == False] = 0
    #             if greedy:
    #                 action = np.argmax(job_action_prob)
    #             else:
    #                 # TODO
    #                 # 按照概率选择动作
    #                 job_action_prob = job_action_prob / np.sum(job_action_prob)
    #                 action = np.random.choice(
    #                     np.arange(len(job_action_prob)), p=job_action_prob
    #                 )
    #             choose_actions.append(action)
    #             allocated = self.machine_list[action].allocation_job(
    #                 curr_job, self.curr_time
    #             )

    #             # TODO 实验无法放置的情况
    #             # assert allocated == True, "本实验不应该出现无法放置的情况"
    #             if not allocated:
    #                 # 分配失败 进入阻塞队列
    #                 assert allocated == True, "本实验不应该出现无法放置的情况"
    #             #     self.block_jobs.append(curr_job)
    #             # 重置插槽
    #             self.curr_allocate_jobs[job_index] = None

    #         else:
    #             choose_actions.append(-1)

    #     # 计算reward
    #     reward = self.get_reward()

    #     # 缓冲为空且未终止时候 时间走到下一步
    #     while self.buffer.empty() and not self.done:
    #         self.time_proceed_step()

    #     # 缓冲不为空而且插槽为空时候任务出队
    #     for jop_index, job in enumerate(self.curr_allocate_jobs):
    #         if job is None and not self.buffer.empty():
    #             self.curr_allocate_jobs[jop_index] = self.buffer.get()

    #     # 观察新状态
    #     ob = self.observe()

    #     # 获取记录
    #     info = self.jobs_record
    #     done = self.done

    #     return choose_actions, ob, reward, done, info

    # 执行完所有任务
    def run_to_end(self):
        all_done = False
        while not all_done and not self.buffer.full():
            running_job_num = 0
            # running_machine_num = 0
            for machine in self.machine_list:
                if machine.state == Machine.RUN:
                    # running_machine_num += 1
                    running_job_num += len(machine.running_jobs)

            if (
                self.buffer.empty()
                and running_job_num == 0
                # and running_machine_num == 1
            ):
                all_done = True

            # 继续执行
            # print(running_job_num)
            self.time_proceed_step()

    def get_matrix(self, eval_type):
        if eval_type == "compute_time":
            data = np.array(self.machines_max_rate_record)
            data = data > 0
            return np.sum(data)
