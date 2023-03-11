import numpy as np

from typing import List
from envs.datacenter_env.job import Job


class Machine:
    SLEEP = 0
    RUN = 1

    def __init__(self, args, machine_id) -> None:
        self.args = args
        self.id = machine_id
        self.state = Machine.SLEEP
        self.sleep_delay = 0

        # resource time-series occupancy
        self.available_res = np.ones((args.timeline_size, args.res_num)) * args.res_capacity

        self.finished_job: List[Job] = []
        self.running_jobs: List[Job] = []
        self.finish_time_log = []
        self.state_log = []
        self.curr_time = -1

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(args.job_color_num), 1, 1 / float(args.job_color_num))

        np.random.shuffle(self.colormap)

        # graphical representation
        self.image_represent = np.zeros((args.timeline_size, args.res_num, args.res_capacity))

    def wakeup(self):
        self.state = Machine.RUN
        self.sleep_delay = self.args.sleep_delay  # 强制运行t分钟才休眠

    def sleep(self):
        self.state = Machine.SLEEP

    # 检查连接是否可以放入
    def check_allocate_feasible(self, job: Job):
        allocated = False
        if self.state == Machine.RUN and job is not None:
            for t in range(
                0,
                min(
                    (self.args.timeline_size - job.len) + 1,
                    self.args.pre_allocate_time,
                ),
            ):
                new_available_res = self.available_res[t : t + job.len, :] - job.res_req
                # resource allocability
                if np.all(new_available_res >= 0):
                    allocated = True
                    break
        return allocated

    # 放置任务
    def allocation_job(self, job: Job, curr_time):
        allocated = False
        assert self.args.timeline_size >= job.len, "timeline should large than job len"
        for t in range(
            0,
            min(
                (self.args.timeline_size - job.len) + 1,
                self.args.pre_allocate_time,
            ),
        ):
            new_available_res = self.available_res[t : t + job.len, :] - job.res_req

            # check resource allocability
            if np.all(new_available_res >= 0):
                allocated = True

                # update resource time-series occupancy
                self.available_res[t : t + job.len, :] = new_available_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len
                job.predict_finish_time = job.start_time + job.predict_len

                self.running_jobs.append(job)

                # update graphocal representation
                if self.args.obs_represent == "image":
                    new_color = None
                    used_color = np.unique(self.image_represent[:])
                    for color in self.colormap:
                        if color not in used_color:
                            new_color = color
                            break
                    assert new_color != None, "job_num_color is not enough to represent running job"

                    start_time = t
                    end_time = t + job.len
                    for res in range(self.args.res_num):
                        for t in range(start_time, end_time):
                            available_resource_index = np.where(
                                self.image_represent[t, res, :] == 0
                            )[0]
                            self.image_represent[
                                t, res, available_resource_index[: job.res_req[res]]
                            ] = new_color
                break
        return allocated

    # 判断当前机器资源是否紧张
    def resource_crisis(self):
        # t分钟最大资源占用都大于rate则认为资源紧张
        occupancy_rate = 1 - (self.available_res[: self.args.crisis_time] / self.args.res_capacity)
        occupancy_rate = np.max(occupancy_rate, axis=-1) > self.args.crisis_rate
        crisis = np.sum(occupancy_rate) == self.args.crisis_time
        return crisis

    # 判断当前机器资源是否闲置
    def resource_idle(self):
        # 无占用且sleep_delay等于0
        idle = np.all(self.available_res == self.args.res_capacity) and self.sleep_delay == 0
        return idle

    def observe(self):
        if self.args.obs_represent == "image":
            return self.image_represent
        elif self.args.obs_represent == "timeline":
            if self.state == Machine.RUN:
                return (self.args.res_capacity - self.available_res) / self.args.res_capacity
            else:
                return np.ones_like(self.available_res) * -1

    def time_proceed(self, curr_time):
        self.curr_time = curr_time
        if self.state == Machine.RUN:
            # 更新资源容量
            self.available_res[:-1, :] = self.available_res[1:, :]
            self.available_res[-1, :] = self.args.res_capacity

            # 无任务则减少睡眠延迟
            if np.all(self.available_res == self.args.res_capacity):
                if self.sleep_delay > 0:
                    self.sleep_delay -= 1

            # 存在任务则重置睡眠延迟
            else:
                self.sleep_delay = self.args.sleep_delay

            # 更新任务状态
            for job in self.running_jobs:
                if job.finish_time <= curr_time:
                    self.finished_job.append(job)
                    self.running_jobs.remove(job)

            # 更新状态表示
            if self.args.obs_represent == "image":
                self.image_represent[:-1, :, :] = self.image_represent[1:, :, :]
                self.image_represent[-1, :, :] = 0

        # 记录剩余时间
        self.finish_time_log.append(self.get_max_finish_time())
        self.state_log.append(self.state)

    # 获得当前最大剩余时间
    def get_max_finish_time(self):
        state = self.available_res != self.args.res_capacity
        max_finish_time = np.sum(np.sum(state, axis=1) != 0)
        return max_finish_time

    # 获得当前最大资源占用率
    def get_max_occupancy_rate(self):
        current_occupancy_rate = 1 - (self.available_res[0] / self.args.res_capacity)

        max_occupancy_rate = np.max(current_occupancy_rate)
        return max_occupancy_rate

    # 获得当前资源占用率
    def get_curr_occupancy_rate(self):
        current_occupancy_rate = 1 - (self.available_res[0] / self.args.res_capacity)
        return current_occupancy_rate

    # 后续资源占用情况
    def get_all_occupancy_rate(self):
        current_occupancy_rate = 1 - (self.available_res / self.args.res_capacity)
        return current_occupancy_rate

    # 获得预估占用情况
    def get_all_predict_occupancy_rate(self):
        current_occupancy = np.zeros_like(self.available_res)
        for job in self.running_jobs:
            predict_duration = job.predict_finish_time - self.curr_time
            predict_duration = max(0, predict_duration)
            current_occupancy[:predict_duration] += job.res_req
        current_occupancy_rate = current_occupancy / self.args.res_capacity
        return current_occupancy_rate

    # test n
    def get_all_actual_occupancy_rate(self):
        current_occupancy = np.zeros_like(self.available_res)
        for job in self.running_jobs:
            actual_duration = job.finish_time - self.curr_time
            actual_duration = max(0, actual_duration)
            current_occupancy[:actual_duration] += job.res_req
        current_occupancy_rate = current_occupancy / self.args.res_capacity
        return current_occupancy_rate

    # 获得平均资源最大占用率
    def get_mean_max_occupancy_rate(self):
        occupancy_rate = 1 - (self.available_res[: self.args.crisis_time] / self.args.res_capacity)

        mean_max_occupancy_rate = np.mean(np.max(occupancy_rate, axis=0))
        return mean_max_occupancy_rate

    # 获得当前任务数
    def get_running_job_num(self):
        return len(self.running_jobs)

    # 获得当前功率
    def get_current_power(self):
        power = 0
        if self.state == Machine.RUN:
            current_occupancy_rate = 1 - (self.available_res[0] / self.args.res_capacity)
            power = current_occupancy_rate**1.5
            power = np.array(self.args.res_power_weight) * power
            power = np.sum(power) + self.args.base_power
        return power
