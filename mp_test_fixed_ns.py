import os
import numpy as np
import random
from itertools import count
from multiprocessing import Pool, cpu_count
from copy import deepcopy

from config.test import *
from envs.datacenter_env.env import DatacenterEnv
from utils import *


class NSG:
    def __init__(self, machine_num) -> None:
        self.machine_num = machine_num

    def fast_non_dominated_sort(self, values):
        """
        优化问题一般是求最小值
        :param values: 解集[目标函数1解集,目标函数2解集...]
        :return:返回解的各层分布集合序号。类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]] 其中[1]表示Pareto 最优解对应的序号
        """
        values11 = values[0]  # 函数1解集
        S = [[] for _ in range(0, len(values11))]  # 存放 每个个体支配解的集合。
        front = [[]]  # 存放群体的级别集合,一个级别对应一个[]
        n = [0 for _ in range(0, len(values11))]  # 每个个体被支配解的个数 。即针对每个解,存放有多少好于这个解的个数
        rank = [np.inf for i in range(0, len(values11))]  # 存放每个个体的级别

        for p in range(0, len(values11)):  # 遍历每一个个体
            # ====得到各个个体 的被支配解个数 和支配解集合====
            S[p] = []  # 该个体支配解的集合 。即存放差于该解的解
            n[p] = 0  # 该个体被支配的解的个数初始化为0  即找到有多少好于该解的 解的个数
            for q in range(0, len(values11)):  # 遍历每一个个体
                less = 0  # 的目标函数值小于p个体的目标函数值数目
                equal = 0  # 的目标函数值等于p个体的目标函数值数目
                greater = 0  # 的目标函数值大于p个体的目标函数值数目
                for k in range(len(values)):  # 遍历每一个目标函数
                    if values[k][p] > values[k][q]:  # 目标函数k时,q个体值 小于p个体
                        less = less + 1  # q比p 好
                    if values[k][p] == values[k][q]:  # 目标函数k时,p个体值 等于于q个体
                        equal = equal + 1
                    if values[k][p] < values[k][q]:  # 目标函数k时,q个体值 大于p个体
                        greater = greater + 1  # q比p 差

                if (less + equal == len(values)) and (equal != len(values)):
                    n[p] = n[p] + 1  # q比好,  比p好的个体个数加1

                elif (greater + equal == len(values)) and (equal != len(values)):
                    S[p].append(q)  # q比p差,存放比p差的个体解序号

            # =====找出Pareto 最优解,即n[p]===0 的 个体p序号。=====
            if n[p] == 0:
                rank[p] = 0  # 序号为p的个体,等级为0即最优
                if p not in front[0]:
                    # 如果p不在第0层中
                    # 将其追加到第0层中
                    front[0].append(p)  # 存放Pareto 最优解序号

        # =======划分各层解========
        """
        #示例,假设解的分布情况如下,由上面程序得到 front[0] 存放的是序号1
        个体序号    被支配个数   支配解序号   front
        1           0           2,3,4,5      0
        2           1           3,4,5
        3           1           4,5
        4           3           5
        5           4           0

        #首先 遍历序号1的支配解,将对应支配解[2,3,4,5] ,的被支配个数-1(1-1,1-1,3-1,4-1)
        得到
        表
        个体序号    被支配个数   支配解序号   front
        1          0            2,3,4,5      0
        2          0            3,4,5
        3          0            4,5
        4          2            5
        5          2            0

        #再令 被支配个数==0 的序号 对应的front 等级+1
        得到新表...
        """
        i = 0
        while front[i] != []:  # 如果分层集合为不为空
            Q = []
            for p in front[i]:  # 遍历当前分层集合的各个个体p
                for q in S[p]:  # 遍历p 个体 的每个支配解q
                    n[q] = n[q] - 1  # 则将fk中所有给对应的个体np-1
                    if n[q] == 0:
                        # 如果nq==0
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)  # 存放front=i+1 的个体序号

            i = i + 1  # front 等级+1
            front.append(Q)

        del front[len(front) - 1]  # 删除循环退出 时 i+1产生的[]

        return front  # 返回各层 的解序号集合 # 类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]

    def select_action(self, obs):
        # job_state, machines_state,
        (
            job_res_req_rate,
            job_run_time,
            machines_occupancy_rate,
            machines_run_time,
            _,
            action_mask,
        ) = obs

        # std 越小越好
        action_std = np.ones(self.machine_num) * np.inf
        # action_std = np.ones((10, 4)) * np.inf  # 分别考虑的
        machines_occupancy_rate = machines_occupancy_rate[:, 0, :]
        # 运行时长越小越好
        action_run_time = np.ones(self.machine_num) * np.inf
        # action_run_time_diff = np.ones(self.machine_num) * np.inf

        # 预先放置, 计算一遍可行动作,计算std, 以及剩余运行时长
        for machine_index, mask in enumerate(action_mask):
            # 跳过非法action
            if mask == False:
                continue
            after_machines_occupancy_rate = deepcopy(machines_occupancy_rate)
            after_machines_run_time = deepcopy(machines_run_time)
            after_machines_occupancy_rate[machine_index] += job_res_req_rate
            after_machines_run_time[machine_index] = max(
                after_machines_run_time[machine_index], job_run_time
            )

            # caculate std
            after_std = np.std(after_machines_occupancy_rate, axis=0)  # m*4
            action_std[machine_index] = np.mean(after_std)

            # caculate runtime
            after_run_time = np.mean(after_machines_run_time)  # m*1
            action_run_time[machine_index] = after_run_time
            # action_run_time_diff[machine_index] = abs(
            #     after_machines_run_time[machine_index] - job_run_time
            # )

        # 非支配排序
        # 问题来了 如果希望有个权重如何解决呢？
        action_value = np.concatenate(([action_std], [action_run_time]))
        # action_value = np.concatenate(([action_std], [action_run_time_diff]))
        front = self.fast_non_dominated_sort(action_value)

        # 输出action
        action = random.sample(front[0], 1)[0]
        return action


def test_one_path(args, seq_index, data_save_path, fig_save_path):
    print("start test seq_index: ", seq_index)
    # init agent
    agent = NSG(args.machine_num)

    # init env
    env = DatacenterEnv(args)
    env.seq_index = seq_index

    # start test
    obs = env.reset()
    for _ in count():
        # select and perform an action
        action = agent.select_action(obs)
        # execute action
        next_obs, _, done, _ = env.step(action)
        # move to the next state
        obs = next_obs

        if done:
            break

    # save test result
    # save not run to end data
    machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
    np.save(
        os.path.join(data_save_path, f"occupancy_rate_{seq_index}.npy"),
        machines_occupancy_rate,
    )
    machines_finish_time_record = np.array(env.machines_finish_time_record)
    np.save(
        os.path.join(data_save_path, f"finish_time_{seq_index}.npy"),
        machines_finish_time_record,
    )

    # print mean std and mean run time
    machines_occupancy_std = np.std(machines_occupancy_rate * args.res_capacity, axis=1)
    machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
    std_fitness = np.mean(machines_occupancy_mean_std)
    runtime_fitness = np.mean(machines_finish_time_record)
    print(f"std_fitness {std_fitness} runtime_fitness {runtime_fitness}")

    # save run to end data
    env.run_to_end()
    machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
    np.save(
        os.path.join(data_save_path, f"end_occupancy_rate_{seq_index}.npy"),
        machines_occupancy_rate,
    )
    machines_finish_time_record = np.array(env.machines_finish_time_record)
    np.save(
        os.path.join(data_save_path, f"end_finish_time_{seq_index}.npy"),
        machines_finish_time_record,
    )

    for i in range(4):
        data = machines_occupancy_rate[:, :, i]
        save_name = os.path.join(fig_save_path, "use_rate", f"use_rate_e{seq_index}_{i}.png")
        plot_mutil_lines_chart(
            data,
            save_name=save_name,
            xlabel="time",
            ylabel="utilization",
            title="Container Resource Utilization",
        )

    save_name = os.path.join(fig_save_path, "finish_time", f"finish_time_e{seq_index}.png")
    plot_mutil_lines_chart(
        machines_finish_time_record,
        save_name=save_name,
        xlabel="time",
        ylabel="remaining time",
        title="Container Remaining Time",
    )

    return std_fitness, runtime_fitness, env.job_num


if __name__ == "__main__":
    args = parse_args()
    args.method = "nsg"
    save_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )

    model_save_path = os.path.join(save_dir, "models")
    fig_save_path = os.path.join(save_dir, "fig")
    data_save_path = os.path.join(save_dir, "data")
    os.makedirs(data_save_path, exist_ok=True)
    os.makedirs(os.path.join(fig_save_path, "use_rate"), exist_ok=True)
    os.makedirs(os.path.join(fig_save_path, "finish_time"), exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(fig_save_path, exist_ok=True)

    # mutil process
    mutil_process = []
    pool = Pool(cpu_count())
    for i in range(args.job_seq_num):
        one_process = pool.apply_async(test_one_path, args=(args, i, data_save_path, fig_save_path))
        mutil_process.append(one_process)

    pool.close()
    pool.join()

    # caculate mean performent
    fitness_record = []
    job_num_list = []
    for p in mutil_process:
        std_fitness, runtime_fitness, job_num = p.get()
        job_num_list.append(job_num)
        fitness_record.append((std_fitness, runtime_fitness))
    fitness_record = np.array(fitness_record)
    mean_fitness = np.mean(fitness_record, axis=0)
    std_fitness = np.std(fitness_record, axis=0)
    print(job_num_list)
    np.save(os.path.join(data_save_path, "job_num.npy"), np.array(job_num))
    print(
        "mean std fitness: {:.4f} mean runtime fitness: {:.4f}".format(
            mean_fitness[0], mean_fitness[1]
        )
    )
    print(
        "std std fitness: {:.4f} std runtime fitness: {:.4f}".format(std_fitness[0], std_fitness[1])
    )
    print("done")
