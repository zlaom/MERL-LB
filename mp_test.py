import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from multiprocessing import Pool, cpu_count
from config.test import *
from envs.datacenter_env.env import DatacenterEnv
from utils import *


class RR:
    def __init__(self, machine_num) -> None:
        self.machine_num = machine_num
        self.action_index = 0

    def select_action(self, obs):
        _, _, _, _, _, action_mask = obs
        action = self.action_index
        for i in range(self.machine_num):
            action = (action + 1) % self.machine_num
            if action_mask[action] == True:
                self.action_index = action
                break
        return action


class RD:
    def __init__(self, machine_num) -> None:
        self.machine_num = machine_num

    def select_action(self, obs):
        _, _, _, _, _, action_mask = obs
        action_prob = np.random.random(self.machine_num)
        action_prob = (action_prob + action_mask) / 2
        action = np.argmax(action_prob)
        return action


class LG:
    def select_action(self, obs):
        _, job_run_time, _, machines_run_time, _, action_mask = obs
        gap = np.abs(machines_run_time - job_run_time)
        gap[action_mask == False] = 1e9
        action = np.argmin(gap)
        return action


class LC:
    def select_action(self, obs):
        _, _, _, _, jobs_num, action_mask = obs
        jobs_num[action_mask == False] = 1e9
        action = np.argmin(jobs_num)
        return action


class Actor(nn.Module):
    def __init__(self, absolute=True, dim_list=[126, 32, 1]):
        super().__init__()
        self.absolute = absolute
        self.dim_list = dim_list
        fc = []
        self.param_num = 0
        for i in range(len(dim_list) - 1):
            fc.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            self.param_num += dim_list[i] * dim_list[i + 1] + dim_list[i + 1]
        self.fc = nn.ModuleList(fc)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        x = torch.squeeze(x, dim=-1)
        return x

    def update(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            start = 0
            for fc in self.fc:
                end = start + fc.in_features * fc.out_features
                fc.weight.data = weights[start:end].reshape(fc.out_features, fc.in_features)
                start = end
                end = start + fc.out_features
                fc.bias.data = weights[start:end]
                start = end

    def predict(self, input, action_mask=None):
        predict = self(input)
        if action_mask is not None:
            predict[action_mask == False] += -1e8
        if not self.absolute:
            action_prob = torch.softmax(predict, dim=-1)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            self.action_logprobs = action_dist.log_prob(action).detach()
            action = action.cpu().item()
        else:
            action = torch.argmax(predict, dim=1).cpu().item()
        return action

    def show(self):
        with torch.no_grad():
            for parameters in self.parameters():
                print(parameters.numpy().flatten())


class Agent(nn.Module):
    def __init__(self, absolute=True):
        super(Agent, self).__init__()
        self.job_actor = Actor(absolute=absolute)

    def update(self, job_weights):
        self.job_actor.update(job_weights)

    def select_action(self, obs):
        (
            job_res_req_rate,
            job_run_time,
            machines_all_occupancy_rate,
            machines_run_time,
            _,
            action_mask,
        ) = obs

        # to tensor
        job_state = torch.tensor(np.array([*job_res_req_rate, job_run_time]), dtype=torch.float)
        machines_all_occupancy_rate = torch.tensor(
            np.array([machines_all_occupancy_rate]), dtype=torch.float
        )
        machines_run_time = torch.tensor(np.array([machines_run_time]), dtype=torch.float)
        action_mask = torch.tensor(np.array([action_mask]), dtype=torch.bool)

        # job_state: B*t*r, machines_state: B*n*t*r, buffer_state: B*t
        B, n, t, r = machines_all_occupancy_rate.shape

        machines_occupancy_rate_mean = torch.mean(machines_all_occupancy_rate, dim=1)  # B*t*r
        machines_occupancy_rate_std = torch.std(machines_all_occupancy_rate, dim=1)  # B*t*r

        job_state = job_state.reshape(B, 1, -1)
        job_state = job_state.repeat(1, n, 1)

        machines_occupancy_rate_mean = machines_occupancy_rate_mean.reshape(B, 1, -1)
        machines_occupancy_rate_std = machines_occupancy_rate_std.reshape(B, 1, -1)
        machines_state_mean = torch.cat(
            (
                machines_occupancy_rate_mean,
                machines_occupancy_rate_std,
            ),
            dim=-1,
        )

        machines_occupancy_rate = machines_all_occupancy_rate.reshape(B, n, -1)
        machines_run_time = machines_run_time.reshape(B, n, -1)
        machines_state_mean_std_run_time = machines_state_mean.repeat(1, n, 1)

        job_input = torch.cat(
            (
                job_state,
                machines_occupancy_rate,
                machines_run_time,
                machines_state_mean_std_run_time,
            ),
            dim=-1,
        )  # B*n*dim2

        action = self.job_actor.predict(job_input, action_mask)
        # action = self.job_actor.predict(job_input)
        return action

    def show(self):
        self.job_actor.show()


def get_agent(args):
    method = args.method
    if method == "rr":
        agent = RR(args.machine_num)
    elif method == "rd":
        agent = RD(args.machine_num)
    elif method == "lg":
        agent = LG()
    elif method == "lc":
        agent = LC()
    elif method in ["nsga", "wsga", "deepjs", "igd"]:
        agent = Agent()
        state_dict = torch.load(args.checkpoint_path)
        agent.job_actor.load_state_dict(state_dict)
    elif method in ["ppo"]:
        agent = Agent()
        # agent = Agent(absolute=False)
        state_dict = torch.load(args.checkpoint_path)
        agent.job_actor.load_state_dict(state_dict)
    return agent


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def test_one_path(args, seq_index, data_save_path, fig_save_path):
    print("start test seq_index: ", seq_index)
    # init agent
    agent = get_agent(args)

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
    args.method = "ppo"
    # args.checkpoint_path = "output/train/deepjs/run02/models/e10001_s1_d497.6165_b14.0890"
    # args.checkpoint_path = "output/train/deepjs/run03/models/e3700_s0_d274.3077_b199.8079"
    # args.checkpoint_path = "output/train/deepjs/run01/models/e19000_s0_d386.8642_b19.4361"
    # args.checkpoint_path = "output/train/deepjs/run01/models/e19850_s0_d275.4718_b194.5685"
    # args.checkpoint_path = "output/train/wsga/run05/elite/g13443_3/24_-326.97737_-13.71405.pth"
    # args.checkpoint_path = "/root/workspace/project/version3/output/train/ppo/run_0/model/e10001_s1_d407.9307_b16.3444_actor.pth"
    # args.checkpoint_path = "output/train/wsga/run05/elite/g13443_3/20_-336.39251_-12.79905.pth"
    # args.checkpoint_path = "output/train/ppo/run_0/model/e16679_s9_d376.1445_b18.8828_actor.pth"
    # args.checkpoint_path = (
    #     "output/train/ns_deepjs/run02_no_mask/models/e13919_s9_d380.7892_b22.2165"
    # )
    # args.max_time = 30 * 60
    # args.job_seq_num = 5
    args.tag = "best_run01"
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

    set_seed()

    # mutil process
    mutil_process = []
    # pool = Pool(cpu_count())
    pool = Pool(10)
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

    result1 = [(*mean_fitness, *std_fitness)]
    df = pd.DataFrame(
        result1,
        columns=[
            "balance_fitness_mean",
            "duration_fitness_mean",
            "balance_fitness_std",
            "duration_fitness_std",
        ],
    )
    df.to_csv(os.path.join(save_dir, f"mean_std.csv"))

    df2 = pd.DataFrame(
        fitness_record,
        columns=[
            "balance_fitness",
            "duration_fitness",
        ],
    )
    df2.to_csv(os.path.join(save_dir, f"all_data.csv"))
    print("done")
