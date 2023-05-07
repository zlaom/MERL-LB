import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from itertools import count
from multiprocessing import Pool, cpu_count

from config.test import *
from envs.datacenter_env.env import DatacenterEnv
from utils import *


class Actor(nn.Module):
    def __init__(self, dim_list=[126, 32, 1]):
        super().__init__()
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
        return torch.argmax(predict, dim=1).cpu().item()

    def show(self):
        with torch.no_grad():
            for parameters in self.parameters():
                print(parameters.numpy().flatten())


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.job_actor = Actor()

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


def test_one_path(args, seq_index, data_save_path, fig_save_path):
    print("start test seq_index: ", seq_index)
    # checkpoint_path = "output/train/nsga/run02/elite/g3382_0/15_-349.95341_-19.68042.pth"
    # checkpoint_path = "output/one_job/ga/reward_sum/run02_m15/final_population/g_9796_f_-310.773_-0.026/24_f_-308.432_-0.024.pth"
    agent = Agent()
    # state_dict = torch.load("24_f_-342.436_-0.029.pth")
    # agent.load_state_dict(state_dict)

    state_dict = torch.load(args.checkpoint_path)
    agent.job_actor.load_state_dict(state_dict)

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
    machines_job_num_record = np.array(env.machines_job_num_record)
    np.save(
        os.path.join(data_save_path, f"job_num_{seq_index}.npy"),
        machines_job_num_record,
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
    machines_job_num_record = np.array(env.machines_job_num_record)
    np.save(
        os.path.join(data_save_path, f"end_job_num_{seq_index}.npy"),
        machines_job_num_record,
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
    args.job_seq_num = 5
    args.method = "igd"
    args.tag = "user_load_test02"
    args.actual = True
    # args.checkpoint_path = "output/train/nsga/run03/elite/g1_1/20_-501.30449_-25.49838.pth"
    # args.checkpoint_path = "output/train/nsga/run05/elite/g24214_0/10_-351.04309_-20.52227.pth"
    args.checkpoint_path = (
        "output/train/ns_deepjs/run02_no_mask/models/e13919_s9_d380.7892_b22.2165"
    )

    job_num_list = range(2, 10)
    # user_sigam_list = [0]

    root_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )

    result = []
    result2 = []
    for max_job_num in job_num_list:
        user_load_rate = (
            max_job_num
            / 2
            * args.max_res_req
            / 2
            * args.max_job_len
            / 2
            / args.res_capacity
            / args.machine_num
        )
        if user_load_rate > 1.1:
            break
        print(f"Test user_load_rate {user_load_rate:.3f}")
        save_dir = os.path.join(
            root_dir,
            f"user_load_rate_{user_load_rate:.3f}",
        )
        os.makedirs(save_dir, exist_ok=True)

        fig_save_path = os.path.join(save_dir, "fig")
        data_save_path = os.path.join(save_dir, "data")
        os.makedirs(data_save_path, exist_ok=True)
        os.makedirs(fig_save_path, exist_ok=True)
        os.makedirs(os.path.join(fig_save_path, "use_rate"), exist_ok=True)
        os.makedirs(os.path.join(fig_save_path, "finish_time"), exist_ok=True)

        # save args
        args.max_job_num = max_job_num
        args_dict = args.__dict__
        args_path = os.path.join(save_dir, "args.txt")
        with open(args_path, "w") as f:
            for each_arg, value in args_dict.items():
                f.writelines(each_arg + " : " + str(value) + "\n")

        # mutil process
        mutil_process = []
        pool = Pool(cpu_count())
        for i in range(args.job_seq_num):
            one_process = pool.apply_async(
                test_one_path, args=(args, i, data_save_path, fig_save_path)
            )
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
            result2.append((user_load_rate, std_fitness, runtime_fitness))
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
            "std std fitness: {:.4f} std runtime fitness: {:.4f}".format(
                std_fitness[0], std_fitness[1]
            )
        )
        print("done")
    df = pd.DataFrame(
        result,
        columns=[
            "user_load_rate",
            "balance_fitness_mean",
            "duration_fitness_mean",
            "balance_fitness_std",
            "duration_fitness_std",
        ],
    )
    df.to_csv(os.path.join(root_dir, f"{ args.method}_user_load_exp.csv"))
    df2 = pd.DataFrame(
        result2,
        columns=[
            "user_load_rate",
            "balance_fitness",
            "duration_fitness",
        ],
    )
    df2.to_csv(os.path.join(root_dir, f"{ args.method}_user_load_exp2.csv"))
