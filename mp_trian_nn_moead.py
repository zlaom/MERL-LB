import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from config.ga import *
from typing import List
from envs.datacenter_env.env import DatacenterEnv

from torch.utils.tensorboard import SummaryWriter


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

    def choose_action(self, obs):
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
        return action

    def show(self):
        self.job_actor.show()


class Individual:
    def __init__(self, job_genes=None):
        self.agent = Agent()
        self.param_num = self.agent.job_actor.param_num
        self.job_genes = job_genes
        self.train_fitness = None
        self.eval_fitness = None
        self.std_fitness = np.inf
        self.steps = 0

    def init(self):
        self.job_genes = np.random.uniform(-1, 1, self.param_num)

    def update(self):
        self.agent.update(self.job_genes.copy())


class Individual:
    def __init__(self, job_genes=None):
        self.agent = Agent()
        self.param_num = self.agent.job_actor.param_num
        self.job_genes = job_genes
        self.train_fitness = None
        self.eval_fitness = None
        self.std_fitness = np.inf
        self.steps = 0

    def init(self):
        self.job_genes = np.random.uniform(-1, 1, self.param_num)

    def update(self):
        self.agent.update(self.job_genes.copy())


def run_individual_in_env(id, args, genes, seq_index):
    env = DatacenterEnv(args)
    env.seq_index = seq_index
    env.reset()

    individual = Individual(genes)
    individual.update()

    obs = env.reset()

    done = False
    action_list = []
    reward_list = []
    while not done:
        action = individual.agent.choose_action(obs)
        obs, reward, done, _ = env.step(action)
        action_list.append(action)
        reward_list.append(reward)

    if args.ga_fitness_type == "std":
        # 计算标准差
        machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
        machines_occupancy_std = np.std(machines_occupancy_rate, axis=1)
        machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
        std_fitness = np.sum(machines_occupancy_mean_std)
        fitness = -std_fitness
    elif args.ga_fitness_type == "runtime":
        # 计算运行时长
        machines_finish_time_record = np.array(env.machines_finish_time_record)
        runtime_fitness = np.sum(machines_finish_time_record / 60)  # 避免过大
        fitness = -runtime_fitness
    elif args.ga_fitness_type == "double":
        # 计算标准差
        machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
        machines_occupancy_std = np.std(machines_occupancy_rate * args.res_capacity, axis=1)
        machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
        std_fitness = np.mean(machines_occupancy_mean_std)
        # 计算运行时长
        machines_finish_time_record = np.array(env.machines_finish_time_record)
        runtime_fitness = np.mean(machines_finish_time_record)  # 避免过大
        fitness = np.array([-runtime_fitness, -std_fitness])

    return id, fitness


class MOEAD:
    def __init__(self) -> None:
        self.population = []
        self.EP = []

        pass

    # 进化
    def evolve(self):
        for id, individual in enumerate(self.population):
            pass
        pass
