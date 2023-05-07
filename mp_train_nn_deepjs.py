import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

from config.deepjs import *
from envs.datacenter_env.env import DatacenterEnv

from multiprocessing import Pool, cpu_count

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

    def predict(self, input, action_mask=None, absolute=True):
        predict = self(input)
        if action_mask is not None:
            predict[action_mask == False] += -1e8
        if absolute:
            action = torch.argmax(predict, dim=1).cpu().item()
        else:
            action_probs = torch.softmax(predict, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample().cpu().item()
        return action


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.job_actor = Actor()

    def choose_action(self, obs, absolute=True):
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

        action = self.job_actor.predict(job_input, action_mask, absolute)
        # action = self.job_actor.predict(job_input)
        return action


class JobShopDataset(Dataset):
    def __init__(self, obs_data, action_data, advantage_data) -> None:
        self.obs_data = [i for item in obs_data for i in item]
        self.action_data = [i for item in action_data for i in item]
        self.advantage_data = [i for item in advantage_data for i in item]

    def __getitem__(self, index):
        obs = self.obs_data[index]
        action = self.action_data[index]
        advantage = self.advantage_data[index]
        state, action_mask = self.obs_format(obs)
        return state, action_mask, action, advantage

    def obs_format(self, obs):
        (
            job_res_req_rate,
            job_run_time,
            machines_all_occupancy_rate,
            machines_run_time,
            _,
            action_mask,
        ) = obs
        job_state = torch.tensor(np.array([*job_res_req_rate, job_run_time]), dtype=torch.float)
        machines_all_occupancy_rate = torch.tensor(
            np.array([machines_all_occupancy_rate]), dtype=torch.float
        )
        machines_run_time = torch.tensor(np.array([machines_run_time]), dtype=torch.float)
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
        state = torch.cat(
            (
                job_state,
                machines_occupancy_rate,
                machines_run_time,
                machines_state_mean_std_run_time,
            ),
            dim=-1,
        )  # B*n*dim2
        action_mask = torch.tensor(np.array([action_mask]), dtype=torch.bool)
        return state, action_mask

    def __len__(self):
        return len(self.action_data)


class InputDrive:
    def __init__(self, args) -> None:
        self.args = args
        self.seq_index = 0
        self.seq_num = args.job_seq_num
        self.agent = Agent()
        self.prob = 0.8
        self.prob_step = 2 / self.args.epoch

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    def get_one_experience(self, args, seed, model_state_dict, seq_index, prob=0):
        # 初始化环境
        env = DatacenterEnv(args)
        env.seq_index = seq_index
        env.reset()

        # 初始化agent
        agent = Agent()
        agent.load_state_dict(model_state_dict)

        # 设置随机种子
        self.set_seed(seed)

        # 收集轨迹
        obs = env.reset()
        done = False
        trajectory = []
        agent.eval()
        with torch.no_grad():
            while not done:
                action = agent.choose_action(obs, absolute=False)
                next_obs, reward, done, _ = env.step(action)
                trajectory.append([obs, action, reward, next_obs, done])
                obs = next_obs

        # 收集fitness
        # 计算标准差
        machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
        machines_occupancy_std = np.std(machines_occupancy_rate * args.res_capacity, axis=1)
        machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
        std_fitness = np.mean(machines_occupancy_mean_std)

        # 计算运行时长
        machines_finish_time_record = np.array(env.machines_finish_time_record)
        runtime_fitness = np.mean(machines_finish_time_record)
        fitness = np.array([-runtime_fitness, -std_fitness])

        return trajectory, fitness

    # 计算折扣累积reward
    def get_discount_reward(self, trajectory, reward_index):
        # 统计reward
        reward = []
        for item in trajectory:
            reward.append(item[reward_index])

        # reward 标准化
        # norm_reward_batch = (reward - np.mean(reward, axis=0)) / (np.std(reward, axis=0))

        # 归一化
        # norm_reward_batch = (reward - np.min(reward, axis=0)) / (
        #     np.max(reward, axis=0) - np.min(reward, axis=0)
        # )

        # 目标权重相同
        mean_reward = np.sum(
            np.clip(reward, a_min=[-500, -200], a_max=[0, 0]) / [-500, -200], axis=-1
        )
        # mean_reward = norm_reward_batch[:, 0]

        # mean_reward = np.sum(reward, axis=-1)

        # 计算折扣累积reward
        trajectory_len = len(trajectory)
        discount_reward = np.zeros(trajectory_len)
        for index in reversed(range(trajectory_len - 1)):
            discount_reward[index] = mean_reward[index] + self.args.gamma * mean_reward[index + 1]
        return discount_reward

    # 收集经验
    def get_experience(self, seq_index):
        # 多线程收集经验
        pool = Pool(min(cpu_count(), self.args.experience_num))
        all_record = []
        for seed in range(self.args.experience_num):
            record = pool.apply_async(
                self.get_one_experience,
                args=(
                    self.args,
                    seed,
                    self.agent.state_dict(),
                    seq_index,
                    self.prob,
                ),
            )
            all_record.append(record)
        pool.close()
        pool.join()

        all_trajectory = []
        all_fitness = []

        for record in all_record:
            trajectory, fitness = record.get()
            all_trajectory.append(trajectory)
            all_fitness.append(fitness)

        return all_trajectory, all_fitness

    # 计算baseline
    def get_advantage(self, all_trajectory):
        # 计算累积reward
        all_reward = []
        all_reward_flat = []
        max_reward_len = 0
        for trajectory in all_trajectory:
            max_reward_len = max(max_reward_len, len(trajectory))
            reward = []
            for item in trajectory:
                reward.append(item[2])
                all_reward_flat.append(item[2])
            all_reward.append(reward)
        all_reward_flat = np.array(all_reward_flat)
        reward_mean = np.mean(all_reward_flat, axis=0)
        reward_std = np.std(all_reward_flat, axis=0)

        all_discount_reward = []
        for reward in all_reward:
            # norm_reward = (reward - reward_mean) / (reward_std + 1e-7)
            # mean_reward = np.mean(norm_reward, axis=-1)
            # mean_reward = np.sum(norm_reward * [[0.2, 0.8]], axis=-1)
            # mean_reward = np.sum(norm_reward * [[0.8, 0.2]], axis=-1)
            # mean_reward = np.sum(norm_reward * [[1, 0]], axis=-1)
            # mean_reward = np.sum(norm_reward * [[0, 1]], axis=-1)

            # mean_reward = np.sum(np.array(reward) * np.array([[1 / 600, 1 / 50]]), axis=-1)

            mean_reward = np.sum(
                (np.clip(reward, a_min=[-500, -200], a_max=[0, 0]) - [-500, -200]) / [500, 200],
                axis=-1,
            )
            reward_len = len(reward)
            discount_reward = np.zeros(reward_len)
            for index in reversed(range(reward_len - 1)):
                discount_reward[index] = (
                    mean_reward[index] + self.args.gamma * mean_reward[index + 1]
                )
            all_discount_reward.append(discount_reward)

        # padding
        all_padded_discount_reward = [
            np.concatenate([discount_reward, np.zeros(max_reward_len - len(discount_reward))])
            for discount_reward in all_discount_reward
        ]

        # 计算baseline
        baseline = np.mean(all_padded_discount_reward, axis=0)

        # 计算advantage
        all_advantage = [
            discount_reward - baseline[: len(discount_reward)]
            for discount_reward in all_discount_reward
        ]

        return all_advantage

    def train(self):
        optimizer = optim.AdamW(self.agent.parameters(), lr=self.args.lr)
        best_fitness = [np.array([np.inf, np.inf])] * self.args.job_seq_num
        i_episode = 0
        EP = []
        fitness_list = []
        for epoch in range(self.args.epoch):
            for seq_index in range(self.args.job_seq_num):
                # 收集经验
                all_trajectory, all_fitness = self.get_experience(seq_index)

                all_obs = []
                all_action = []

                for trajectory in all_trajectory:
                    _obs = []
                    _action = []
                    for item in trajectory:
                        _obs.append(item[0])
                        _action.append(item[1])
                    all_obs.append(_obs)
                    all_action.append(_action)

                # 结果汇总
                mean_fitness = -np.mean(all_fitness, axis=0)
                print(f"train epoch {epoch} seq_index {seq_index} i_episode {i_episode}")
                print("mean_fitness: ", mean_fitness)
                # writer.add_scalar(
                #     "current/ws_score",
                #     mean_fitness[0] / 600 + mean_fitness[1] / 50,
                #     i_episode,
                # )
                fitness_list.append(mean_fitness)

                # 记录fitness
                writer.add_scalar("current/duration_score", mean_fitness[0], i_episode)
                writer.add_scalar("current/balance_score", mean_fitness[1], i_episode)

                # 记录 mean fitness
                fitness_mean = np.mean(fitness_list[-args.job_seq_num :], axis=0)
                writer.add_scalar("mean/duration_score", fitness_mean[0], i_episode)
                writer.add_scalar("mean/balance_score", fitness_mean[1], i_episode)

                # 记录最优非支配曲面
                d_n = 0
                remove_list = []
                for item in EP:
                    _, item_fitness = item
                    if np.all(fitness_mean < item_fitness):
                        remove_list.append(item)
                    if np.all(fitness_mean > item_fitness):
                        d_n += 1
                    if d_n != 0:
                        break
                if d_n == 0:
                    for item in remove_list:
                        EP.remove(item)
                    EP.append((i_episode, fitness_mean))

                # 打印曲面
                EP_fitness = np.array([i[1] for i in EP])
                x = EP_fitness[:, 1]
                y = EP_fitness[:, 0]
                figure = plt.figure(figsize=(8, 8), dpi=100)
                plt.scatter(x, y, label="train")
                plt.scatter(16.2658, 534.9209, label="lc")
                # plt.scatter(x, y, lable="rr")
                plt.scatter(66.8868, 349.5121, label="lg")
                plt.scatter(17.0905, 351.4006, label="wsga")
                plt.xlim((0, 250))
                plt.ylim((200, 600))
                plt.xlabel("balance")
                plt.ylabel("duration")
                plt.title("Target distribution")
                plt.legend()
                writer.add_figure("Target distribution", figure, i_episode)
                plt.close()

                # 模型保存
                model_name = (
                    f"e{i_episode}_s{seq_index}_d{mean_fitness[0]:.4f}_b{mean_fitness[1]:.4f}"
                )
                model_save_path = os.path.join(model_save_dir, model_name)
                torch.save(self.agent.job_actor.state_dict(), model_save_path)

                # 计算advantage
                all_advantage = self.get_advantage(all_trajectory)

                # 训练模型
                # 构建dataloader
                dataset = JobShopDataset(
                    obs_data=all_obs,
                    action_data=all_action,
                    advantage_data=all_advantage,
                )

                dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=10)

                # 清空梯度
                optimizer.zero_grad()
                self.agent.train()
                # 梯度累加
                for batch in dataloader:
                    state, action_mask, action, advantage = batch
                    action_predict = self.agent.job_actor(state)
                    # 直接赋值会导致无法梯度回传
                    # TODO 如何把mask用上？
                    action_predict[action_mask == False] += -1e9
                    action_predict = torch.squeeze(action_predict, dim=1)
                    action_probs = torch.softmax(action_predict, dim=-1)
                    action_dist = Categorical(action_probs)
                    action_logprobs = action_dist.log_prob(action)

                    """
                    优化目标是loss越小越好
                    advantage大于0说明该动作好要增大该动作的概率 即减小 -action_logprobs * advantage
                    """
                    loss = -action_logprobs * advantage

                    # 一次梯度回传
                    loss.mean().backward()

                # 梯度更新
                optimizer.step()
                i_episode += 1

            # 更新随机权重
            self.prob = max(self.prob - self.prob_step, self.prob)


if __name__ == "__main__":
    args = parse_args()
    args.method = "ws_deepjs"
    args.tag = "run01"
    save_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # save args
    args_dict = args.__dict__
    args_path = os.path.join(save_dir, "args.txt")
    with open(args_path, "w") as f:
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + " : " + str(value) + "\n")
    writer = SummaryWriter(os.path.join(save_dir, "log"))

    inputdrive = InputDrive(args)

    inputdrive.train()
