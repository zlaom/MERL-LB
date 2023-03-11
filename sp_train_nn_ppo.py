import os
import random
import numpy as np
import torch
from collections import namedtuple, deque
from itertools import count
from config.ppo import *

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.datacenter_env.env import DatacenterEnv
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition",
    (
        "state",
        "action_mask",
        "action",
        "action_logprobs",
        "reward",
        "done",
    ),
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, dim_list=[126, 32, 1]):
        super().__init__()
        fc = []
        for i in range(len(dim_list) - 1):
            fc.append(nn.Linear(dim_list[i], dim_list[i + 1]))
        self.fc = nn.ModuleList(fc)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        x = torch.squeeze(x, dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, dim_list=[126, 32, 1]):
        super().__init__()
        fc = []
        for i in range(len(dim_list) - 1):
            fc.append(nn.Linear(dim_list[i], dim_list[i + 1]))
        self.fc = nn.ModuleList(fc)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        x = torch.squeeze(x, dim=-1)
        x = torch.sum(x, dim=-1)
        return x


class PPO:
    def __init__(
        self,
        args,
    ) -> None:
        self.args = args
        self.learn_step_counter = 0
        self.action_logprobs = None  # 缓存
        self._build_net()

    def _build_net(self):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.memory = ReplayMemory(5000)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": args.ppo_actor_lr},
                {"params": self.critic.parameters(), "lr": args.ppo_critic_lr},
            ]
        )
        self.critic_loss = nn.MSELoss()

    def choose_action(self, obs, absolute=False):
        state, action_mask = self.obs_format(obs)
        predict = self.actor(state)
        predict[action_mask == False] += -torch.inf
        if not absolute:
            action_prob = torch.softmax(predict, dim=-1)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            self.action_logprobs = action_dist.log_prob(action).detach()
            action = action.cpu().item()
        else:
            action = torch.argmax(predict, dim=1).cpu().item()
        return action

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
        return state.to(device), action_mask.to(device)

    def remember(self, obs, action, reward, done):
        state, action_mask = self.obs_format(obs)

        action_logprobs = self.action_logprobs
        action = torch.tensor(action, dtype=torch.int32)

        self.memory.push(
            state,
            action_mask,
            action,
            action_logprobs,
            reward,
            done,
        )

    def learn(self):
        if len(self.memory) < self.args.ppo_update_timestep:
            return

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state, dim=0).to(device)
        action_batch = torch.vstack(batch.action).to(device)
        action_mask_batch = torch.cat(batch.action_mask, dim=0).to(device)
        action_logprobs_batch = torch.vstack(batch.action_logprobs).to(device)

        reward_batch = np.array(batch.reward)
        done_batch = np.array(batch.done)

        # reward 标准化
        reward_batch = (reward_batch - np.mean(reward_batch, axis=0)) / (
            np.std(reward_batch, axis=0) + 1e-7
        )

        # reward 缩放
        # reward_batch = reward_batch * np.array([[0.001, 1]])

        # # 归一化
        # norm_reward_batch = (reward_batch - np.min(reward_batch, axis=0)) / (
        #     np.max(reward_batch, axis=0) - np.min(reward_batch, axis=0)
        # )

        # mean_reward_batch = np.mean(norm_reward_batch, axis=-1)

        # 无归一化 或 标准化
        # mean_reward_batch = np.sum(reward_batch, axis=-1)
        # mean_reward_batch = reward_batch[:, 0]

        # Monte Carlo estimate of returns
        # cumulate_rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(
        #     reversed(mean_reward_batch), reversed(done_batch)
        # ):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.args.ppo_gamma * discounted_reward)
        #     cumulate_rewards.insert(0, discounted_reward)

        # cumulate_rewards = torch.tensor(cumulate_rewards, dtype=torch.float32).to(
        #     device
        # )

        # 标准化
        # cumulate_rewards = (cumulate_rewards - cumulate_rewards.mean()) / (
        #     cumulate_rewards.std() + 1e-7
        # )

        cumulate_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward_batch), reversed(done_batch)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.ppo_gamma * discounted_reward)
            cumulate_rewards.insert(0, discounted_reward)

        cumulate_rewards = torch.tensor(cumulate_rewards, dtype=torch.float32).to(device)

        # 标准化
        cumulate_rewards = (cumulate_rewards - cumulate_rewards.mean(dim=0)) / (
            cumulate_rewards.std(dim=0) + 1e-7
        )

        # 合并两个目标的reward
        cumulate_rewards = cumulate_rewards * torch.tensor([[0.5, 0.5]]).to(device)
        cumulate_rewards = torch.sum(cumulate_rewards, dim=-1)
        # cumulate_rewards = cumulate_rewards[:, 0]

        # Optimize policy for K epochs
        for epoch in range(self.args.ppo_epochs):
            new_action_predict = self.actor(state_batch)
            new_action_predict[action_mask_batch == False] += -torch.inf
            new_action_probs = torch.softmax(new_action_predict, dim=-1)
            new_action_dist = Categorical(new_action_probs)
            new_action_entropy = new_action_dist.entropy()
            new_action_logprobs = new_action_dist.log_prob(action_batch.reshape(-1))

            state_values = self.critic(state_batch)

            advantages = cumulate_rewards - state_values.detach()

            ratios = torch.exp(new_action_logprobs - action_logprobs_batch.reshape(-1))

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.args.ppo_eps_clip, 1 + self.args.ppo_eps_clip)
                * advantages
            )

            # loss = -advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.critic_loss(state_values, cumulate_rewards)
                - 0.01 * new_action_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.learn_step_counter += 1

        # TODO Copy new weights into old policy
        # self.policy_old.load_state_dict(self.policy.state_dict()

        # 清空缓冲区
        self.memory.reset()

    def save(self, save_path):
        torch.save(self.actor.state_dict(), save_path + "_actor.pth")
        torch.save(self.critic.state_dict(), save_path + "_critic.pth")


if __name__ == "__main__":
    args = parse_args()
    args.method = "ppo"
    args.tag = "run_0"
    save_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)

    # save args
    args_dict = args.__dict__
    args_path = os.path.join(save_dir, "args.txt")
    with open(args_path, "w") as f:
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + " : " + str(value) + "\n")
    writer = SummaryWriter(os.path.join(save_dir, "log"))

    env = DatacenterEnv(args)
    ppo = PPO(args)

    score_list = []
    fitness_list = []
    EP = []

    for i_episode in range(args.num_episodes):
        print("i_episode: ", i_episode)
        # Initialize the environment and state
        seq_index = i_episode % args.job_seq_num
        env.seq_index = seq_index
        obs = env.reset()
        score = np.zeros(2)
        for t in count():
            # Select and perform an action
            action = ppo.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            if done:
                print("done")

            # Store the transition in memory
            ppo.remember(obs, action, reward, done)

            # Move to the next state
            obs = next_obs

            # Perform one step of the optimization (on the policy network)
            ppo.learn()

            if done:
                ppo.memory.reset()  # 是否需要清除缓冲呢？
                break

        score_list.append(score)

        # 收集fitness
        # 计算标准差
        machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
        machines_occupancy_std = np.std(machines_occupancy_rate * args.res_capacity, axis=1)
        machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
        std_fitness = np.mean(machines_occupancy_mean_std)

        # 计算运行时长
        machines_finish_time_record = np.array(env.machines_finish_time_record)
        runtime_fitness = np.mean(machines_finish_time_record)
        fitness = np.array([runtime_fitness, std_fitness])

        # 记录fitness
        writer.add_scalar("current/duration_score", fitness[0], i_episode)
        writer.add_scalar("current/balance_score", fitness[1], i_episode)

        print("train fitness", fitness)
        fitness_list.append(fitness)
        fitness_mean = np.mean(fitness_list[-args.job_seq_num :], axis=0)
        print("train mean fitness", fitness_mean)

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

        # 记录fitness
        writer.add_scalar("mean/duration_score", fitness_mean[0], i_episode)
        writer.add_scalar("mean/balance_score", fitness_mean[1], i_episode)

        # 保存模型
        model_save_path = os.path.join(
            model_save_dir,
            f"e{i_episode}_s{seq_index}_d{fitness_mean[0]:.4f}_b{fitness_mean[1]:.4f}",
        )
        ppo.save(model_save_path)
    print("Complete")
