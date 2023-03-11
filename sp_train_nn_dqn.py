import os
import random
import numpy as np
import torch
from collections import namedtuple, deque
from itertools import count
from config.dqn import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envs.datacenter_env.env import DatacenterEnv

from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple(
    "Transition",
    (
        "state",
        "action_mask",
        "action",
        "next_state",
        "next_action_mask",
        "reward",
        "done",
    ),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

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


class EpsScheduler:
    def __init__(self, max, mini, step) -> None:
        self.max = max
        self.mini = mini
        self.curr = max
        self.step = (max - mini) / step

    def update(self):
        self.max = self.max - self.step

    @property
    def eps(self):
        return self.max


class DoubelDQN:
    def __init__(self, args) -> None:
        self.args = args
        self.learn_step_counter = 0
        self.action_index = 0

        self.steps_done = 0
        self._build_net()
        self.eps = EpsScheduler(args.eps_start, args.eps_end, args.num_episodes)

    def _build_net(self):
        self.policy_net = Actor().to(device)
        self.target_net = Actor().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(5000)

    def choose_action(self, obs, absolute=False):
        self.steps_done += 1
        state, action_mask = self.obs_format(obs)
        if not absolute and random.random() < self.eps.eps:
            random_prob = torch.rand((1, self.args.machine_num)).to(device)
            random_prob[action_mask == False] += -1e9
            action = torch.argmax(random_prob, dim=-1).cpu().item()
        else:
            predict = self.policy_net(state)
            predict[action_mask == False] += -1e9
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

    def remember(self, obs, action, next_obs, reward, done):
        state, action_mask = self.obs_format(obs)
        if next_obs is None:
            # 避免为None报错 会导致bug吗？
            next_state, next_action_mask = state, action_mask
        else:
            next_state, next_action_mask = self.obs_format(next_obs)

        action = torch.tensor(np.array([[action]]), dtype=torch.int64).to(device)
        reward = torch.tensor(np.array([reward]), dtype=torch.float).to(device)
        done = torch.tensor(np.array([done]), dtype=torch.bool).to(device)

        self.memory.push(
            state,
            action_mask,
            action,
            next_state,
            next_action_mask,
            reward,
            done,
        )

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        if len(self.memory) < self.args.batch_size:
            return

        transitions = self.memory.sample(self.args.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)  # n*2

        # reward 归一化
        reward_batch = (reward_batch - torch.mean(reward_batch, dim=0)) / (
            torch.std(reward_batch, dim=0) + 1e-7
        )

        # 两个目标的均值作为reward
        reward_batch = torch.mean(reward_batch, dim=-1)

        # 单目标 std 或者 运行时长
        # reward_batch = reward_batch[:, 0]

        non_final_mask = torch.cat(batch.done) == False
        non_final_next_states = torch.cat(batch.state)[non_final_mask]
        non_final_next_action_mask = torch.cat(batch.next_action_mask)[non_final_mask]

        # for each batch state according to policy_net
        policy_predict = self.policy_net(state_batch)
        state_action_values = policy_predict.gather(1, action_batch)

        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.args.batch_size, device=device)

        # action mask
        target_predict = self.target_net(non_final_next_states)  # B*10
        target_predict[non_final_next_action_mask == False] = -torch.inf
        next_state_values[non_final_mask] = target_predict.max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, save_path):
        torch.save(self.target_net.state_dict(), save_path + "_target_net.pth")
        torch.save(self.policy_net.state_dict(), save_path + "_policy_net.pth")


if __name__ == "__main__":
    args = parse_args()
    args.method = "dqn"
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
    dqn = DoubelDQN(args)

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
            action = dqn.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            if done:
                print("done")

            # Store the transition in memory
            dqn.remember(obs, action, next_obs, reward, done)

            # Move to the next state
            obs = next_obs

            # Perform one step of the optimization (on the policy network)
            dqn.learn()

            if done:
                dqn.eps.update()
                print("eps: ", dqn.eps.eps)
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

        dqn.save(model_save_path)

        if i_episode % args.target_update == 0:
            dqn.update_target_net()

    print("Complete")
