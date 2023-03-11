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


class GA:
    def __init__(self, args):

        self.args = args
        self.p_size = args.ga_parent_size
        self.c_size = args.ga_children_size
        self.job_genes_len = 0
        self.mutate_rate = args.ga_mutate_rate
        self.mutate_scale = args.ga_mutate_scale
        self.population: List[Individual] = []
        self.elitism_population: List[Individual] = []

        self.avg_fitness = 0
        self.seq_index = 0
        self.seq_num = args.job_seq_num
        self.generation = 0

    def setup_seed(self):
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def generate_ancestor(self):
        for _ in range(self.p_size):
            individual = Individual()
            individual.init()
            self.population.append(individual)
        self.job_genes_len = individual.param_num

    def inherit_ancestor(self):
        """Load genes(nn model parameters) from file."""
        for i in range(self.p_size):
            pth = os.path.join("model", "all_individual", str(i) + "_nn.pth")
            nn = torch.load(pth)
            genes = []
            with torch.no_grad():
                for parameters in nn.parameters():
                    genes.extend(parameters.numpy().flatten())
            self.population.append(Individual(np.array(genes)))

    def crossover(self, c1_genes, c2_genes):
        """Single point crossover."""
        p1_genes = c1_genes.copy()
        p2_genes = c2_genes.copy()

        point = np.random.randint(0, (self.job_genes_len))
        c1_genes[: point + 1] = p2_genes[: point + 1]
        c2_genes[: point + 1] = p1_genes[: point + 1]

    def mutate(self, c_genes):
        """Gaussian mutation with scale"""
        mutation_array = np.random.random(c_genes.shape) < self.mutate_rate
        mutation = np.random.normal(size=c_genes.shape)
        mutation[mutation_array] *= self.mutate_scale
        c_genes[mutation_array] += mutation[mutation_array]

    # def elitism_selection(self):
    #     # 归一化
    #     fitness_list = []
    #     for individual in self.population:
    #         fitness_list.append(individual.train_fitness)
    #     fitness_list = np.array(fitness_list)
    #     norm_fitness_list = (fitness_list - np.min(fitness_list, axis=0)) / (
    #         np.max(fitness_list, axis=0) - np.min(fitness_list, axis=0)
    #     )

    #     # 权重相加排序
    #     norm_fitness_list = np.sum(
    #         norm_fitness_list * self.args.ga_fitness_wight, axis=-1
    #     )
    #     population_sorted_index = np.argsort(norm_fitness_list)  # 升序取后面几位
    #     population_sorted_index = population_sorted_index[-self.p_size :]

    #     self.elitism_population = [
    #         self.population[index] for index in population_sorted_index
    #     ]
    #     self.avg_fitness = np.mean(fitness_list[population_sorted_index], axis=0)
    #     self.elitism_norm_fitness_list = norm_fitness_list[population_sorted_index]

    def elitism_selection(self):
        # 归一化值
        fitness_list = []
        for individual in self.population:
            fitness_list.append(individual.train_fitness)
        fitness_list = np.array(fitness_list)
        norm_fitness_list = (fitness_list - np.min(fitness_list, axis=0)) / (
            np.max(fitness_list, axis=0) - np.min(fitness_list, axis=0)
        )

        # 快速非支配排序越小越好 所以转换为正数
        fm_fitness_list = -np.array(fitness_list).T

        # 快速非支配排序
        front_list = self.fast_non_dominated_sort(fm_fitness_list)

        # 拥挤度计算
        crowded_distance_list = []
        for front in front_list:
            front_values = fm_fitness_list[:, front]
            crowded_distance = self.crowded_distance(front_values)
            crowded_distance_list.append(crowded_distance)

        # 精英选择
        elitism_index = []
        save_best_front = False
        for front, crowded_distance in zip(front_list, crowded_distance_list):
            # 保存最前沿模型
            if not save_best_front:
                best_front_population = []
                for index in front:
                    best_front_population.append(self.population[index])
                self.best_front_population = best_front_population
                save_best_front = True

            # 根据拥挤度排序
            front = np.array(front)
            sorted_index = np.argsort(crowded_distance)  # 升序排序
            sorted_front = front[sorted_index[::-1]]  # 降序排序取拥挤度大的

            # 选择精英
            # 选择的个数是不是可以定义？
            for index in sorted_front:
                if len(elitism_index) < self.p_size:
                    elitism_index.append(index)
                else:
                    break

        # [0.5, 05] 权重相加排序
        norm_fitness_list = np.sum(norm_fitness_list * self.args.ga_fitness_wight, axis=-1)

        elitism_population = [self.population[index] for index in elitism_index]
        # 检查精英变化数量
        elite_change_num = len(elitism_population)
        for elite in elitism_population:
            if elite in self.elitism_population:
                elite_change_num -= 1
        self.elitism_population = elitism_population
        self.fitness_list = fitness_list
        self.avg_fitness = np.mean(fitness_list[elitism_index], axis=0)
        self.elitism_norm_fitness_list = norm_fitness_list[elitism_index]
        return elite_change_num

    # 轮盘赌选择子代
    def roulette_wheel_selection(self, size) -> List[Individual]:
        # 值越大被取到的概率就越大
        selection = []
        wheel = sum(self.elitism_norm_fitness_list)
        for _ in range(size):
            pick = np.random.uniform(0, wheel)
            current = 0
            for i, individual_fitness in enumerate(self.elitism_norm_fitness_list):
                current += individual_fitness
                if current > pick:
                    selection.append(self.elitism_population[i])
                    break
        return selection

    # 随机选择
    def random_select_parent(self, size):
        # 随机选择两个父代
        selection = random.sample(self.elitism_population, size)
        return selection

    # 产生子代
    def generate_children(self):
        children_population = []
        while len(children_population) < self.c_size:
            # p1, p2 = self.roulette_wheel_selection(2)
            p1, p2 = self.random_select_parent(2)
            c1_genes, c2_genes = p1.job_genes.copy(), p2.job_genes.copy()

            self.crossover(c1_genes, c2_genes)
            self.mutate(c1_genes)
            self.mutate(c2_genes)
            c1 = Individual(c1_genes)
            c2 = Individual(c2_genes)
            children_population.extend([c1, c2])
        self.children_population = children_population

    def save_population(self, population: list[Individual], label=""):
        save_dir = os.path.join(
            self.args.save_path,
            self.args.method,
            self.args.tag,
            label,
            f"g{self.generation}_{self.seq_index}",
        )
        os.makedirs(save_dir, exist_ok=True)
        mean_fitness_list = []
        for id, individual in enumerate(population):
            mean_fitness = np.array(individual.train_fitness)
            mean_fitness_list.append([self.generation, id, *mean_fitness.tolist()])
            model_save_path = os.path.join(
                save_dir, "{}_{:.5f}_{:.5f}.pth".format(id, *mean_fitness.tolist())
            )
            individual.update()
            torch.save(individual.agent.job_actor.state_dict(), model_save_path)
        mean_fitness_list = np.array(mean_fitness_list)
        np.save(os.path.join(save_dir, "mean_fitness_record.npy"), mean_fitness_list)
        return mean_fitness_list

    # 进化
    def evolve(self):
        # 普通循环测试
        # population = []
        # for individual in self.population:
        #     individual = run_individual_in_env(
        #         self.args,
        #         individual.job_genes,
        #         self.seq_index,
        #     )
        #     population.append(individual)

        # 多进程
        population_num = self.args.ga_parent_size + self.args.ga_children_size
        pool_num = min(cpu_count(), population_num)
        print(f"use {pool_num} cup core")
        pool = Pool(pool_num)

        mutil_process = []
        for id, individual in enumerate(self.population):
            if individual.train_fitness is not None:
                continue
            # 在坏境中运行个体获得个体适应度
            one_process = pool.apply_async(
                run_individual_in_env,
                args=(
                    id,
                    self.args,
                    individual.job_genes,
                    self.seq_index,
                ),
            )
            mutil_process.append(one_process)

        pool.close()
        pool.join()

        # 收集进程结果
        for one_process in mutil_process:
            id, fitness = one_process.get()
            self.population[id].train_fitness = fitness

        # 保存所有结果
        self.save_population(self.population, "all")

        # 精英选择
        elite_change_num = self.elitism_selection()

        # 保存精英
        elite_fitness_list = self.save_population(self.elitism_population, "elite")

        # 子代生成
        self.generate_children()

        new_population = []
        new_population.extend(self.elitism_population)
        new_population.extend(self.children_population)

        self.population = new_population
        self.seq_index = (self.seq_index + 1) % self.seq_num
        self.generation += 1
        return elite_change_num, elite_fitness_list

    # 值排序
    def sort_by_values(self, values):
        # 升序排序
        sorted_index_list = []
        for value in values:
            sorted_index = np.argsort(value)
            sorted_index_list.append(sorted_index)
        return sorted_index_list

    # 拥挤度计算
    def crowded_distance(self, values):
        distances = []
        sorted_index_list = self.sort_by_values(values)  # 升序排序

        for value, sorted_index in zip(values, sorted_index_list):
            distance = np.ones(len(sorted_index)) * 1e5
            for i in range(1, len(sorted_index) - 1):
                pre_index = sorted_index[i - 1]
                curr_index = sorted_index[i]
                after_index = sorted_index[i + 1]
                distance[curr_index] = (value[after_index] - value[pre_index]) / (
                    max(value) - min(value)
                )
            distances.append(distance)

        distances = np.array(distances)
        distance = np.sum(distances, axis=0)
        return distance

    # 快速非支配排序
    def fast_non_dominated_sort(self, values):
        # 值越小越好
        values11 = values[0]  # 函数1解集
        S = [[] for _ in range(0, len(values11))]  # 存放 每个个体支配解的集合
        front = [[]]  # 存放群体的级别集合,一个级别对应一个[]
        n = [0 for _ in range(0, len(values11))]  # 每个个体被支配解的个数 即针对每个解 存放有多少好于这个解的个数
        rank = [np.inf for _ in range(0, len(values11))]  # 存放每个个体的级别

        # 遍历每一个个体得到各个个体的被支配解个数和支配解集合
        # 目标函数值越小越好
        for p in range(0, len(values11)):
            S[p] = []  # 该个体支配解的集合 即存放差于该解的解
            n[p] = 0  # 该个体被支配的解的个数初始化为0  即找到有多少好于该解
            for q in range(0, len(values11)):  # 遍历每一个个体
                less = 0  # 的目标函数值小于p个体的目标函数值数目
                equal = 0  # 的目标函数值等于p个体的目标函数值数目
                greater = 0  # 的目标函数值大于p个体的目标函数值数目
                for k in range(len(values)):  # 遍历每一个目标函数
                    if values[k][p] > values[k][q]:  # 目标函数k时 q个体值小于p个体
                        less = less + 1  # q比p 好
                    if values[k][p] == values[k][q]:  # 目标函数k时 p个体值等于于q个体
                        equal = equal + 1
                    if values[k][p] < values[k][q]:  # 目标函数k时 q个体值大于p个体
                        greater = greater + 1  # q比p差

                if (less + equal == len(values)) and (equal != len(values)):
                    n[p] = n[p] + 1  # q比好 比p好的个体个数加1

                elif (greater + equal == len(values)) and (equal != len(values)):
                    S[p].append(q)  # q比p差 存放比p差的个体解序号

            # 找出Pareto最优解 即n[p]=0的个体p序号
            if n[p] == 0:
                rank[p] = 0  # 序号为p的个体 等级为0即最优
                if p not in front[0]:
                    # 如果p不在第0层中 将其追加到第0层中
                    front[0].append(p)  # 存放Pareto最优解序号

        # 划分各层解
        i = 0
        while front[i] != []:  # 如果分层集合为不为空
            Q = []
            for p in front[i]:  # 遍历当前分层集合的各个个体p
                for q in S[p]:  # 遍历p个体的每个支配解q
                    n[q] = n[q] - 1  # 则将支配解中所有给对应的个体np-1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)  # 存放front=i+1的个体序号

            i = i + 1  # front等级+1
            front.append(Q)

        del front[len(front) - 1]  # 删除循环退出时i+1产生的[]

        return front  # 返回各层的解序号集合 类似[[1],[9],[0, 8],[7, 6],[3, 5],[2, 4]]


if __name__ == "__main__":
    args = parse_args()
    args.method = "nsga"
    args.job_seq_num = 1
    args.tag = "run05"

    save_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )
    os.makedirs(save_dir, exist_ok=True)

    # save args
    args_dict = args.__dict__
    args_path = os.path.join(save_dir, "args.txt")
    with open(args_path, "w") as f:
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + " : " + str(value) + "\n")
    writer = SummaryWriter(os.path.join(save_dir, "log"))

    ga = GA(args)
    ga.setup_seed()

    if args.ga_choice == "generate":
        ga.generate_ancestor()
    else:
        ga.inherit_ancestor()

    fitness_list = []
    mean_best_fitness = [-np.inf] * args.ga_fitness_num

    while True:
        print("=" * 100)
        print(f"evolve generation {ga.generation}")
        elite_change_num, elite_fitness_list = ga.evolve()

        # log to tensorbord
        writer.add_scalar("Elite change num", elite_change_num, ga.generation)

        elite_fitness_list = np.array(elite_fitness_list)
        elite_fitness_list = -elite_fitness_list[:, -2:]
        y = elite_fitness_list[:, 0]
        x = elite_fitness_list[:, 1]
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
        writer.add_figure("Target distribution", figure, ga.generation)
        plt.close()

        max_elite_fitness = np.max(elite_fitness_list, axis=0)
        min_elite_fitness = np.min(elite_fitness_list, axis=0)
        writer.add_scalar("Balance fitness max", max_elite_fitness[1], ga.generation)
        writer.add_scalar("Duration fitness max", max_elite_fitness[0], ga.generation)
        writer.add_scalar("Balance fitness min", min_elite_fitness[1], ga.generation)
        writer.add_scalar("Duration fitness min", min_elite_fitness[0], ga.generation)
