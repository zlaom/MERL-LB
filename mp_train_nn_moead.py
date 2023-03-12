import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from config.moead import *
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


def run_individual_in_env(id1, id2, args, genes, seq_index):
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

    # 计算标准差
    machines_occupancy_rate = np.array(env.machines_occupancy_rate_record)
    machines_occupancy_std = np.std(machines_occupancy_rate * args.res_capacity, axis=1)
    machines_occupancy_mean_std = np.mean(machines_occupancy_std, axis=1)
    std_fitness = np.mean(machines_occupancy_mean_std)
    # 计算运行时长
    machines_finish_time_record = np.array(env.machines_finish_time_record)
    runtime_fitness = np.mean(machines_finish_time_record)  # 避免过大
    fitness = np.array([runtime_fitness, std_fitness])

    return id1, id2, fitness


class MOEAD:
    def __init__(self, args) -> None:
        self.args = args
        self.EP: List[Individual] = []  # 最优曲面
        self.EP_N_ID = []  # 最优曲面
        self.N = args.moead_n  # 权重划分数量
        self.M = args.moead_m  # 目标个数
        self.T = args.moead_t  # 邻居个数
        self.B = []  # 邻居下标 根据权重相似度计算
        self.Z = [0, 0]  # 理想点 最小值就是[0,0]所以理想点为0
        self.population: List[Individual] = []  # 种群
        self.generation = 0
        self.seq_index = 0
        self.seq_num = args.job_seq_num

        # 初始化
        self.set_weight()
        self.get_neighbor()
        self.generate_ancestor()

    def setup_seed(self):
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_weight(self):
        # 划分权重
        self.W = np.zeros((self.N, self.M))
        W = np.linspace(0, 1, self.N)
        self.W[:, 0] = W
        self.W[:, 1] = 1 - W

    def get_neighbor(self):
        # 计算权重的T个邻居
        for i in range(self.N):
            bi = self.W[i]
            distance = np.sum((self.W - bi) ** 2, axis=1)
            neighbor = np.argsort(distance)
            self.B.append(neighbor[1 : self.T + 1])

    def generate_ancestor(self):
        # 初代种群
        for _ in range(self.N):
            individual = Individual()
            individual.init()
            self.population.append(individual)
        self.job_genes_len = individual.param_num

    def crossover(self, c1_genes, c2_genes):
        """Single point crossover."""
        p1_genes = c1_genes.copy()
        p2_genes = c2_genes.copy()

        point = np.random.randint(0, (self.job_genes_len))
        c1_genes[: point + 1] = p2_genes[: point + 1]
        c2_genes[: point + 1] = p1_genes[: point + 1]

    def mutate(self, c_genes):
        """Gaussian mutation with scale"""
        if np.random.random() < self.args.mutate_rate * 2:
            mutation_array = np.random.random(c_genes.shape) < self.args.mutate_rate
            mutation = np.random.normal(size=c_genes.shape)
            mutation[mutation_array] *= self.args.mutate_scale
            c_genes[mutation_array] += mutation[mutation_array]

    # 产生子代
    def generate_children(self, p1: Individual, p2: Individual):
        c1_genes, c2_genes = p1.job_genes.copy(), p2.job_genes.copy()
        self.crossover(c1_genes, c2_genes)
        self.mutate(c1_genes)
        self.mutate(c2_genes)
        c1 = Individual(c1_genes)
        c2 = Individual(c2_genes)
        return c1, c2

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
        all_evaluate_list: list[list[individual]] = []
        for pi in range(self.N):
            Bi = self.B[pi]  # 邻居集合
            # 随机选择邻居进行交叉变异
            k = random.randint(0, len(Bi) - 1)
            l = random.randint(0, len(Bi) - 1)
            ki = Bi[k]
            li = Bi[l]
            xp = self.population[pi]
            xk = self.population[ki]
            xl = self.population[li]
            c1, c2 = self.generate_children(xp, xk)
            c3, c4 = self.generate_children(xk, xl)

            evaluate_list = [xp, xk, xl, c1, c2, c3, c4]
            all_evaluate_list.append(evaluate_list)

        # 评估这些模型
        pool = Pool(cpu_count())
        mutil_process = []
        for id1 in range(self.N):
            for id2, individual in enumerate(all_evaluate_list[id1]):
                # 跳过已经评估过的个体 加速训练
                if individual.train_fitness is not None:
                    continue
                one_process = pool.apply_async(
                    run_individual_in_env,
                    args=(
                        id1,
                        id2,
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
            id1, id2, fitness = one_process.get()
            all_evaluate_list[id1][id2].train_fitness = fitness

        # 根据结果进行迭代
        elite_change_num = 0
        for pi in range(self.N):
            evaluate_list = all_evaluate_list[pi]
            fitness_list = []
            for individual in evaluate_list:
                fitness_list.append(individual.train_fitness)
            fitness_list = np.array(fitness_list)
            tchebycheff_list = fitness_list * self.W[pi]
            # 取最大值作为比较
            tchebycheff_list = np.max(tchebycheff_list, axis=-1).reshape(-1)
            best_i1 = np.argmin(tchebycheff_list[:3])
            best_i2 = np.argmin(tchebycheff_list)
            best_i = best_i2
            # 以一定概率进行详细比较 避免陷入局部最优
            mi = random.randint(0, self.M - 1)
            if random.random() < 0.5:
                if (
                    evaluate_list[best_i1].train_fitness[mi]
                    < evaluate_list[best_i2].train_fitness[mi]
                ):
                    best_i = best_i1

            best_individual = evaluate_list[best_i]

            # # 没有找到更好的解则跳过更新
            # if best_i == 0:
            #     continue
            # self.population[pi] = best_individual

            # 更新邻居
            for nj in self.B[pi]:
                nei_individual = self.population[nj]
                nei_tchebycheff = np.max(np.array(nei_individual.train_fitness) * self.W[pi])
                cur_tchebycheff = np.max(np.array(best_individual.train_fitness) * self.W[pi])
                if cur_tchebycheff < nei_tchebycheff:
                    self.population[nj] = best_individual
                    elite_change_num += 1

            # 更新EP
            if abs(tchebycheff_list[best_i2] - tchebycheff_list[0]) > 1:
                remove_list = []
                n = 0
                for individual in self.EP:
                    if np.all(best_individual.train_fitness < individual.train_fitness):
                        remove_list.append(individual)
                    elif np.all(best_individual.train_fitness > individual.train_fitness):
                        n += 1
                    if n != 0:
                        break
                if n == 0:
                    for individual in remove_list:
                        self.EP.remove(individual)
                    self.EP.append(best_individual)

        # 保存前沿
        self.save_population(self.EP, "elite")
        self.save_population(self.population, "population")

        self.generation += 1
        self.seq_index = (self.seq_index + 1) % self.seq_num
        elite_fitness_list = []
        for individual in self.EP:
            elite_fitness_list.append(individual.train_fitness)
        population_fitness_list = []
        for individual in self.population:
            population_fitness_list.append(individual.train_fitness)
        return elite_change_num, elite_fitness_list, population_fitness_list


if __name__ == "__main__":
    args = parse_args()
    args.method = "moead"
    args.job_seq_num = 1
    args.tag = "run02"

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

    moead = MOEAD(args)
    moead.setup_seed()

    fitness_list = []

    while True:
        print("=" * 100)
        print(f"evolve generation {moead.generation}")
        elite_change_num, elite_fitness_list, population_fitness_list = moead.evolve()

        # log to tensorbord
        writer.add_scalar("Elite change num", elite_change_num, moead.generation)

        elite_fitness_list = np.array(elite_fitness_list)
        y = elite_fitness_list[:, 0]
        x = elite_fitness_list[:, 1]
        figure = plt.figure(figsize=(8, 8), dpi=100)
        plt.scatter(x, y, label="train")
        plt.scatter(16.2658, 534.9209, label="lc")
        plt.scatter(66.8868, 349.5121, label="lg")
        plt.scatter(17.0905, 351.4006, label="wsga")
        plt.xlim((0, 250))
        plt.ylim((200, 600))
        plt.xlabel("balance")
        plt.ylabel("duration")
        plt.title("Elite Target Distribution")
        plt.legend()
        writer.add_figure("Elite Target Distribution", figure, moead.generation)
        plt.close()

        population_fitness_list = np.array(population_fitness_list)
        y = population_fitness_list[:, 0]
        x = population_fitness_list[:, 1]
        figure = plt.figure(figsize=(8, 8), dpi=100)
        plt.scatter(x, y, label="train")
        plt.scatter(16.2658, 534.9209, label="lc")
        plt.scatter(66.8868, 349.5121, label="lg")
        plt.scatter(17.0905, 351.4006, label="wsga")
        plt.xlim((0, 250))
        plt.ylim((200, 600))
        plt.xlabel("balance")
        plt.ylabel("duration")
        plt.title("Population Target Distribution")
        plt.legend()
        writer.add_figure("Population Target Distribution", figure, moead.generation)
        plt.close()

        max_elite_fitness = np.max(elite_fitness_list, axis=0)
        min_elite_fitness = np.min(elite_fitness_list, axis=0)
        writer.add_scalar("Balance fitness max", max_elite_fitness[1], moead.generation)
        writer.add_scalar("Duration fitness max", max_elite_fitness[0], moead.generation)
        writer.add_scalar("Balance fitness min", min_elite_fitness[1], moead.generation)
        writer.add_scalar("Duration fitness min", min_elite_fitness[0], moead.generation)
