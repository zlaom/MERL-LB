import os
import numpy as np
import pandas as pd
from itertools import count
from multiprocessing import Pool, cpu_count

from config.test import *
from envs.datacenter_env.env import DatacenterEnv
from utils import *


class LC:
    def select_action(self, obs):
        _, _, _, _, jobs_num, action_mask = obs
        jobs_num[action_mask == False] = 1e9
        action = np.argmin(jobs_num)
        return action


def test_one_path(args, seq_index, data_save_path, fig_save_path):
    print("start test seq_index: ", seq_index)
    # init agent
    agent = LC()

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

    args.method = "lc"
    args.tag = "user_sigam_test"
    args.actual = False

    user_sigam_list = np.linspace(0, 7.5 * 60 // 3, 10, dtype=np.int32)

    root_dir = os.path.join(
        args.save_path,
        args.method,
        args.tag,
    )

    result = []
    result2 = []
    for user_sigma in user_sigam_list:
        print(f"Test user sigma {user_sigma}")
        save_dir = os.path.join(
            root_dir,
            f"user_sigma_{user_sigma}",
        )
        os.makedirs(save_dir, exist_ok=True)

        fig_save_path = os.path.join(save_dir, "fig")
        data_save_path = os.path.join(save_dir, "data")
        os.makedirs(data_save_path, exist_ok=True)
        os.makedirs(fig_save_path, exist_ok=True)
        os.makedirs(os.path.join(fig_save_path, "use_rate"), exist_ok=True)
        os.makedirs(os.path.join(fig_save_path, "finish_time"), exist_ok=True)

        # save args
        args.user_sigma = user_sigma
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
            result2.append((user_sigma, std_fitness, runtime_fitness))
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
            "user_sigma",
            "balance_fitness_mean",
            "duration_fitness_mean",
            "balance_fitness_std",
            "duration_fitness_std",
        ],
    )
    df.to_csv(os.path.join(root_dir, f"{ args.method}_user_sigma_exp.csv"))
    df2 = pd.DataFrame(
        result2,
        columns=[
            "user_sigma",
            "balance_fitness",
            "duration_fitness",
        ],
    )
    df2.to_csv(os.path.join(root_dir, f"{ args.method}_user_sigma_exp2.csv"))
