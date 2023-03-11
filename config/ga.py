import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--reset_type",
        default="index",
        choices=["new", "repeat", "cycle", "index"],
        type=str,
    )
    parser.add_argument("--job_seq_num", default=10, type=int)
    parser.add_argument("--actual", default=True, type=bool)
    parser.add_argument("--user_sigma", default=10 * 60 // 6 // 3, type=int)
    parser.add_argument("--max_time", default=10 * 60, type=int)
    parser.add_argument("--max_end_time", default=60 * 60, type=int)
    parser.add_argument("--max_job_num", default=5, type=int)
    parser.add_argument("--max_res_req", default=10, type=int)
    parser.add_argument("--max_job_len", default=10 * 60, type=int)
    parser.add_argument("--job_allocate_num", default=1, type=int)
    # parser.add_argument("--job_small_rate", default=0.6, type=float)
    # parser.add_argument(
    #     "--level_job_num",
    #     # default=[5, 5, 20, 20, 10, 10, 20, 20, 5, 5],
    #     # default=[10, 10, 8, 4, 2, 10, 10, 8, 4, 2],
    #     default=[2, 4, 8, 8, 6, 4, 6, 8, 8, 4, 2],
    #     type=list,
    # )
    # parser.add_argument(
    #     "--level_job_long_rate",
    #     default=[0.9, 0.6, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9],
    #     type=list,
    # )
    parser.add_argument("--sleep_delay", default=3, type=int)
    parser.add_argument("--pre_allocate_time", default=1, type=int)
    parser.add_argument("--crisis_rate", default=0.8, type=float)
    parser.add_argument("--max_crisis_rate", default=0.8, type=float)
    parser.add_argument("--crisis_time", default=5, type=int)
    parser.add_argument("--res_num", default=4, type=int)
    parser.add_argument("--base_power", default=0.5, type=float)
    parser.add_argument("--res_capacity", default=500, type=int)
    parser.add_argument("--machine_num", default=10, type=int)
    parser.add_argument("--max_expand_num", default=5, type=int)
    parser.add_argument("--res_power_weight", default=[0.25, 0.25, 0.25, 0.25], type=list)
    parser.add_argument("--res_std_weight", default=[0.25, 0.25, 0.25, 0.25], type=list)
    parser.add_argument("--res_var_weight", default=[0.25, 0.25, 0.25, 0.25], type=list)
    parser.add_argument("--buffer_size", default=2000, type=int)
    parser.add_argument("--timeline_size", default=10 * 60, type=int)
    parser.add_argument("--job_color_num", default=40, type=int)
    parser.add_argument(
        "--job_generate",
        default="uniform",
        choices=[
            "uniform",
            "level_uniform",
            "level_bi_model",
        ],
        type=str,
    )
    parser.add_argument(
        "--obs_represent",
        default="timeline",
        choices=["image", "timeline"],
        type=str,
    )
    parser.add_argument(
        "--end_mode",
        default="all_allocate",
        choices=[
            "all_allocate",
            "all_done",
            "max_time",
        ],
        type=str,
    )
    parser.add_argument("--reward_scale", default=1, type=int)
    parser.add_argument(
        "--reward_type",
        default="zero",
        choices=[
            "machine_run_num",
            "machine_power",
            "job_slowdown",
            "curr_res_rate",
            "res_std",
            "res_var",
            "run_time_and_var",
            "utilization_and_std",
            "zero",
        ],
        type=str,
    )

    parser.add_argument(
        "--save_path",
        default="output/train",
        type=str,
    )
    parser.add_argument(
        "--method",
        default="nsga",
        type=str,
    )
    parser.add_argument(
        "--tag",
        default="run03",
        type=str,
    )
    parser.add_argument("--observe_type", default=4, type=int)

    # genetic
    parser.add_argument("--ga_parent_size", default=25, type=int)
    parser.add_argument("--ga_children_size", default=25, type=int)
    parser.add_argument("--ga_mutate_rate", default=0.25, type=float)
    parser.add_argument("--ga_mutate_scale", default=0.05, type=float)
    parser.add_argument("--ga_choice", default="generate", type=str)
    parser.add_argument("--ga_fitness_num", default=2, type=int)
    parser.add_argument("--ga_fitness_wight", default=[0.4, 0.6], type=list)
    parser.add_argument(
        "--ga_fitness_type",
        default="double",
        choices=["std", "runtime", "double"],
        type=str,
    )

    args, unknown = parser.parse_known_args()
    return args
