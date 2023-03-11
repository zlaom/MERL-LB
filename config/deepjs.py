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
        default="runtime_and_std",
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
            "runtime_and_std",
        ],
        type=str,
    )

    # input drive
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
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--experience_num", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=100000, type=float)
    parser.add_argument("--observe_type", default=4, type=int)

    args, unknown = parser.parse_known_args()
    return args
