import numpy as np


class JobGenerator:
    def __init__(self, args) -> None:
        self.args = args
        self.big_job_len_lower = args.max_job_len * 2 / 3
        self.big_job_len_upper = args.max_job_len
        self.small_job_len_lower = 1
        self.small_job_len_upper = args.max_job_len / 5

        self.dominant_res_lower = args.max_res_req / 2
        self.dominant_res_upper = args.max_res_req
        self.other_res_lower = 1
        self.other_res_upper = args.max_res_req / 5

    # 均匀分布
    def uniform_dist(self):
        timeline_job_data = []
        for _ in range(self.args.job_seq_num):
            actual_timeline_job_len = []
            predict_timeline_job_len = []
            timeline_job_res_req = []
            for t in range(self.args.max_time):
                job_num = np.random.randint(0, self.args.max_job_num)
                actual_nj_len = []
                predict_nj_len = []
                nj_res_req = []
                for _ in range(job_num):
                    predict_job_len = np.random.randint(1, self.args.max_job_len + 1, 1)[0]
                    actual_job_len = (
                        np.random.normal(predict_job_len, self.args.user_sigma, 1)
                        .clip(1, self.args.max_job_len)
                        .astype(int)[0]
                    )
                    res_req = np.random.randint(1, self.args.max_res_req + 1, self.args.res_num)
                    actual_nj_len.append(actual_job_len)
                    predict_nj_len.append(predict_job_len)
                    nj_res_req.append(res_req)
                actual_timeline_job_len.append(np.array(actual_nj_len, dtype=np.int64))
                predict_timeline_job_len.append(np.array(predict_nj_len, dtype=np.int64))
                timeline_job_res_req.append(np.array(nj_res_req, dtype=np.int64))
            timeline_job_data.append(
                (actual_timeline_job_len, predict_timeline_job_len, timeline_job_res_req)
            )
        return timeline_job_data

    # 连接数量变动
    def level_uniform_dist(self):
        level_len = len(self.args.level_job_num)
        base = self.args.max_time / level_len
        timeline_job_data = []
        for _ in range(self.args.job_seq_num):
            timeline_job_len = []
            timeline_job_res_req = []
            for t in range(self.args.max_time):
                t_level = self.args.level_job_num[int(t // base)]
                # job_num = np.random.randint(0, t_level)
                job_num = min(np.random.poisson(t_level), self.args.max_job_num)
                nj_len = np.random.randint(1, self.args.max_job_len + 1, job_num)
                nj_res_req = np.random.randint(
                    1, self.args.max_res_req + 1, (job_num, self.args.res_num)
                )

                timeline_job_len.append(nj_len)
                timeline_job_res_req.append(nj_res_req)
            timeline_job_data.append((timeline_job_len, timeline_job_res_req))
        return timeline_job_data

    # small & large job distribution
    def level_bi_model_dist(self):
        level_len = len(self.args.level_job_num)
        base = self.args.max_time / level_len
        timeline_job_data = []
        for _ in range(self.args.job_seq_num):
            timeline_job_len = []
            timeline_job_res_req = []
            for t in range(self.args.max_time):
                t_level = self.args.level_job_num[int(t // base)]
                job_num = min(np.random.poisson(t_level), self.args.max_job_num)

                nj_len = np.zeros(job_num, dtype=np.int32)
                nj_big_index = np.random.random(job_num) > self.args.job_small_rate

                big_nj_len = np.random.randint(
                    self.big_job_len_lower, self.big_job_len_upper + 1, job_num
                )

                small_nj_len = np.random.randint(
                    self.small_job_len_lower, self.small_job_len_upper + 1, job_num
                )

                nj_len[nj_big_index == 1] = big_nj_len[nj_big_index == 1]
                nj_len[nj_big_index == 0] = small_nj_len[nj_big_index == 0]

                nj_res_req = np.zeros((job_num, self.args.res_num), dtype=np.int32)

                nj_dominant_rate = np.random.random((job_num, self.args.res_num))
                max_index = np.max(nj_dominant_rate, axis=-1, keepdims=True)
                nj_dominant_index = nj_dominant_rate == max_index

                nj_dominant_res_req = np.random.randint(
                    self.dominant_res_lower,
                    self.dominant_res_lower + 1,
                    (job_num, self.args.res_num),
                )

                nj_other_res_req = np.random.randint(
                    self.other_res_lower,
                    self.other_res_upper + 1,
                    (job_num, self.args.res_num),
                )

                nj_res_req[nj_dominant_index == True] = nj_dominant_res_req[
                    nj_dominant_index == True
                ]
                nj_res_req[nj_dominant_index == False] = nj_other_res_req[
                    nj_dominant_index == False
                ]

                timeline_job_len.append(nj_len)
                timeline_job_res_req.append(nj_res_req)
            timeline_job_data.append((timeline_job_len, timeline_job_res_req))
        return timeline_job_data

    def get_new_job(self):
        if self.args.job_generate == "uniform":
            return self.uniform_dist()
        if self.args.job_generate == "level_uniform":
            return self.level_uniform_dist()
        elif self.args.job_generate == "level_bi_model":
            return self.level_bi_model_dist()
        else:
            return NotImplemented
