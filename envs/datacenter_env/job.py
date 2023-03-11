import numpy as np


class Job:
    def __init__(self, args, job_id, res_req, job_len, job_predict_len, enter_time) -> None:
        self.args = args
        self.id = job_id
        self.res_req = res_req
        self.len = job_len
        self.predict_len = job_predict_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        self.predict_finish_time = -1

    def observe(self):
        job_state = np.zeros((self.args.timeline_size, self.args.res_num))

        for r in range(self.args.res_num):
            job_state[: self.len, r] = self.res_req[r] / self.args.res_capacity

        return job_state

    def res_req_rate(self):
        return np.array(self.res_req, dtype=np.float32) / self.args.res_capacity
