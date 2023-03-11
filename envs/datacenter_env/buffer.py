import queue
import numpy as np


class Buffer(queue.Queue):
    def __init__(self, args) -> None:
        super(Buffer, self).__init__(args.buffer_size)
        self.args = args
        self.occupation_rate_record = []

    # 记录占用率
    def record_rate(self):
        self.occupation_rate_record.append(self.qsize() / self.args.buffer_size)

    # 获得缓冲的观测值
    def observe(self):
        buffer_history = np.zeros(self.args.timeline_size)

        # 第一位为当前的实时占用
        buffer_history[0] = self.qsize() / self.args.buffer_size

        r_size = len(self.occupation_rate_record)
        if r_size > self.args.timeline_size - 1:
            r_size = self.args.timeline_size - 1

        buffer_history[1 : r_size + 1] = self.occupation_rate_record[-r_size:]

        return buffer_history
