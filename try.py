from multiprocessing.pool import Pool
from time import sleep


def worker2(id):
    print(f"worker2-id{id}")


def worker1(id):
    print(f"worker1-id{id}")
    for id in range(5):
        pool.apply_async(worker2, args=(id,))


# def task(message):
#     # report a message
#     print(f"Task executing: {message}", flush=True)
#     sleep(1)
#     print(f"Task done: {message}", flush=True)


print("concurrent:")
pool = Pool()

for id in range(5):
    pool.apply_async(worker1, args=(id,))
# pool.apply_async(task, args=("Hello world",))

pool.close()
pool.join()

# # SuperFastPython.com
# # example of issuing a task with apply_async() to the process pool with arguments

# from multiprocessing.pool import Pool

# # task executed in a worker process
# def task(message):
#     # report a message
#     print(f"Task executing: {message}", flush=True)
#     # block for a moment
#     sleep(1)
#     # report a message
#     print(f"Task done: {message}", flush=True)


# # protect the entry point
# if __name__ == "__main__":
#     # create and configure the process pool
#     pool = Pool()
#     # issue tasks to the process pool
#     pool.apply_async(task, args=("Hello world",))
#     # close the process pool
#     pool.close()
#     # wait for all tasks to finish
#     pool.join()
