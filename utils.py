import matplotlib.pyplot as plt
import numpy as np


def plot_mutil_lines_chart(
    input,
    legend_list=None,
    max_line=np.inf,
    save_name="test.png",
    xlabel="",
    ylabel="",
    title=""
):
    data = np.array(input)
    data = data.T
    plt.figure(figsize=(12, 6))
    for y in data:
        if max_line > 0:
            x = range(y.size)
            plt.plot(x, y)
            max_line -= 1
    if legend_list is not None:
        plt.legend(legend_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_name)
    plt.close()
    # plt.show()


def plot_sum_lines_chart(data):
    data = np.array(data)
    y = np.sum(data, axis=-1)
    plt.figure(figsize=(12, 6))
    x = range(y.size)
    plt.plot(x, y)
    plt.show()


def plot_mean_lines_chart(data):
    data = np.array(data)
    y = np.mean(data, axis=-1)
    plt.figure(figsize=(12, 6))
    x = range(y.size)
    plt.plot(x, y)
    plt.show()
