#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import pprint
import csv
import pdb

if len(sys.argv) <= 2:
    print('Usage: metrics_as_function_of_iterations.py path_to_input_csv path_to_output')
    exit()

input_path = sys.argv[1]
output_path = sys.argv[2]

alg_dict = {}

def average(l):
    return sum(l) / len(l)

def get_average_from_algorithm_iteration(alg_dict, alg, ref_it):
    d = [alg_dict[alg][1][i] for i in range(len(alg_dict[alg][0])) if alg_dict[alg][0][i] == ref_it]
    return average(d)

def get_standard_deviation_from_algorithm_iteration(alg_dict, alg, ref_it):
    d = [alg_dict[alg][1][i] for i in range(len(alg_dict[alg][0])) if alg_dict[alg][0][i] == ref_it]
    return np.std(d)

class ExecutionInfo:
  def __init__(self, surprise, speedup, permanence):
    self.surprise = surprise
    self.speedup = speedup
    self.permanence = permanence

with open(input_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    lines = list(reader)

    for l in lines:
        name = l[0]
        gen = int(l[1])
        surprise = float(l[2])
        type_of_test = l[3]
        speedup = float(l[4])
        num_cores = int(l[5])
        num_nodes = int(l[6])
        num_edges = int(l[7])
        min_parallelism = int(l[8])
        permanence = float(l[9])

        if name not in alg_dict:
            alg_dict[name] = {}

        if num_cores not in alg_dict[name]:
            alg_dict[name][num_cores] = {}

        if min_parallelism not in alg_dict[name][num_cores]:
            alg_dict[name][num_cores][min_parallelism] = {}

        if num_nodes not in alg_dict[name][num_cores][min_parallelism]:
            alg_dict[name][num_cores][min_parallelism][num_nodes] = {}

        if num_edges not in alg_dict[name][num_cores][min_parallelism][num_nodes]:
            alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges] = {}

        if gen not in alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges]:
            alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges][gen] = []

        alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges][gen].append(ExecutionInfo(surprise, speedup, permanence))

    #return filter(lambda x: x[12] in ("00GG", "05FT", "66DM")), list(reader))

x = [[], [], []]
y1 = [[], [], []]
y2 = [[], [], []]
y3 = [[], [], []]

for (i, alg) in enumerate(alg_dict.keys()):
    if (i == 0):
        continue

    print(f"{i}, {alg}")
    for num_cores in alg_dict[alg]:
        for min_parallelism in alg_dict[alg][num_cores]:
            for num_nodes in alg_dict[alg][num_cores][min_parallelism]:
                for num_edges in alg_dict[alg][num_cores][min_parallelism][num_nodes]:
                    for gen in alg_dict[alg][num_cores][min_parallelism][num_nodes][num_edges]:
                        info = alg_dict[alg][num_cores][min_parallelism][num_nodes][num_edges][gen]
                        x[i].append(gen)
                        speedups = list(map(lambda x: x.speedup, info))
                        surprises = list(map(lambda x: x.surprise, info))
                        permanences = list(map(lambda x: x.permanence, info))
                        average_speedup = sum(speedups)/len(speedups)
                        average_surprise = sum(surprises)/len(surprises)
                        average_permanence = sum(permanences)/len(permanences)
                        y1[i].append(average_speedup)
                        y2[i].append(average_surprise)
                        y3[i].append(average_permanence)

def line_graph(x, y1, y2, suffix):
    fig, ax = plt.subplots(constrained_layout=True)
     
    ax.set_xlabel("Number of generations")
    ax.set_ylabel("Speedup over serial execution")
    ax.set_title("Speedup as function of number of generations")

    ax2 = ax.twinx()
    ax2.set_ylabel(suffix)
     
    ax.set_xscale('log')

    colors = ["#219ebc","#ffc533","#126782","#fda01a"]

    for i in range(1, len(x)):
        alg = ""
        if i == 1:
            alg = "Original"
        elif i == 2:
            alg = "Tree"

        #pdb.set_trace()

        ax.plot(x[i], y1[i], color = colors[(i - 1) * 2], label=alg)
        ax2.plot(x[i], y2[i], color = colors[(i - 1) * 2 + 1], label=alg)
        #ax2.plot(x[i], y3[i], color ='yellow')

    ax.legend(loc = 'center left', bbox_to_anchor=(0.0, 0.57))
    ax2.legend(loc = 'center left', bbox_to_anchor=(0.0, 0.43))
    plt.savefig(f"{output_path}_{suffix.lower()}.png")

line_graph(x, y1, y2, 'Surprise')
line_graph(x, y1, y3, 'Permanence')
