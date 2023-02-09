#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import pprint
import csv

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
  def __init__(self, surprise, speedup):
    self.surprise = surprise
    self.speedup = speedup

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
            alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges][gen] = None

        alg_dict[name][num_cores][min_parallelism][num_nodes][num_edges][gen] = ExecutionInfo(surprise, speedup)

    #return filter(lambda x: x[12] in ("00GG", "05FT", "66DM")), list(reader))

x = []
y1 = []
y2 = []

for alg in alg_dict.keys():
    for num_cores in alg_dict[alg]:
        for min_parallelism in alg_dict[alg][num_cores]:
            for num_nodes in alg_dict[alg][num_cores][min_parallelism]:
                for num_edges in alg_dict[alg][num_cores][min_parallelism][num_nodes]:
                    for gen in alg_dict[alg][num_cores][min_parallelism][num_nodes][num_edges]:
                        info = alg_dict[alg][num_cores][min_parallelism][num_nodes][num_edges][gen]
                        x.append(gen)
                        y1.append(info.speedup)
                        y2.append(info.surprise)

fig = plt.figure(figsize = (10, 5))
 
plt.plot(x, y1, color ='maroon')
plt.xscale('log')
 
plt.xlabel("Number of generations")
plt.ylabel("Speedup over random task distribution")
plt.title("Speedup as function of number of generations")
plt.savefig(f"{output_path}.png")

