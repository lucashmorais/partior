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

with open(input_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    lines = list(reader)

    for l in lines:
        name = l[0]
        gen = int(l[1])
        surprise = 10 ** (float(l[2]) - 400)
        #surprise = float(l[2])

        if name not in alg_dict:
            alg_dict[name] = [[], []]

        alg_dict[name][0].append(gen)
        alg_dict[name][1].append(surprise)

    #return filter(lambda x: x[12] in ("00GG", "05FT", "66DM")), list(reader))

pprint.pprint(alg_dict.keys())
#pprint.pprint(alg_dict)
fig, ax = plt.subplots(constrained_layout=True)
ax2 = ax.twinx()
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Surprise (lower is better)')
ax2.set_ylabel('Standard deviation')
ax.set_title('Comparing various Surprise optimizers')

for alg in alg_dict.keys():
    #Raw data
    #plt.scatter(alg_dict[alg][0], alg_dict[alg][1], s=2, label=alg)

    num_iterations = max(alg_dict[alg][0])
    x = list(range(num_iterations))
    y = [get_average_from_algorithm_iteration(alg_dict, alg, i) for i in range(num_iterations)]
    std = [get_standard_deviation_from_algorithm_iteration(alg_dict, alg, i) for i in range(num_iterations)]
    ax.scatter(x, y, s=2, label=alg)
    ax2.plot(x, std, linewidth=0.5, label=alg)

    #plt.scatter(x, y, s=2, label=alg)

ax.legend(loc='lower left')
ax.set_yscale('log',base=10)
ax2.set_yscale('log',base=10)
fig.savefig(f"{output_path}.png", dpi=300)

"""
n, bins, patches = plt.hist(x=d, bins=20, color='#ffb703',
                            alpha=1, rwidth=0.85, zorder=10)

plt.grid(axis='y', alpha=0.75, zorder=-1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Surprise distribution')

plt.savefig(f"{output_path}.png")
"""
