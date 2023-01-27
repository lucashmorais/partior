#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

if len(sys.argv) <= 2:
    print('Usage: metrics_as_function_of_iterations.py path_to_input_csv path_to_output')
    exit()

input_path = sys.argv[1]
output_path = sys.argv[2]

alg_dict = {}

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

print(alg_dict)

plt.scatter(alg_dict["Firefly"][0], alg_dict["Firefly"][1], s=2, label="Firefly")
plt.scatter(alg_dict["Differential Evolution"][0], alg_dict["Differential Evolution"][1], s=2, label="Differential Evolution")
plt.legend()
plt.yscale('log',base=10)
plt.savefig(f"{output_path}.png", dpi=300)

"""
n, bins, patches = plt.hist(x=d, bins=20, color='#ffb703',
                            alpha=1, rwidth=0.85, zorder=10)

plt.grid(axis='y', alpha=0.75, zorder=-1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Surprise distribution')

plt.savefig(f"{output_path}.png")
"""
