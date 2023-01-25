#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys

assert (len(sys.argv) > 2), 'Usage: histogram.py path_to_input_csv path_to_output'

input_path = sys.argv[1]
output_path = sys.argv[2]

d = np.loadtxt(input_path, skiprows=1, delimiter=',')

n, bins, patches = plt.hist(x=d, bins=20, color='#ffb703',
                            alpha=1, rwidth=0.85, zorder=10)

plt.grid(axis='y', alpha=0.75, zorder=-1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Surprise distribution')

plt.savefig(f"{output_path}.png")
