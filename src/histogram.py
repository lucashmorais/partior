#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt('../foo.csv', skiprows=1, delimiter=',')

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=d, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Surprise distribution')

plt.savefig("hist.png")
