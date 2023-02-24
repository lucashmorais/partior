#!/usr/bin/python3

from decimal import *

def space_size(num_partitions, num_nodes):
    return num_partitions ** num_nodes

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n > 0

def multi_level_space_size(num_partitions, num_nodes):
    # This ensures that num_partitions is a power of two
    assert is_power_of_two(num_partitions)
    assert is_power_of_two(num_nodes)

    p_target = 2
    multiplier = 1
    sum = 0

    while p_target <= num_partitions:
        sum += multiplier * pow(2, (num_nodes // multiplier)) 
        multiplier *= 2
        p_target *= 2

    return sum

def scientific_notation(v): # Note that v should be a string for eval()
        d = Decimal(v)
        e = format(d, '.6e')
        a = e.split('e')
        b = a[0].replace('0','')
        return b + 'e' + a[1]

for num_partitions in [2, 4, 8, 16, 32]:
    for num_nodes in [32, 64, 128, 256]:
        a = space_size(num_partitions, num_nodes)
        b = multi_level_space_size(num_partitions, num_nodes)

        print(f"Single-step (num_partitions, num_nodes) = ({num_partitions}, {num_nodes}): {scientific_notation(a)}")
        print(f"Multi-level (num_partitions, num_nodes) = ({num_partitions}, {num_nodes}): {scientific_notation(b)}")
        print(f"Space size reduction (num_partitions, num_nodes) = ({num_partitions}, {num_nodes}): {scientific_notation(a // b)}")
        print()
