# Python file for running hnc TCCW 2 cases
# Zach Johnson
# June 2023

from pandas import read_csv
import numpy as np

mixture_file = "/home/zach/plasma/hnc/data/TCCW_binary_mixture_data.csv"
tccw_mixture_data = read_csv(mixture_file)
print(tccw_mixture_data)

