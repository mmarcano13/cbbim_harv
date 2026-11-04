#!/usr/bin/env python3
"""
author: Mariano Marcano
contact: mariano.marcano@gmail.com
date: 2026-10-26
license: BSD 3-Clause
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from models_cls import initiate, reset, bpoints

process_nums = os.cpu_count()
phi_array = np.arange(100, -10, -10)/100 # variation of percentage
                                         # of coffee harvest
switch = False # use reproduction rate constant (False) or
               # variable (True)
phi = np.nan # coffee harvest percentage
b0 = 0 # initial amount of CBBs
mu = 0 # number of CBBs coming from outside in each seasson CBB/tree
if switch:
    params_file = Path('..') / 'data' / 'params_varrepro.csv'
else:
    params_file = Path('..') / 'data' / 'params_constrepro.csv'
initiate_simulations = initiate(switch, phi, b0, mu, params_file)
parameters = initiate_simulations.parameter_values()
parameters = reset(parameters, b0=0, delta_bo=1, max_itr=1E5)
b_diagram = bpoints(parameters)

if __name__ == "__main__":   
    """
    Computation done in parallel for different values of harvesting
    percentage (phi)
    """
    st = time.perf_counter()
    stable_array = []
    unstable_array = []
    stable_lonely = []
    fixed_points = partial(b_diagram.solve, parameters=parameters)
    with Pool(processes=process_nums, maxtasksperchild=1) as pool:
        results = pool.map(fixed_points, phi_array)
        for arr in results:
            if len(arr[0]) == 2:
               stable_array.append(arr[0][:])
               unstable_array.append(arr[1][:])
            else:
               stable_lonely.append(arr[0][:])
    stable_array = np.array(stable_array)
    unstable_array = np.array(unstable_array)
    stable_lonely = np.array(stable_lonely)
    et = (time.perf_counter() - st)
    print('Model parameters:')
    pd.set_option('display.max_colwidth', None)
    print(initiate_simulations.df) # Show the parameters in a table
    print(' ')
    _, (ax1, ax2) = b_diagram.outputs(phi_array, stable_array,
                                      unstable_array, stable_lonely,
                                      et, process_nums)
    plt.show()
