#!/usr/bin/env python3
"""
author: Mariano Marcano
contact: mariano.marcano@gmail.com
date: 2026-10-26
license: BSD 3-Clause
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models_cls import initiate, reset, model, multiseason, map_1d

def main():
    """
    Computer simulations of a coffee berry borer infestation model
    """
    switch = False # use reproduction rate constant (False) or
                   # variable (True)
    phi = 80 # coffee harvest percentage
    b0 = 0 # initial amount of CBBs
    mu = 0 # number of CBBs coming from outside in each seasson 
           # after the 1st, CBB/tree
    if switch:
        params_file = Path('..') / 'data' / 'params_varrepro.csv'
    else:
        params_file = Path('..') / 'data' / 'params_constrepro.csv'
    initiate_simulations = initiate(switch, phi, b0, mu, params_file)
    parameters = initiate_simulations.parameter_values()
    print(f'Model parameters:')
    pd.set_option('display.max_colwidth', None)
    print(initiate_simulations.df) # show the parameters in a table
    print(' ')

    # Compute fixed points of a one-dimensional map and classify
    # them as stable or unstable
    parameters = reset(parameters, b0=0, delta_bo=1, max_itr=1E5)
    map_simulation = map_1d(parameters)
    g, bo_array, stable_pts, unstable_pts, et = map_simulation.solve()
    map_simulation.printing(g, bo_array, stable_pts, unstable_pts, et)
    fig1, ax = map_simulation.plotting(g, bo_array, stable_pts, unstable_pts)

    # If the map exhibits two stable fixed points with one unstable
    # fixed point in between, then show model simulations for one 
    # season and for multiple seasons
    if unstable_pts.size > 0:
        b0 = round(0.9*unstable_pts[0]) # initial amount of CBB/tree
        parameters = reset(parameters, b0=b0)
        model_simulation = model(parameters)
        sol_bef, sol_aft = model_simulation.solve()
        model_simulation.printing(sol_bef, sol_aft)
        fig2, ax = model_simulation.plotting(sol_bef, sol_aft)
        parameters = reset(parameters, stable_1=stable_pts[0], ny_max=50)
        multi_simulation = multiseason(parameters)
        fig3, ax = multi_simulation.outputs()
    else:
        print('Because there is only one stable fixed point '
              'that leads to\ntotal infestation, we do not show '
              'model simulations.')

    plt.show()

if __name__ == "__main__":
    main()  
else:       
    print(f'Verify that the function ', __name__, 
           'is in top-level code environment.')
