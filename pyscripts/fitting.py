#!/usr/bin/env python3
"""
author: Mariano Marcano
contact: mariano.marcano@gmail.com
date: 2026-10-26
license: BSD 3-Clause
"""
import matplotlib.pyplot as plt
from pathlib import Path
from models_cls import initiate
from calibration_cls import estimation

def main():
    """
    Parameter estimation for a coffee berry borer infestation model
    using field data
    """
    switch = False # use reproduction rate constant (False) or
                   # variable (True)
    phi = 0 # coffee harvest percentage
    b0 = 0 # initial amount of borers, one of the parameters to vary
    mu = None
    if switch:
        params_file = Path('..') / 'data' / 'params_varrepro_fixed.csv'
    else:
        params_file = Path('..') / 'data' / 'params_constrepro_fixed.csv'
    initiate_simulations = initiate(switch, phi, b0, mu, params_file)
    parameters = initiate_simulations.parameter_values()
    df_params = initiate_simulations.df # dataframe of the fixed parameters
    
    model_fitting = estimation(parameters)
    model_fitting.data()
    model_fitting.bounds() # set up parameters to estimate and their bounds
    opt = model_fitting.solve()
    model_fitting.evaluate(opt.x)
    model_fitting.printing(opt, df_params)
    fig, ax = model_fitting.plotting()
    plt.show()

if __name__ == "__main__":
    main()  
else:       
    print(f'Verify that the function ', __name__,
           'is in top-level code environment.')
