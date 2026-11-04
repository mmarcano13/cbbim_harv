#!/usr/bin/env python3
"""
author: Mariano Marcano
contact: mariano.marcano@gmail.com
date: 2026-10-26
license: BSD 3-Clause
"""
import sys
from datetime import date, datetime
import random
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from models_cls import model

class estimation():
    """
    Fit the model equations to the field data by solving
    a least-squares problem
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.x_0 = np.array([])
        self.x_lower = np.array([])
        self.x_upper = np.array([])
        self.var_params = []
        self.points = []
        self.array = []
        self.time = []
        self.n_t = []
        self.data_time = []
        self.df_data = pd.DataFrame()
        self.t_d = []

    def data(self):
        """
        Read field data of CBB infestation for a selection of dates
        in different years. For the data used in Marcano et al 2021
        the dates were the 1st of June, September, October, and November
        of the years 2010, 2011, 2015, and 2018. That can be changed in
        the data file depending the data available.
        Return dataframes of the data (df_data) and the time ticks (t_d)
        to compare model result with data.
        """
        data_file = Path('..') / 'data' / 'data_infestation.csv'
        try:
            self.df_data = pd.read_csv(data_file)
        except FileNotFoundError:
            sys.exit(f'Please download and save the file {data_file}')
        self.df_data.set_index('Year', inplace=True)
        year_ref = self.df_data.index[0]
        date_ref = date(year_ref, 5, 1) # season starting date, May 1st
        for i in range(len(self.df_data.columns)):
            mo = datetime.strptime(self.df_data.columns[i], '%B').month
            self.t_d.append((date(year_ref, mo, 1)-date_ref).days) # time
                                 # tick to compare model result with data

    def bounds(self):
        """
        Set up lower and upper bounds of the variable rate parameters
        and the initial free CBB at the beginning of each year of the
        field data. Select randomly an initial guess within the
        parameter bounds.
        """
        varrate_bnds = {'eps_c':
                        np.array([0.58649999999999991, 0.79349999999999987]),
                        'eps_ci': np.array([0.017, 0.023]),
                        'eps_bo': np.array([0.029325, 0.039675])}
        varb0s_lower = (np.array([45, 270, 90, 225])
                        /self.parameters['kappa_b'])
        varb0s_upper = (np.array([65, 330, 110, 275])
                        /self.parameters['kappa_b'])
        for key, value in varrate_bnds.items():
            self.x_lower = np.append(self.x_lower, value[0])
            self.x_upper = np.append(self.x_upper, value[1])
            self.var_params.append(key)
        self.x_lower = np.append(self.x_lower, varb0s_lower)
        self.x_upper = np.append(self.x_upper, varb0s_upper)
        self.var_params.extend(['b0', 'bo'])
        self.var_params = tuple(self.var_params) # a tuple with the labels
                                              # of the variable parameters
        r = random.random()
        self.x_0 = (1-r)*self.x_lower + r*self.x_upper

    def solve(self):
        """
        Solve the non-linear least-squares problem
        """
        def fun(x):
            """
            Objective function for the non-linear least-squares problem
            read x, an optimum candidate
            return the residual, i.e., the difference between the model
            outcomes and the data points
            """
            residual = []             
            for ind, item in enumerate(self.var_params[:-2]):
                self.parameters[item] = x[ind]
            for i,_ in enumerate(self.df_data.index):
                self.parameters[self.var_params[-2]] = (
                                                x[i+len(self.var_params)-2]
                                               *self.parameters['kappa_b'])
                self.parameters[self.var_params[-1]] = (
                                               x[i+len(self.var_params)-2])
                model_simulation = model(self.parameters)
                sol_bef,_ = model_simulation.solve()
                ci2_bef = self.parameters['cu0']-(sol_bef.y[0,]
                                                  +sol_bef.y[1,])
                for j,_ in enumerate(self.df_data.columns):
                    ind = np.argwhere(sol_bef.t==self.t_d[j])[0]
                    residual.append(                # Residual vector
                      self.df_data.iloc[i].values[j]/100
                      -sol_bef.y[1,ind]-ci2_bef[ind])
            return np.squeeze(np.array(residual))
        return least_squares(fun, self.x_0, bounds=(self.x_lower,
                             self.x_upper))

    def evaluate(self, x_opt):
        """ Evaluate the optimal parameters in the model equations """
        k = 0
        for ind, item in enumerate(self.var_params[:-2]):
            self.parameters[item] = x_opt[ind]
        for i,_ in enumerate(self.df_data.index):
            self.parameters[self.var_params[-2]] = (
                                              x_opt[i+len(self.var_params)-2]
                                              *self.parameters['kappa_b'])
            self.parameters[self.var_params[-1]] = (
                                             x_opt[i+len(self.var_params)-2])
            model_simulation = model(self.parameters)
            sol_bef,_ = model_simulation.solve()
            ci2_bef = self.parameters['cu0']-(sol_bef.y[0,]+sol_bef.y[1,])
            self.array.append(sol_bef.y[1,]+ci2_bef)
            self.time.append(sol_bef.t+k)
            k += sol_bef.t.shape[0] 
            for j,_ in enumerate(self.df_data.columns):
                ind = np.argwhere(sol_bef.t==self.t_d[j])[0]
                self.points.append(sol_bef.y[1,ind]+ci2_bef[ind])
        self.n_t = sol_bef.t.shape[0]

    def printing(self, opt, df_params):
        """ 
        Print relevant information and save the parameter values to
        external files:
        params_constrepro.csv: all parameters, those that were kept fixed
                               and the optimal, using constant reproduction
                               rate
        params_varrepro.csv: all parameters, those that were kept fixed
                             and the optimal, using variable reproduction
                             rate
        """
        # Creating a dataframes with model predictions of coffee infestation
        self.points = np.array(self.points).reshape(self.df_data.shape)
        df_prediction = pd.DataFrame(self.points*100,
                                     index=self.df_data.index,
                                     columns=self.df_data.columns)
        df_prediction[self.var_params[-1]+'(CBB/tree)'] = (
                                          opt.x[len(self.var_params)-2:]
                                          *self.parameters['kappa_b'])
        # Adding the optimum to the parameters dataframe
        values_add = []
        for item in self.var_params[:-2]:
            values_add.append(self.parameters[item])
        df_params_add = pd.DataFrame({
                                    'Parameter': list(self.var_params[:-2]),
                                    'Value': values_add})
        df_params.reset_index(inplace=True)
        for item in self.var_params[-2:]:
            ind = df_params.index[df_params['Parameter'] == item][0]
            df_params.loc[ind, 'Value'] = 'various'
        df_params = pd.concat([df_params, df_params_add], ignore_index=True)
        df_params.set_index('Parameter', inplace=True)
        
        # Updating the dataframe column of the parameter description
        file_path = Path('..') / 'data' / 'params_descriptions.json'
        try:
            with open(file_path, 'r') as file:
                dict = json.load(file)
        except FileNotFoundError:
            sys.exit(f'Please download and save the file {file_path}')
        for ind, item in enumerate(self.var_params[:-2]):
            df_params.loc[item, 'Description'] = dict[item]
        
        # A dataframe (df_x) with the optimum and the parameter bounds
        labels = list(self.var_params)
        labels[-2:] = (self.var_params[-1]+'_'
                       +df_prediction.index.astype('str'))
        df_x = pd.DataFrame({'Parameter': labels[:len(self.var_params)-2],
                            'x_lower': self.x_lower[:len(self.var_params)-2],
                            'x_opt': opt.x[:len(self.var_params)-2],
                            'x_upper': self.x_upper[:len(self.var_params)-2]
                            })
        df_add = pd.DataFrame({'Parameter': labels[len(self.var_params)-2:],
                             'x_lower': self.x_lower[len(self.var_params)-2:]
                              *self.parameters['kappa_b'],
                             'x_opt': opt.x[len(self.var_params)-2:]
                              *self.parameters['kappa_b'],
                             'x_upper': self.x_upper[len(self.var_params)-2:]
                              *self.parameters['kappa_b']})
        df_x = pd.concat([df_x, df_add], ignore_index=True)
        df_x.set_index('Parameter', inplace=True)

        print('Model parameters:')
        pd.set_option('display.max_colwidth', None)
        display(df_params) # show the parameters in a table
        print(' ')
        pd.options.display.float_format = '{:.2f}'.format
        print('Field dataset of coffee infestation (%):')
        display(self.df_data)
        print(' ')
        print('Model-predicted values of coffee infestation (%):')
        display(df_prediction)
        print(' ')
        np.set_printoptions(precision=4, suppress=True)
        pd.options.display.float_format = '{:.4f}'.format
        if self.parameters['switch']:
           print('Optimum using variable reproduction rate:')
        else:
           print('Optimum using constant reproduction rate:')
        display(df_x)
        print(' ')
        print(f'Least-square error: {opt.cost*100:.0f}%')
        print(' ')
        print(f'Algorithm termination: {opt.status}, {opt.message}')
        print(' ')

        # Saving the values of the parameters to an external file 
        if self.parameters['switch']:
            print('Writing file of parameters: ../data/params_varrepro.csv')
            file_path_w = Path('..') / 'data' / 'params_varrepro.csv'
            file_path_r = Path('..') / 'data' / 'params_varrepro_fixed.csv'
        else:
            print(('Writing file of parameters: '
                   '../data/params_constrepro.csv'))
            file_path_w = Path('..') / 'data' / 'params_constrepro.csv'
            file_path_r = Path('..') / 'data' / 'params_constrepro_fixed.csv'
        try:
            df_all_params = pd.read_csv(file_path_r)
        except FileNotFoundError:
            sys.exit(f'Please download and save the file {file_path_r}')
        df_all_params = pd.concat([df_all_params, df_params_add],
                                  ignore_index=True)
        df_all_params.set_index('Parameter', inplace=True)
        df_all_params.to_csv(file_path_w)
        print(' ')

    def plotting(self):
        """ Plot the graph of the model fit to the field data """
        k = 0
        n, m = self.df_data.shape
        t_d_list = np.array(self.t_d)
        for i,_ in enumerate(self.df_data.index):
            self.data_time.append(t_d_list+k)
            k += self.n_t
        self.data_time = np.array(self.data_time).reshape(n,m)
        self.time = np.array(self.time).reshape(n,self.n_t)
        self.array = np.array(self.array).reshape(n,self.n_t)

        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(7,5.5))
        for i,_ in enumerate(self.df_data.index):
            ax.plot(self.time[i,], self.array[i,]*100, 'k-')
            ax.plot(self.data_time[i,], self.df_data.iloc[i,], 'ko')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)
        xtick_mark, xtick_label = [], []
        for i,_ in enumerate(self.df_data.index):
            xtick_mark.append(self.time[i, self.n_t//2])
            xtick_label.append(self.df_data.index[i])
        ax.set_xticks(xtick_mark)
        ax.set_xticklabels(xtick_label)
        ax.set_ylabel('infested coffee ($\\%$)', fontsize=18)
        ax.set_title('Model fitting to data', fontsize=20)
        return fig, ax
