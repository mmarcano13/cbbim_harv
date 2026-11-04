#!/usr/bin/env python3
"""
author: Mariano Marcano
contact: mariano.marcano@gmail.com
date: 2026-10-26
license: BSD 3-Clause
"""
import sys
from pathlib import Path
import math
import time
import json
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from IPython.display import display

class initiate():
    """
    Initiate the parameter and initial condition values
    The compartmental unit of measure is a tree
    Read: switch (boolean), uses constant CBB reproduction rate (False)
                or variable reproduction rate (True) depending on the
                temperature change along time
          phi, percentage of coffee harvesting
          b0, amount of free CBBs at the start of the season
          mu, number of CBBs coming from outside in each season
          external file params_file, contains parameter values from
                Table 1 in Marcano et al 2021
    Output: a dictionary, parameters, with the parameter values
    """
    def __init__(self, switch, phi, b0, mu, params_file):
        self.switch = switch
        self.phi = phi
        self.b0 = b0
        self.mu = mu
        try:
            self.df = pd.read_csv(params_file) # read parameter values from
                            # an external file and save them in a dataframe
        except FileNotFoundError:
            sys.exit(f'Please download and save the file {params_file}')

    def parameter_values(self):
        """
        Read parameter values from the dataframe (df) and create
        a dictionary (parameters) with the parameters
        """
        parameters = {'switch': self.switch, 'phi': self.phi/100, 
                      'b0': self.b0, 'mu': self.mu}
        n = len(self.df)
        for row in self.df.itertuples(index=False):
            parameters[row.Parameter] = row.Value

        parameters['kappa_b'] = (parameters['kappa_b_berry'] # CBB carrying
                                 *parameters['coffee_total'] # capacity
                                 *parameters['cu0'])      # per tree
        if parameters['switch']:
            parameters['eps_be'] = 'varies' 
            parameters['rho_b'] = 'varies' 
        else:
            parameters['eps_be'] = (parameters['epsbe2rhob'] # CBB exiting
                                    *parameters['rho_b'])    # rate 
        parameters['cu'] = parameters['cu0'] # fraction of initial
                                             # uninfested coffee
        parameters['bo'] = self.b0/parameters['kappa_b'] # fraction of
                                                         # initial free CBB
        # Adding some parameters to the dataframe
        label = []
        if parameters['switch']:
            label.append('rho_b') 
        label = label + ['kappa_b', 'eps_be', 'cu', 'bo', 'switch', 'phi',
                         'b0', 'mu']
        new_rows = pd.DataFrame({'Parameter': label, 'Value':
                                 [parameters[lab] for lab in label]})
        self.df = pd.concat([self.df, new_rows], ignore_index=True)
        self.df.set_index('Parameter', inplace=True)
        
        # Adding a column to the dataframe with parameter descriptions
        file_path = Path('..') / 'data' / 'params_descriptions.json'
        try:
            with open(file_path, 'r') as file:
                dict = json.load(file)
        except FileNotFoundError:
            sys.exit(f'Please download and save the file {file_path}')
        descrip = []
        for key in self.df.index:
            descrip.append(dict[key])
        self.df['Description'] = descrip
        return parameters

def reset(parameters, **kwargs):
    """ Reset or add parameters as needed by the model """
    for key, value in kwargs.items():
        parameters[key] = value
        if key == 'b0':
            parameters['bo'] = value/parameters['kappa_b']
        if key == 'delta_bo':
            parameters[key] = value/parameters['kappa_b']
    return parameters

class model():
    """
    Solve the model equations of the dynamics between the coffee
    and the CBB
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def solve(self):
        """ Solve the model equations before and after harvesting """

        def variable_repro(t, epsbe2rhob):
            """
            Compute the CBB reproduction rate as a function of temperature;
            the CBB exit rate is also updated to satisfy the feasibility 
            condition (21) in Marcano et al 2021
            """
            a, b = 15, 30 # celcius degree
            nd = (self.parameters['tf'] - self.parameters['t0']) + 1
            temp = a+(b-a)*math.sin(((t+90)%nd)*math.pi/nd)
            rho, tempmax, delta, lamb = 0.00358, 34.258, 1.10846, -1.0551
            rho_b = (math.exp(rho*temp)-math.exp(rho*tempmax
                       -(tempmax-temp)/delta)+lamb) # Jaramillo et al 2009
            eps_be = epsbe2rhob*rho_b
            return rho_b, eps_be

        def yp(t, y):
            """ The right-hand side equations of the ODE model """
            cu = y[0]
            ci1 = y[1]
            bo = y[2]
            be = y[3]
            if self.parameters['switch']:
                self.parameters['rho_b'], self.parameters['eps_be'] = \
                             variable_repro(t, self.parameters['epsbe2rhob'])
            ci2 = self.parameters['cu0']-cu-ci1 # Conservation before
                                                # harvesting
            if t>self.parameters['th']:
               # Conservation after harvesting
               ci2 = self.parameters['cu0']*(1-self.parameters['phi'])-cu-ci1
            eq_1 = -self.parameters['eps_c']*cu*bo
            eq_2 = (self.parameters['eps_c']*cu*bo
                    - self.parameters['eps_ci']*ci1)
            eq_3 = (self.parameters['eps_be']*be*ci2
                    - self.parameters['eps_bo']*cu*bo
                    - self.parameters['del_b']*bo)
            eq_4 = (self.parameters['rho_b']*be*(ci2-be)
                    + self.parameters['eps_bo'] *cu*bo
                    - self.parameters['eps_be']*be*ci2)
            return [eq_1, eq_2, eq_3, eq_4]

        t_span = [self.parameters['t0'], self.parameters['th']
                          +self.parameters['delta_t']]
        x_init = [self.parameters['cu'], self.parameters['ci1'],
                  self.parameters['bo'], self.parameters['be']]
        t_eval = np.arange(self.parameters['t0'], self.parameters['th']
                           +self.parameters['delta_t'],
                           self.parameters['delta_t'])
        sol_before = solve_ivp(yp,  t_span, x_init, method='RK45',
                               t_eval=t_eval, rtol=1e-6, atol=1e-8)
        t_span = [self.parameters['th']+self.parameters['delta_t'],
                  self.parameters['tf']+self.parameters['delta_t']]
        x_init = [(1-self.parameters['phi'])*sol_before.y[0,-1],
                  (1-self.parameters['phi'])*sol_before.y[1,-1],
                  sol_before.y[2,-1],
                  (1-self.parameters['phi'])*sol_before.y[3,-1]]
        t_eval = np.arange(self.parameters['th']+self.parameters['delta_t'],
                           self.parameters['tf']+self.parameters['delta_t'],
                           self.parameters['delta_t'])
        sol_after = solve_ivp(yp, t_span, x_init, method='RK45',
                              t_eval=t_eval, rtol=1e-6, atol=1e-8)
        return sol_before, sol_after

    def printing(self, sol_before, sol_after):
        """ Printing relevant information """
        print(f'Results of the model for a coffee season:')
        print((f'Percentage of consumable coffee at harvest: {round(100
                *(sol_before.y[0,-1]+sol_before.y[1,-1]))}%'))
        print((f'Percentage of consumable coffee at the end of the season: '
              f'{round( 100*(sol_after.y[0,-1]
                +sol_after.y[1,-1]))}%'))
        print((f'Amount of CBB at the beginning of the season: {round(
                 self.parameters['kappa_b']*sol_before.y[2,0])} '
                'CBB/tree'))
        print((f'Amount of CBB at the end of the season: {round(
                self.parameters['kappa_b']*(sol_after.y[2,-1]
                +sol_after.y[3,-1]))} CBB/tree'))
        print(' ')
        if (sol_before.status or sol_after.status)!=0:
           print(('Check, one of the integrations did not reach the end '
                  'of the integration interval.'))
           print(' ')

    def plotting(self, sol_before, sol_after):
        """ Plotting the population graphs """
        ci2_before = self.parameters['cu0']-(sol_before.y[0,]
                                                  +sol_before.y[1,])
        ci2_after = ((1-self.parameters['phi'])*self.parameters['cu0']
                           -(sol_after.y[0,]+sol_after.y[1,]))
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(7,5.5))
        ax.plot(sol_before.t, 
                self.parameters['coffee_total']*self.parameters['cu0']
                *sol_before.y[0,], 'r-', label=r'$c_{\rm u}$')
        ax.plot(sol_after.t,
                self.parameters['coffee_total']*self.parameters['cu0']
                *sol_after.y[0,], 'r-')
        ax.plot(sol_before.t,
                self.parameters['coffee_total']*self.parameters['cu0']
                *sol_before.y[1,], 'r--', label=r'$c_{\rm i1}$')
        ax.plot(sol_after.t,
                self.parameters['coffee_total']*self.parameters['cu0']
                *sol_after.y[1,], 'r--')
        ax.plot(sol_before.t,
                self.parameters['coffee_total']*self.parameters['cu0']
                *ci2_before, 'r-.', label=r'$c_{\rm i2}$')
        ax.plot(sol_after.t,
                self.parameters['coffee_total']*self.parameters['cu0']
                *ci2_after, 'r-.')
        ax.plot(sol_before.t, self.parameters['kappa_b']
                *sol_before.y[2,], 'b-', label=r'$b_{\rm o}$')
        ax.plot(sol_after.t, self.parameters['kappa_b']
                *sol_after.y[2,], 'b-')
        ax.plot(sol_before.t, self.parameters['kappa_b']
                *sol_before.y[3,], 'b--', label=r'$b_{\rm e}$')
        ax.plot(sol_after.t, self.parameters['kappa_b']
                *sol_after.y[3,], 'b--')
        ax.plot(sol_before.t[-1], 
                self.parameters['coffee_total']*self.parameters['cu0']
                *sol_before.y[0,-1], marker='o', markersize=8,
                alpha=0.75, markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=1.5, label=r'$c_{\rm u}(t_{\rm h})$', ls=' ')
        ax.plot(sol_before.t[-1], 
                self.parameters['coffee_total']*self.parameters['cu0']
                *(sol_before.y[0,-1]+sol_before.y[1,-1]), 
                marker='s', markersize=8, alpha=0.75,
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=1.5,
                label=r'$c_{\rm u}(t_{\rm h})+c_{\rm i1}(t_{\rm h})$',
                ls = ' ')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('time (day)', fontsize=16)
        ax.set_ylabel('population (per tree)', fontsize=16)
        ax.legend(shadow=True)
        ax.set_title('Coffee and CBB dynamics', fontsize=18)
        return fig, ax

class multiseason():
    """
    Solve the model equations for multiple seasons using this control
    practice until the CBB is controlled or eliminated
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def outputs(self):
        """ 
        Solve the model equations for multiple seasons, plot and print
        relevant results
        """
        tol = 2E-5 # stopping criterium
        ny = 0 # current season
        nd = self.parameters['tf']-self.parameters['t0'] # length of a season
        t0_save, th_save = self.parameters['t0'], self.parameters['th']
        tf_save = self.parameters['tf'] 
        simulation = model(self.parameters)
        sol_before, sol_after = simulation.solve() # solve the model
                                          # equations for the current season
        th_array = np.array(sol_before.t[-1]/nd)
        tf_array = np.array(sol_after.t[-1]/nd)
        t_before_harvest = sol_before.t/nd
        t_after_harvest = sol_after.t/nd
        cu_before_harvest = sol_before.y[0,]
        cu_after_harvest = sol_after.y[0,]
        cu_at_harvest = sol_before.y[0,-1]
        ci1_at_harvest = sol_before.y[1,-1]
        bo_before_harvest = sol_before.y[2,]
        bo_after_harvest = sol_after.y[2,]
        be_before_harvest = sol_before.y[3,]
        be_after_harvest = sol_after.y[3,]
        total_borer_at_end = sol_after.y[2,-1] + sol_after.y[3,-1]

        # Iterate over subsequent seasons
        self.parameters['bo'] = (total_borer_at_end
                       + self.parameters['mu']/self.parameters['kappa_b'])
        while ((abs(self.parameters['bo']-self.parameters['stable_1']
                    /self.parameters['kappa_b']) > tol) and
               (ny<self.parameters['ny_max'])):
            ny += 1
            self.parameters['t0'] += nd
            self.parameters['th'] += nd
            self.parameters['tf'] += nd
            
            simulation = model(self.parameters)
            sol_before, sol_after = simulation.solve() # solve the model
                                          # equations for the current season
            th_array = np.concatenate((th_array, sol_before.t[-1]/nd),
                                        axis=None)
            tf_array = np.concatenate((tf_array, sol_after.t[-1]/nd),
                                        axis=None)
            t_before_harvest = np.vstack((t_before_harvest, sol_before.t/nd))
            t_after_harvest = np.vstack((t_after_harvest, sol_after.t/nd))
            cu_before_harvest = np.vstack((cu_before_harvest,
                                           sol_before.y[0,]))
            cu_after_harvest = np.vstack((cu_after_harvest,
                                          sol_after.y[0,]))
            cu_at_harvest = np.concatenate((cu_at_harvest,
                                            sol_before.y[0,-1]), axis=None)
            ci1_at_harvest = np.concatenate((ci1_at_harvest,
                                            sol_before.y[1,-1]), axis=None)
            bo_before_harvest = np.vstack((bo_before_harvest,
                                           sol_before.y[2,]))
            bo_after_harvest = np.vstack((bo_after_harvest, sol_after.y[2,]))
            be_before_harvest = np.vstack((be_before_harvest,
                                           sol_before.y[3,]))
            be_after_harvest = np.vstack((be_after_harvest, sol_after.y[3,]))
            total_borer_at_end = np.concatenate((total_borer_at_end,
                              sol_after.y[2,-1] + sol_after.y[3,-1]),
                              axis=None)
            self.parameters['bo'] = (total_borer_at_end[-1] 
                                     + self.parameters['mu']
                                     /self.parameters['kappa_b'])
        self.parameters = reset(self.parameters, t0=t0_save, th=th_save,
                                tf=tf_save)

        # Printing relevant information
        print(f'Results of multiple coffee seasons:')
        print(f'Initial amount of CBBs: {self.parameters['b0']}')
        print((f'CBBs at the end of season {ny+1}: '
               f'{round(self.parameters['bo']
                  *self.parameters['kappa_b'])} CBB/tree '))
        print(' ')

        # Plotting the population graphs 
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(7,5.5))
        ax.plot(t_before_harvest[0,],
                self.parameters['coffee_total']*self.parameters['cu0']
                *cu_before_harvest[0,], 'r-', label=r'$c_{\rm u}$')
        ax.plot(t_after_harvest[0,],
                self.parameters['coffee_total']*self.parameters['cu0']
                *cu_after_harvest[0,], 'r-')
        ax.plot(t_before_harvest[0,],
                self.parameters['kappa_b']*bo_before_harvest[0,], 'b-',
                label=r'$b_{\rm o}$')
        ax.plot(t_after_harvest[0,],
                self.parameters['kappa_b']*bo_after_harvest[0,], 'b-')
        ax.plot(t_before_harvest[0,],
                self.parameters['kappa_b']*be_before_harvest[0,], 'b--',
                label=r'$b_{\rm e}$')
        ax.plot(t_after_harvest[0,],
                self.parameters['kappa_b']*be_after_harvest[0,], 'b--')
        ax.plot(th_array[0,], 
                 self.parameters['coffee_total']*self.parameters['cu0']
                 *cu_at_harvest[0,], marker='o', markersize=8, alpha=0.75,
                 markerfacecolor='white', markeredgecolor='black',
                 markeredgewidth=1.5, label=r'$c_{\rm u}(t_{\rm h})$',
                 ls=' ')
        ax.plot(th_array[0,], 
                 self.parameters['coffee_total']*self.parameters['cu0']
                 *(cu_at_harvest[0,]+ci1_at_harvest[0,]), marker='s',
                 markersize=8, alpha=0.75, markerfacecolor='white',
                 markeredgecolor='black', markeredgewidth=1.5,
                 label=r'$c_{\rm u}(t_{\rm h})+c_{\rm i1}(t_{\rm h})$',
                 ls = ' ')
        ax.plot(tf_array[0,]*np.ones(1000), np.arange(1000), 'k:')

        for i in range(1,ny+1):
            ax.plot(t_before_harvest[i,],
                    self.parameters['coffee_total']*self.parameters['cu0']
                    *cu_before_harvest[i,], 'r-')
            ax.plot(t_after_harvest[i,],
                    self.parameters['coffee_total']*self.parameters['cu0']
                    *cu_after_harvest[i,], 'r-')
            ax.plot(t_before_harvest[i,],
                    self.parameters['kappa_b']*bo_before_harvest[i,], 'b-')
            ax.plot(t_after_harvest[i,],
                    self.parameters['kappa_b']*bo_after_harvest[i,], 'b-')
            ax.plot(t_before_harvest[i,],
                    self.parameters['kappa_b']*be_before_harvest[i,], 'b--')
            ax.plot(t_after_harvest[i,],
                    self.parameters['kappa_b']*be_after_harvest[i,], 'b--')
            ax.plot(th_array[i,], 
                    self.parameters['coffee_total']*self.parameters['cu0']
                    *cu_at_harvest[i,], marker='o', markersize=8, alpha=0.75,
                    markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=1.5) 
            ax.plot(th_array[i,], 
                    self.parameters['coffee_total']*self.parameters['cu0']
                    *(cu_at_harvest[i,]+ci1_at_harvest[i,]), marker='s',
                    markersize=8, alpha=0.75, markerfacecolor='white',
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(tf_array[i,]*np.ones(1000), np.arange(1000), 'k:')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        xticks_list = range(round(ax.get_xlim()[0]),
                            round(ax.get_xlim()[1])+1, 2)
        ax.set_xticks(xticks_list)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('time (year)', fontsize=16)
        ax.set_ylabel('population (per tree)', fontsize=16)
        ax.legend(shadow=True, loc='center right')
        ax.set_title((f'Coffee and CBB dynamics of {ny+1} consecutive '
                      f'seasons \n for {self.parameters['phi']*100:.0f}'
                      f'$\\%$ harvesting and immigration of '
                      f'{self.parameters['mu']} CBB/tree'), fontsize=18)
        return fig, ax

class map_1d():
    """
    One-dimensional map to compute fixed points of the map for the CBB
    infestation model and classify them as stable or unstable
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def solve(self):
        """
        Compute the map and classify the fixed points as stable or unstable
        """
        g = []
        bo_array = []
        stable_pts = []
        unstable_pts = []
     
        st = time.perf_counter()
        bo_array.append(self.parameters['bo']*self.parameters['kappa_b'])
        simulation = model(self.parameters)
        sol_before, sol_after = simulation.solve() # solve the model
                                  # equations before and after coffee harvest
        g.append((sol_after.y[2,-1]+sol_after.y[3,-1]) # Evaluate the
                  *self.parameters['kappa_b']+self.parameters['mu']) # fixed-
                                          # point equation: bo_k+1 = g (bo_k)
        ind = 0 
        # Seeking 2 stable fixed points, with CBB immigration may be only one
        while ((len(stable_pts)!=2) and (ind<=1000 or (len(unstable_pts)!=0
               or len(stable_pts)!=1)) and (ind<self.parameters['max_itr'])):
            ind = ind + 1
            self.parameters['bo'] = (self.parameters['bo']
                 + self.parameters['delta_bo'])
            bo_array.append(self.parameters['bo']*self.parameters['kappa_b'])
            simulation = model(self.parameters)
            sol_before, sol_after = simulation.solve()
            g.append((sol_after.y[2,-1]+sol_after.y[3,-1])
                      *self.parameters['kappa_b']+self.parameters['mu'])
            # Fixed-point classification: stable or unstable
            if (g[ind-1]==bo_array[ind-1] and g[ind]<bo_array[ind]):
                stable_pts.append(round(g[ind-1]))
            elif (g[ind-1]>bo_array[ind-1] and g[ind]<bo_array[ind]):
                if abs(g[ind-1]-bo_array[ind-1]) < abs(g[ind]-bo_array[ind]):
                    stable_pts.append(round(g[ind-1]))
                else:
                    stable_pts.append(round(g[ind]))
            if (g[ind-1]==bo_array[ind-1] and g[ind]>bo_array[ind]):
                unstable_pts.append(round(g[ind-1]))
            elif (g[ind-1]<bo_array[ind-1] and g[ind]>bo_array[ind]):
                if abs(g[ind-1]-bo_array[ind-1]) < abs(g[ind]-bo_array[ind]):
                    unstable_pts.append(round(g[ind-1]))
                else:
                    unstable_pts.append(round(g[ind]))

        # Adding more points at the end of the map to show the 2nd stable
        # point neighborhood
        for k in range(1000):
            self.parameters['bo'] = (self.parameters['bo'] 
                                     + self.parameters['delta_bo'])
            bo_array.append(self.parameters['bo']*self.parameters['kappa_b'])
            # Solve the model equation before and after coffee harvest
            simulation = model(self.parameters)
            sol_before, sol_after = simulation.solve()
            # CBBs available for the next season, bo(tf)+be(tf)
            g.append((sol_after.y[2,-1]+sol_after.y[3,-1])
                     *self.parameters['kappa_b']+self.parameters['mu'])

        # Converting lists to numpy arrays
        g = np.array(g)
        bo_array = np.array(bo_array)
        stable_pts = np.array(stable_pts)
        unstable_pts = np.array(unstable_pts)
        et = time.perf_counter() - st

        return g, bo_array, stable_pts, unstable_pts, et

    def printing(self, g, bo_array, stable_pts, unstable_pts, et):
        """ Printing relevant map's information """
        print((f'Percentage of coffee harvested: '
               f'{self.parameters['phi']*100:.0f}%'))
        print((f'CBBs coming from outside in each season after the 1st: '
               f'{self.parameters['mu']} CBB/tree'))
        if self.parameters['switch']:
           print('Using variable reproduction rate')
        else:
           print('Using constant reproduction rate')
        print(' ')

        print('Results of the map:')
        if stable_pts.size == 2:
            print(f'first attractor: {stable_pts[0]} CBB/tree')
            print(f'threshold: {unstable_pts[0]} CBB/tree')
            print(f'second attractor: {stable_pts[1]} CBB/tree')
            print('-------------------------------------------------------')
            print('attractor: attracting or stable fixed point')
            print('threshold: repelling or unstable fixed point')
            print('Note:')
            print('For any value of initial free CBB/tree in the interval')
            print((f'[{stable_pts[0]}, {unstable_pts[0]}) the '
                   f'amount of free CBB at the start of subsequent seasons'))
            print((f'decreases to {stable_pts[0]} CBB/tree if one '
                   f'maintains at least {self.parameters['phi']*100:.0f}% '
                    'of harvesting.'))
            print(('Whereas for any value of initial free CBB/tree in the '
                  'interval'))
            print((f'({unstable_pts[0]}, {stable_pts[1]}] the '
                  f'amount of free CBB at the start of subsequent seasons'))
            print((f'increases to {stable_pts[1]} CBB/tree if one '
                  f'maintains at least {self.parameters['phi']*100:.0f}% of '
                   'harvesting.'))
        else:
            print(f'The only attractor: {stable_pts[0]} CBB/tree')
            print('-------------------------------------------------------')
            print('attractor: attracting or stable fixed point')
            print('Note:')
            print(('Owing to CBB immigration for any value of initial '
                  'free CBB/tree'))
            print(('the amount of free CBB at the start of subsequent '
                   'seasons increases'))
            print((f'to {stable_pts[0]} CBB/tree if one maintains '
                   f'{self.parameters['phi']*100:.0f}% of harvesting or '
                    'less.'))
        print(' ')
        print(f'Execution time of the map: {et/60} min')
        print(' ')

    def plotting(self, g, bo_array, stable_pts, unstable_pts):
        """ Plot the one-dimensional map """

        def arrow_pos(arrow_start_index, arrow_end_index):
            """ Using arrow patch to insert arrows on the map """
            arrow_start = (bo_array[arrow_start_index],
                           g[arrow_start_index])
            arrow_end = (bo_array[arrow_end_index],
                         g[arrow_end_index])
            arrow_style = patches.ArrowStyle('fancy', head_length=0.75,
                                        head_width=0.75, tail_width=0.001)
            arrow = patches.FancyArrowPatch(arrow_start, arrow_end,
                                  mutation_scale=12,  # Size of the arrowhead
                                  arrowstyle=arrow_style,  # Arrow style
                                  fc='red',  # Fill color
                                  ec='red',  # Edge color
                                  lw=0)  # Line width
            return arrow

        def arrow_add(ax):
            """ Add arrows to the plot """
            thr_ind, att_1_ind, att_2_ind = [], [], []
            if unstable_pts.size > 0:
                thr_ind = np.argwhere(bo_array <=unstable_pts[0])[-1,0]
                att_1_ind = np.argwhere(bo_array <=stable_pts[0])[-1,0]
                att_2_ind = np.argwhere(bo_array <=stable_pts[1])[-1,0]
                zoom_ind = np.argwhere(bo_array <=2*unstable_pts[0])[-1,0]
                thr_order = round(math.log10(unstable_pts[0]))
                delta_att_1 = (thr_ind - att_1_ind)//4
                delta_att_2_zi = (zoom_ind - thr_ind)//(2**(thr_order+1))
                delta_att_2 = (att_2_ind - thr_ind)//6
                for k in range(1,4):
                    arrow_sind = att_1_ind+(k+1)*delta_att_1
                    arrow_eind = arrow_sind - delta_att_1//2
                    if (arrow_sind <= len(bo_array)) and (arrow_eind
                        <= len(bo_array)):
                        arrow = arrow_pos(arrow_sind, arrow_eind)
                        ax[1].add_patch(arrow) # Add the arrow to the axes
                    arrow_sind = thr_ind + k*delta_att_2_zi
                    arrow_eind = arrow_sind + delta_att_2_zi//2
                    if (arrow_sind <= len(bo_array)) and (arrow_eind
                        <= len(bo_array)):
                        arrow = arrow_pos(arrow_sind, arrow_eind)
                        ax[1].add_patch(arrow) # Add the arrow to the axes
                    arrow_sind = thr_ind + k*delta_att_2
                    arrow_eind = arrow_sind + delta_att_2//2
                    if (arrow_sind <= len(bo_array)) and (arrow_eind
                        <= len(bo_array)):
                        arrow = arrow_pos(arrow_sind, arrow_eind)
                        ax[0].add_patch(arrow) # Add the arrow to the axes
            else:
                att_1_ind = np.argwhere(g<=stable_pts[0])[-1,0]
                delta_att_1 = att_1_ind//4
                for k in range(1,4):
                    # Index to start the arrow
                    arrow_sind = k*delta_att_1
                    # Index to end the arrow
                    arrow_eind = arrow_sind + delta_att_1//2
                    arrow = arrow_pos(arrow_sind, arrow_eind)
                    ax.add_patch(arrow) # Add the arrow to the axes
            return ax, thr_ind, att_1_ind
    
        def handles():
            """ Legend handles """
            lmap = mlines.Line2D([], [], linestyle='-', color='blue',
                                 label='map')
            lid = mlines.Line2D([], [], linestyle='-', color='black',
                                label='identity')
            bmarker = mlines.Line2D([], [], marker='o', markersize=8,
                                alpha=1, markerfacecolor='black',
                                markeredgecolor='black', markeredgewidth=1.5,
                                ls=' ', label='attractor')
            wmarker = mlines.Line2D([], [], marker='o', markersize=8,
                                alpha=1, markerfacecolor='white',
                                markeredgecolor='black',
                                markeredgewidth=1.5, ls=' ',
                                label='repeller')
            return lmap, lid, bmarker, wmarker

        # Plot the graph of the map
        plt.rcParams['text.usetex'] = True
        if unstable_pts.size > 0:
            fig, ax = plt.subplots(ncols=2, figsize=(7,6))
            for i in range(len(ax)):
                ax[i].plot(bo_array, g, 'b-', linewidth=1.5)
                ax[i].plot(bo_array, bo_array, 'k-', linewidth=1.5)
                for pt in stable_pts:
                    ax[i].plot(pt, pt, marker='o', markersize=8, alpha=1,
                               markerfacecolor='black',
                               markeredgecolor='black',
                               markeredgewidth=1.5, ls=' ')
                for pt in unstable_pts:
                    ax[i].plot(pt, pt, marker='o', markersize=8, alpha=1,
                               markerfacecolor='white',
                               markeredgecolor='black',
                               markeredgewidth=1.5, ls=' ')
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['top'].set_visible(False)
                ax[i].tick_params(axis='x', direction='out', labelsize=12)
                ax[i].tick_params(axis='y', direction='out', labelsize=12)

            # Adding arrows on the map curve
            ax, thr_ind, att_ind = arrow_add(ax)

            # Plot legend
            lmap, lid, bmarker, wmarker = handles()
            ax[0].legend(handles=[lmap, lid, bmarker, wmarker], shadow=True)

            # Zooming in the map to show the threshold point
            ax[1].set_xlim(max(-0.1*unstable_pts[0], -20), 2*unstable_pts[0])
            ax[1].set_ylim(max(-0.1*g[thr_ind],-40), 2*g[thr_ind])
            ax[1].set_title(f'zoom in of the map', fontsize=14)
            ax[0].set_title(f'entire map', fontsize=14)
        else:
            fig, ax = plt.subplots(figsize=(7,6))
            ax.plot(bo_array, g, 'b-', linewidth=1.5)
            ax.plot(bo_array, bo_array, 'k-', linewidth=1.5)
            for pt in stable_pts:
                ax.plot(pt, pt, marker='o', markersize=8, alpha=1,
                        markerfacecolor='black', markeredgecolor='black',
                        markeredgewidth=1.5, ls=' ')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', direction='out', labelsize=12)
            ax.tick_params(axis='y', direction='out', labelsize=12) 
            # Adding arrows on the map curve
            ax, thr_ind, att_ind = arrow_add(ax)
            # Plot legend
            lmap, lid, bmarker, wmarker = handles()
            ax.legend(handles=[lmap, lid, bmarker], shadow=True)
            xmin, _ = ax.get_xlim()
            ymin, _ = ax.get_ylim()
            ax.set_xlim(xmin, 1.3*stable_pts[0])
            ax.set_ylim(ymin, 1.3*g[att_ind])

        fig.supxlabel('$s_n$ (CBB/tree)', fontsize=14)
        fig.supylabel('$s_{n+1}$ (CBB/tree)', fontsize=14)
        fig.suptitle((f'One-dimensional map, ' 
                      f'{self.parameters['phi']*100:.0f}$\\%$ '
                      f'harvesting and {self.parameters['mu']} immigration'),
                     fontsize=16)
        return fig, ax

class bpoints():
    """
    Using the one-dimesional map to compute fixed points of the map
    and classify them as stable or unstable for different harvest
    percentages
    """
    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def solve(phi, parameters):
        """
        For each value of harvest percentage, phi, compute a fixed
        point of the map and classify them as stable or unstable
        """
        parameters['phi'] = phi
        map_simulation = map_1d(parameters)
        _, _, stable_pts, unstable_pts, _ = map_simulation.solve()
        return [stable_pts, unstable_pts]

    def outputs(self, phi_array, stable_array, unstable_array,
                stable_lonely, execution_time, process_nums):
        """
        Print the map fixed points for different values of phi and
        plot the graph of the bifurcation diagram
        """
        num_2stable_pts = stable_array.shape[0] 

        att_1 = []
        att_2 = []
        thr = []
        for ind,_ in enumerate(phi_array):
            if num_2stable_pts > ind:
               att_1.append(stable_array[ind,0])
               thr.append(unstable_array[ind,0])
               att_2.append(stable_array[ind,1])
            else:
               att_1.append('--')
               thr.append('--')
               att_2.append(stable_lonely[ind-num_2stable_pts,0])

        data = {'Harvesting(%)': phi_array*100, 'First attractor': att_1,
                'Threshold': thr, 'Second attractor': att_2}

        df = pd.DataFrame(data)
        df = df.astype({'Harvesting(%)': int})
        df.set_index('Harvesting(%)', inplace=True)

        # Printing relevant information
        print((f'CBBs coming from outside: '
               f'{self.parameters['mu']} CBB/tree'))
        if self.parameters['switch']:
           print('Using variable reproduction rate')
        else:
           print('Using constant reproduction rate')
        print(' ')
        display(df)
        print(' ')
        print('attractor (CBB/tree): attracting or stable fixed point')
        print('threshold (CBB/tree): repelling or unstable fixed point')
        print('Note:')
        print(('1) To be able to control the infestation of CBB choose a '
               'percentage'))
        print(('of harvesting such that the initial free CBB/tree in the '
               'first'))
        print('season is less than the corresponding threshold.')
        print(('2) Graph scale on the vertical axis is split: lower values '
               '0 to 500'))
        print(('correspond to the threshold and first attractor branch; '
               'higher'))
        print(('values 1 to 7 ×10^4 correspond to the second attractor '
               'branch.'))
        print(' ')
        print((f'Execution time of the bifurcation diagram: '
              f'{execution_time/60} min'))
        print(f'Using {process_nums} CPU cores in parallel')
        print(' ')

        # Ploting the bifurcation diagram of the three branches
        plt.rcParams['text.usetex'] = True
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

        for ind, phi in enumerate(phi_array):
            if num_2stable_pts > ind:
               for pt in stable_array[ind,:-1]:
                   ax2.scatter(phi*100, pt, c='black', s=50, alpha=1,
                               edgecolors='black')
               for pt in unstable_array[ind,]:
                   ax2.scatter(phi*100, pt, c='white', s=50, alpha=1,
                               edgecolors='black')
               ax1.scatter(phi*100, stable_array[ind, -1], c='black', s=50,
                           alpha=1, edgecolors='black')
            else:
               ax1.scatter(phi*100, stable_lonely[ind-num_2stable_pts,],
                           c='black', s=50, alpha=1, edgecolors='black')

        ax1.set_ylim(9000, 70000)  # the third branch only
        ax2.set_ylim(-50, 550)  # first two branches

        ax1.spines['bottom'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax1.tick_params(axis='x', which='both', bottom=False)
        ax2.tick_params(axis='x', which='both', top=False)
        ax1.tick_params(axis='y', direction='out', labelsize=16)
        ax2.tick_params(axis='both', direction='out', labelsize=16)
        ax2.set_xlabel('Harvesting ($\\%$)', fontsize=16)
        fig.supylabel('$s^{*}$ (CBB/tree)', fontsize=16)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,3),
                             useMathText=True)
                      
        d = 0.5  # proportion of vertical to horizontal extent of the
                 # slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1,
                      clip_on=False)
        ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
        ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

        bmarker = mlines.Line2D([], [], marker='o', markersize=8, alpha=1,
                   markerfacecolor='black', markeredgecolor='black',
                   markeredgewidth=1.5, ls=' ', label='attractor')
        wmarker = mlines.Line2D([], [], marker='o', markersize=8, alpha=1,
                           markerfacecolor='white', markeredgecolor='black',
                           markeredgewidth=1.5, ls=' ', label='threshold')
        ax1.legend(handles=[bmarker, wmarker], shadow=True)
        fig.suptitle(('Bifurcation diagram for different harvesting '
                      'percentages'), fontsize=18)
        return fig, (ax1, ax2)
