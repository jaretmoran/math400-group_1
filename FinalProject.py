#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:37:38 2022

@author: cristobalpadilla
"""
import sys
#from pytexit import py2tex
import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
import bisection
import secant
import newton
import center_difference

def newtonMethod(f, x0, delta, maxIterations):    
    print("\niteration\t\t x value \t\t\t\t\t\t f(x)")
    
    # Lists used for the graphs
    x = []
    fx = []
    m = []

    iter_counter = 0
    h = 0.0125
    #h = np.arange(0, h_step*maxIterations + h_step, h_step)
    
    while abs(f(x0)) > delta and iter_counter < maxIterations:
        print(iter_counter + 1, "\t\t\t ", end=" ")
        try:
            x0 = newton.newtonStep(f, x0, h)
            
            # Data for graphs
            x.append(x0)
            fx.append(f(x0))
            m.append(center_difference.center_diff(f, x0, h))
            
        except:
            raise Exception("Division by zero. f'(x) = 0")
        
        print(x0, "\t\t\t", f(x0))
        iter_counter += 1
        
    return x0, iter_counter, x, fx, m

def newtonBisection(f, a, b, delta, maxIterations, s=0.1):
    x0 = a
    iter_counter = 0
    a_initial = a
    b_initial = b
    h = 0.2
    
    # Lists used for the graphs
    x = []
    fx = []
    interval = []
    method = []
    
    # Initial values of f(a) and f(b)
    fa = f(a);     
    fb = f(b);
    
    if fa*fb > 0:
        print("Error! f must have different signs at the endpoints. Aborting")
        sys.exit(1)

    while(abs(f(x0)) > delta and iter_counter < maxIterations):
        interval.append(b-a)
        
        # Bisection Step
        if((a >= x0 and x0 >= b) or (abs(b-a) > s*(b_initial - a_initial))):
            x0 = float(a+b)/2.0
            method.append("Bisection")
            
            fx0 = f(x0)
            if fx0*fb < 0:
                a = x0; fa = fx0
            else:
                b = x0; fb = fx0
        
        # Newton Step
        else:
            x0 = newton.newtonStep(f, x0, h)
            method.append("Newton")
            
        x.append(x0)
        fx.append(f(x0))
        iter_counter += 1
   
    return x0, iter_counter, method, x, fx, interval

def f(x):
    return np.tanh(x)
    
def main():
    
    ########################################################################
    ####### Problem 1
    ########################################################################
    
    precision = "0.6E"   # Output precision format
    
    # Use this f_str if py2tex has been installed.
    #f_str = str(py2tex("np.tanh(x)"))[1:-1]
    # Else, use this f_str
    f_str = "tanh(x)"
    
    # Part a.1
    '''
    initial_guess = 1.08
    '''
    
    # Part a.2
    '''
    initial_guess = 1.09
    '''
    '''
    print("\nInitial guess: ", initial_guess)    
    solution, itera, x, fx, m = newtonMethod(f, initial_guess, 1e-6, 5)
    
    x_vals = np.linspace(-10, 10)
    for i in range(0, len(x)):
        
        abline_values = [m[i]*(n - x[i]) + fx[i] for n in x_vals]    
        plt.plot(x_vals, abline_values, '--', label='iteration: {0}'.format(i+1))
    
    
    plt.title("Root Approximation of " + f_str + ",\n with initial guess x = {0}".format(initial_guess))
    plt.plot(np.zeros(len(x_vals)), np.linspace(-5, 5), '--', c="lightgray")
    plt.plot(x_vals, np.zeros(len(x_vals)), '--', c='lightgray')
    plt.plot(x_vals, np.tanh(x_vals))
    plt.ylim(-1, 1)
    plt.legend(loc='best')
    plt.show()
    '''
    
    # Part b.1
    '''
    x0 = 1.08
    x1 = 1.09
    '''
    
    # Part b.2
    '''
    x0 = 1.09
    x1 = 1.1
    '''
    
    # Part b.3
    '''
    x0 = 1
    x1 = 2.3
    '''
    
    # Part b.3
    '''
    x0 = 1
    x1 = 2.4
    '''
    '''
    tolerance = 1e-6
    max_steps = 10

    solution, no_iterations = secant.secant(f, x0, x1, tolerance, max_steps)
    print("Number of iterations = ",no_iterations)
    print("An estimate of the root is ",solution)
    '''
    
    # Part c
    '''
    # Interval end points
    x0 = -5
    xf = 3    

    solution, no_iterations = bisection.bisection(f, x0, xf,1e-6, str_precision=precision)
    print("\nNumber of iterations = ",no_iterations)
    print("An estimate of the root is %{0}".format(precision) % (solution))
    
    # Plot graph.
    x = np.linspace(0, 5, 50)
    
    plt.title("Graph of f(x) = " + f_str)
    plt.ylabel("f(x)")
    plt.xlabel("x") 
    plt.grid()
    plt.plot(x, f(x))  # plot f(x)
    plt.plot(x, np.zeros(len(x)), '--', c="gray")  # plot x-axis  
    '''
    
    
    # Part d
    '''
    solution, iterations, method, x, fx, intervals = newtonBisection(f, -10, 15, 1e-6, 10)

    print("\nInterval [-10, 15]")
    print("\nInterval Length \t\t Method \t\t\t\t x-value \t\t\t f(x)")
    print("---------------------------------------------------------------------------")

    
    for i in range(0,len(x)):
        print("%{0} \t\t".format(precision) % (intervals[i]), end=" ")
        print(method[i].ljust(10, " ") + "\t\t", end=" ")
        print("%{0:>10} \t\t".format(precision) % (x[i]), end=" ")
        print("%{0:>10} \t\t".format(precision) % (fx[i]))

    print("\nNumber of iterations: ", iterations)    
    print("Solution: ", solution)
    '''
    
    return 0

if __name__ == "__main__":
    main()