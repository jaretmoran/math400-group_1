#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:37:38 2022

@author: Team 1
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
import bisection
import secant
import newton
import center_difference
import gaussian_elimination_w_pp as pp

def newtonMethod(f, x0, delta, maxIterations):
    precision = "0.6E"    
    print("\niteration\t\t x value \t\t\t f(x)")
    
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
        
        print("%{0:>10} \t\t %{0:>10}".format(precision) % (x0, f(x0)))
        #print(x0, "\t\t\t", f(x0))
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

def centerDifference(function, x, var, h = 0.0125):
    result = 0
    
    if var == 'x':
        result = (function(x[0] + h, x[1], x[2]) - function(x[0] - h, x[1], x[2])) / (2 * h)
        
    elif var == 'y':
        result = (function(x[0], x[1] + h, x[2]) - function(x[0], x[1] - h, x[2])) / (2 * h)
        
    elif var == 'z':
        result = (function(x[0], x[1], x[2] + h) - function(x[0], x[1], x[2] - h)) / (2 * h)
        
    return result

def newtonMethod_multivariate(xn, h=0.0125):
    steps = 0
    
    while True:
        steps += 1
        xn_previous = xn
        xn = xn + gaussianElimination(jacobian(xn), negativeFArrow(xn))
        
        print("\nStep number: ", steps, ", x = ", xn, end=" ")
        if np.linalg.norm(xn - xn_previous, np.inf) < 5e-6 or steps == 100:
            break
    
    return xn, steps

def jacobian(xn, h=0.0125):
    return np.matrix([
        [centerDifference(f, xn, 'x', h), centerDifference(f, xn, 'y', h), centerDifference(f, xn, 'z', h)],
        [centerDifference(g, xn, 'x', h), centerDifference(g, xn, 'y', h), centerDifference(g, xn, 'z', h)],
        [centerDifference(m, xn, 'x', h), centerDifference(m, xn, 'y', h), centerDifference(m, xn, 'z', h)]
    ])

def negativeFArrow(xn):
    return np.matrix([[-f(xn[0], xn[1], xn[2])], [-g(xn[0], xn[1], xn[2])], [-m(xn[0], xn[1], xn[2])]])

def gaussianElimination(A, b):
    return pp.part_piv_ge(A, b)

def f_pm1(x):
    return np.tanh(x)

def f(x, y, z):
    return x**2 + y**2 + z**2 - 10

def g(x, y, z):
    return x + 2 * y - 2

def m(x, y, z):
    return x + 3 * z - 9
 
def partA(initial_guess, f_str, tolerance, maxSteps, precision = "0.6E"):
    print("\nInitial guess: ", initial_guess)    
    solution, itera, x, fx, m = newtonMethod(f, initial_guess, tolerance, maxSteps)
    
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
        
def partB(f, x0, x1, tolerance, f_str, maxSteps):
    precision = "0.6E"
    
    solution, no_iterations = secant.secant(f, x0, x1, tolerance, maxSteps)
    print("\nNumber of iterations = ", no_iterations)
    print("An estimate of the root is %{0:>10}".format(precision) % (solution))
    
def partC(f, x0, xf, tolerance, f_str, maxSteps):
    precision = "0.6E"
    
    solution, no_iterations = bisection.bisection(f, x0, xf, tolerance, str_precision=precision)
    print("\nNumber of iterations = ", no_iterations)
    print("An estimate of the root is %{0:>10}".format(precision) % (solution))
    
    # Plot graph.
    x = np.linspace(0, 5, 50)
    
    plt.title("Graph of f(x) = " + f_str)
    plt.ylabel("f(x)")
    plt.xlabel("x") 
    plt.grid()
    plt.plot(x, f(x))  # plot f(x)
    plt.plot(x, np.zeros(len(x)), '--', c="gray")  # plot x-axis  
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
        
def partD(f, a, b, tolerance, maxSteps):
    precision = "0.6E"
    solution, iterations, method, x, fx, intervals = newtonBisection(f, a, b, tolerance, maxSteps)

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
 
def pm2(x0, h=0.0125):
    x, steps = newtonMethod_multivariate(x0, h)
    
    print("\n\nNumber of steps: ", steps)
    print("x =", x)
    print("fx = ", f(x[0], x[1], x[2]))
    print("gx = ", g(x[0], x[1], x[2]))
    print("mx = ", m(x[0], x[1], x[2]))
    
def main():
    
    ########################################################################
    ####### Problem 1
    ########################################################################
    tolerance = 1e-6
    precision = "0.6E"
    f_str = "tanh(x)"
    
    # Part a.1; Initial Guess: 1.08
    #partA(1.08, f_str, tolerance, 20)
    
    # Part a.2; initial Guess: 1.09
    #partA(1.09, f_str, tolerance, 20)
    
    # Part b.1
    #partB(f_pm1, 1.08, 1.09, tolerance, 20)
    
    # Part b.2
    #partB(f_pm1, 1.09, 1.10, tolerance, 20)
    
    # Part b.3
    #partB(f_pm1, 1, 2.3, tolerance, 20)
    
    # Part b.4
    #partB(f_pm1, 1, 2.4, tolerance, 20)
    
    # Part c
    #partC(f_pm1, -5, 3, tolerance, f_str, 20)
    
    # Part d
    #partD(f_pm1, -10, 5, tolerance, 20)
    
    ########################################################################
    ####### Problem 2
    ########################################################################
    '''
    x0 = np.array([2, 0, 2])
    pm2(x0)
    '''
    
    return 0

if __name__ == "__main__":
    main()