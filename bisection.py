#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:32:06 2021

@author: hboateng
bisection method
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

def bisection(f, a, b, delta, str_precision="0.6E", return_x_list=False):
    """
    bisection method for finding the root of a 
    function f(x)

    Parameters
    ----------
    f : The function whose root we want to find.
    a : left hand side of initial interval
    b : right hand side of initial interval
    delta : error tolerance
    str_precision: Numerical precision print format.
    return_x_list : If True returns a list of
        all estimates during the iteration. 
        The default is False.

    Returns
    -------
    The estimate of the root and the number of
    iterations

    """ 
    
    # Initial values of f(a) and f(b)
    fa = f(a);     
    fb = f(b);
    
    if fa*fb > 0:
        print("Error! f must have different signs at the endpoints. Aborting")
        sys.exit(1)
    
    print("\ninitial interval: ",
          "a = %{0}".format(str_precision) % (a),
          " b = %{0}".format(str_precision) % (b),
          " f(a) = %{0}".format(str_precision) % (fa),
          " f(b) = %{0}".format(str_precision) % (fb), "\n")
    
    iteration_counter = 0
    if return_x_list:
        x_list=[]
        
    while(abs(b-a) >  2*delta):   # while the size  of interval is 
                                  # larger  than the tolerance
        c = float(b+a)/2.0        # set c to the midpoint of the interval
 
        fc = f(c)                 # calculate the value of f at c
        
        if fc*fb < 0:             # if f(c) and f(b) have different signs
            a = c; fa = fc        # assign midpoint to a
        else:         
            b = c; fb = fc        # assign midpoint to b
            
        print("a = %{0:>10}".format(str_precision) % (a), 
              " b = %{0:>10}".format(str_precision) % (b), 
              " c = %{0:>10}".format(str_precision) % (c), 
              " f(a) = %{0:>10}".format(str_precision) % (fa), 
              " f(b) = %{0:>10}".format(str_precision) % (fb), 
              " f(c) = %{0:>10}".format(str_precision) % (fc))
            
        iteration_counter += 1
        if return_x_list:
            x_list.append(c)
            
    if return_x_list:
        return x_list, iteration_counter
    else:
        return c, iteration_counter
        
    
def f(x):
    '''
    Returns the value of the function evaluated at x.

    Parameters
    ----------
    x : Number
        Function input.

    Returns
    -------
    Number
        Value of f(x) at x.

    '''
    return   np.tanh(x) #x**2-3 #(5.0-x)*math.exp(x)-5


def main():
    
    precision = ".6E"   # Output precision format
    
    # Interval end points
    x0 = -5
    xf = 3    

    solution, no_iterations = bisection(f, x0, xf,1e-6, str_precision=precision)
    print("\nNumber of iterations = ",no_iterations)
    print("An estimate of the root is %{0}".format(precision) % (solution))
    
    # Plot graph.
    x = np.linspace(0, 5, 50)
    y = f(x)
    
    plt.title("Graph of f(x) = $(5-x)e^x-5$")
    plt.ylabel("f(x)")
    plt.xlabel("x") 
    plt.grid()
    plt.plot(x, y)  # plot f(x)
    plt.plot(x, np.zeros(len(x)), '--', c="gray")  # plot x-axis  
    
    
    return 0

if __name__ == "__main__":
    main()    