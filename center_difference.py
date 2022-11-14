#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:35:59 2022

@author: cristobalpadilla
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:08:29 2022

@author: hboateng

This program provides an example of truncation and round-off error
using a forward difference approximation
"""
from matplotlib import pyplot as plt
import numpy as np

# Forward difference approximation
def center_diff(f, x, h=0.0125):
    """
    Parameters
    ----------
    f : The function to be differentiated.
        
    x : TYPE: float
        The point where the derivative is needed
    h : TYPE: float
        The step/discretization size (can be a list).

    Returns
    -------
    fp : Approximation to f'(x) at x (can be a list).

    """
    return (f(x+h) - f(x-h))/(2*h)

def center_diff_multivariate(function, x, var, h = 0.0125):
    result = 0
    
    if var == 'x':
        result = (function(x[0] + h, x[1], x[2]) - function(x[0] - h, x[1], x[2])) / (2 * h)
        
    elif var == 'y':
        result = (function(x[0], x[1] + h, x[2]) - function(x[0], x[1] - h, x[2])) / (2 * h)
        
    elif var == 'z':
        result = (function(x[0], x[1], x[2] + h) - function(x[0], x[1], x[2] - h)) / (2 * h)
        
    return result


def main():
    # Define the function to be differentiated and its derivative
    f = lambda x : np.cos(x); fp = lambda x: -np.sin(x)

    # The point where the derivative is needed
    x = np.pi / 4;
    exact_deriv_value = fp(x)

    #The discretization sizes
    #h = 1/2**np.linspace(1,6,6)
    h = np.array([2**-n for n in range(1,7)])


    # Find the forward difference approximation to the derivative f'(x) for
    # different h values
    fpappx = center_diff(f,x,h)
    error  = exact_deriv_value - fpappx
    abserror = abs(error)
    eh     = error/h;
    eh2    = error/h**2;
    
    plt.loglog(h,abserror,'o-')
    plt.xlabel('h (step size)')
    plt.ylabel('error')
    plt.show()
    
    #print('h ','error ', 'error/h')

    print("\n\t\t\t\t\t\t\t\t\t Center Difference\n")
    print("\t%s\t\t \t\t%s\t\t \t%s\t\t \t%s \t\t \t\t%s \n" %("h", "approx", "error", "error/h", "error/h^2"))
    for i in range(0,len(h)):
        #print(f" {h[i]}, {error[i]}, {eh[i]}")
        print("%12.8e\t %12.8e\t %12.8e\t %12.8e \t %12.8e\n" %(h[i], fpappx[i], error[i], eh[i], eh2[i]))
        
        print("Exact value = ", exact_deriv_value)
        
    return 0

if __name__ == "__main__":
    main()