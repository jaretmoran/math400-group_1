#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:33:03 2021

@author: hboateng
"""
import numpy as np
from mpmath import *
import center_difference
                    
def newton(f,fp,x0,delta,Nmax):
    """
    
    Parameters
    ----------
    f     : function whose root we want to find
    fp    : first derivative of f 
    x0    : Initial guess for the root of f
    delta : The tolerance/accuracy we desire
    Nmax  : Maximum number of iterations to be performed

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    x0 : The approximation to the root
        DESCRIPTION.
    iter_counter : Number of iterations it takes to satisfy tolerance

    """
    iter_counter = 0  # set iteration counter to zero
    
    while abs(f(x0)) > delta and iter_counter < Nmax:
        try:
            x0 = x0 - f(x0)/fp(x0)  # Newton's method
        except:
            raise Exception("Division by zero. f'(x) = 0")
        
        print("value = ", x0)
        iter_counter +=1 
          
    return x0, iter_counter

def f(x):
    return tanh(x) #x**2-3 #(5.0-x)*math.exp(x)-5

def newtonStep(f, x0, h=0.0125):
    return x0 - f(x0)/center_difference.center_diff(f, x0, h)

def fp(x):
    return (sech(x))**2

def main():

    solution, no_iterations = newton(f,fp,1.09,1e-6,5)
    print("Number of iterations = ",no_iterations)  
    print("An estimate of the root is ",solution)

    return 0

if __name__ == "__main__":
    main()        