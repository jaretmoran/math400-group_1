#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:37:38 2022
@author: Team 1
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpmath as math
import bisection
import secant
import newton
import center_difference as cd
import gaussian_elimination_w_pp as pp
from scipy.integrate import quad
from scipy.integrate import simpson
# import scipy as sp
import pandas as pd


def newtonMethod(f, x0, delta, maxIterations):
    precision = "0.6E"
    print("\niteration\t\t x value \t\t\t f(x)")

    # Lists used for the graphs
    x = []
    fx = []
    m = []

    iter_counter = 0
    h = 0.0125

    while abs(f(x0)) > delta and iter_counter < maxIterations:
        print(iter_counter + 1, "\t\t\t ", end=" ")
        try:
            x0 = newton.newtonStep(f, x0, h)
            
            
            # Data for graphs
            x.append(x0)
            fx.append(f(x0))
            m.append(cd.center_diff(f, x0, h))

        except:
            raise Exception("Division by zero. f'(x) = 0")

        print("%{0:>10} \t\t %{0:>10}".format(precision) % (x0, f(x0)))
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

    if fa * fb > 0:
        print("Error! f must have different signs at the endpoints. Aborting")
        sys.exit(1)

    while (abs(f(x0)) > delta and iter_counter < maxIterations):
        interval.append(b - a)

        # Bisection Step
        if ((a >= x0 and x0 >= b) or (abs(b - a) > s * (b_initial - a_initial))):
            x0 = float(a + b) / 2.0
            method.append("Bisection")

            fx0 = f(x0)
            if fx0 * fb < 0:
                a = x0;
                fa = fx0
            else:
                b = x0;
                fb = fx0

        # Newton Step
        else:
            x0 = newton.newtonStep(f, x0, h)
            method.append("Newton")

        x.append(x0)
        fx.append(f(x0))
        iter_counter += 1

    return x0, iter_counter, method, x, fx, interval


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
        [cd.center_diff_multivariate(f, xn, 'x', h), cd.center_diff_multivariate(f, xn, 'y', h),
         cd.center_diff_multivariate(f, xn, 'z', h)],
        [cd.center_diff_multivariate(g, xn, 'x', h), cd.center_diff_multivariate(g, xn, 'y', h),
         cd.center_diff_multivariate(g, xn, 'z', h)],
        [cd.center_diff_multivariate(m, xn, 'x', h), cd.center_diff_multivariate(m, xn, 'y', h),
         cd.center_diff_multivariate(m, xn, 'z', h)]
    ])

'''
def jacobian(xn):
    return np.matrix([ [2*xn[0], 2*xn[1], 2*xn[2] ], [ 2*xn[0], 0, 2*xn[2] ], [ 2*xn[0], 2*xn[1], -4] ])
'''
def negativeFArrow(xn):
    return np.matrix([[-f(xn[0], xn[1], xn[2])], [-g(xn[0], xn[1], xn[2])], [-m(xn[0], xn[1], xn[2])]])


def gaussianElimination(A, b):
    return pp.part_piv_ge(A, b)


def f_pm1(x):
    return np.tanh(x)


def f(x, y, z):
    return x**2 + y**2 + z**2 -1 

def g(x, y, z):
    return x**2 + z**2 - 0.25

def m(x, y, z):
    return x**2 + y**2 - 4*z


def partA(f, initial_guess, f_str, tolerance, maxSteps, precision="0.6E"):
    print("\nInitial guess: ", initial_guess)
    solution, iterations, x, fx, m = newtonMethod(f, initial_guess, tolerance, maxSteps)

    # Setup figure and number of subplots
    nrows = int(math.ceil(iterations/2))
    ncols = 2
    figSize = (15, 13)    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figSize)
    
    x_vals = y_vals = np.linspace(-10, 10)
    tangent_index = 0
    for i in range(0, nrows):
        for j in range(0, ncols):
            if tangent_index < iterations:
                
                # Compute tangent line
                tangentLine = [m[tangent_index] * (n - x[tangent_index]) + fx[tangent_index] for n in x_vals]
                
                # Plot function and tangent line on subplot
                axs[i][j].plot(x_vals, tangentLine, '--', label='tangent line')
                axs[i][j].plot(x_vals, np.tanh(x_vals), label='f(x) = tanh(x)')
                
                # Subplot settings
                axs[i][j].set_xlim(-3,3)
                axs[i][j].set_ylim(-1,1)
                axs[i][j].set_title("Iteration: {0}; x={1}".format(tangent_index + 1, x[tangent_index]))
                axs[i][j].legend(loc="best")
                
                # plot x-axis on subplot
                axs[i][j].plot(x_vals, np.zeros(len(x_vals)), '--', color='lightgray')
                axs[i][j].text(2.85, -.25, 'x')
                
                # plot y-axis on subplot
                axs[i][j].plot(np.zeros(len(y_vals)), y_vals, '--', color='lightgray')
                axs[i][j].text(-0.15, 0.75, 'y')
                
            tangent_index += 1

    # Super plot Settings (main plot)
    fig.suptitle("Root Approximation using Newton's Method")
    fig.tight_layout()
    plt.show()

def partB(f, x0, x1, tolerance, f_str, maxSteps=100):
    precision = "0.6E"

    solution, no_iterations = secant.secant(f, x0, x1, tolerance, maxSteps)
    print("\nNumber of iterations = ", no_iterations)
    print("An estimate of the root is %{0:>10}".format(precision) % (solution))


def partC(f, x0, xf, tolerance, f_str, maxSteps=100):
    precision = "0.6E"

    solution, no_iterations = bisection.bisection(f, x0, xf, tolerance, str_precision=precision)
    print("\nNumber of iterations = ", no_iterations)
    print("An estimate of the root is %{0:>10}".format(precision) % (solution))

def partD(f, a, b, tolerance, maxSteps=100):
    precision = "0.6E"
    solution, iterations, method, x, fx, intervals = newtonBisection(f, a, b, tolerance, maxSteps)

    print("\nInterval [-10, 15]")
    print("\nInterval Length \t\t Method \t\t\t\t x-value \t\t\t f(x)")
    print("---------------------------------------------------------------------------")

    for i in range(0, len(x)):
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


def section4():
    # Grab only the data we want - skip the headers and intro
    df = pd.read_csv('hoover_dam_data_past_year.csv', header=None, parse_dates=[0], skiprows=29)
    # print(df.to_string())  # DEBUG

    date_flow = ''
    # Go through parsed data
    for line in range(len(df)):
        # Capture date and flow only
        data = (df.loc[line].str[14:-1].values[0])
        # Split line at tab character '\t'
        columns = data.split('\t')
        # Join each date and flow value into a row separated by comma
        rows = ','.join([columns[0], (columns[1])])
        date_flow += rows + '\n'

    # Encode string to byte type for txt file creation in next cell
    date_flow = date_flow.encode()

    # Grab USGS station number only
    station_number = df.loc[0].str[6:12].values[0]
    # Form file name for txt file
    file_name = 'USGS_Data_for_' + station_number + '.txt'
    # Write data to txt file
    with open(file_name, 'wb') as text:
        text.write(date_flow)

    # Read txt file and assign column names
    columns = ['Date', 'Discharge (cf/s)']
    df = pd.read_csv(file_name, header=None, names=columns, parse_dates=[0])
    df = df.reset_index()

    # Plot data
    ts = df['Discharge (cf/s)'].plot(figsize=(14, 5), title='Hoover Damn', xlabel="Time (s)", ylabel="Discharge (cf/s)",
                                     legend=True)
    ts.plot()
    plt.show()

    # Calculate total discharge
    water_data = []
    for index, row in df.iterrows():
        # print("Cf/s value is: ", df.loc[0][1])  # DEBUG
        # Convert sec --> min --> hour --> day
        water_data.append(row['Discharge (cf/s)'] * 60 * 60 * 24)

    # Set up params for Simpson's rule
    n = 304
    x = np.linspace(1, 305, 305, 305)
    # Using built-in scipy library Simpson's function
    result = simpson(water_data, x)
    actual = sum(water_data)
    # print(result)
    # print(actual)
    # print(len(water_data))  # DEBUG

    print(f"\nThe Hoover Dam has discharged {result} cubic feet of water in the past calendar year.")
    print(f"There is an error/difference of {actual - result} cubic feet when calculating the total amount of water"
          f" via the actual sum of the data versus the result using Simpson's Method.")

'''
def simpson(f, a, b, n):
    h = float(b - a) / n
    k = 0.0
    x = a + h
    for i in np.arange(1, n / 2 + 1):
        k += 4 * f(x)
        x += 2 * h

    x = a + 2 * h
    for i in np.arange(1, n / 2):
        k += 2 * f(x)
        x += 2 * h
    return (h / 3) * (f(a) + f(b) + k)
 ''' 


# NOT USED: our own Simpson's function that takes in the dataset as a parameter
def simpsons_with_data(data, a, b, n):
    h = float(b - a) / n
    print("h = ", h)
    accumulator = 0
    
    print(data[0])
    for i in range(1, len(data) - 1):
        if i % 2 == 0:
            print(data[i])
            accumulator += 2 * data[i]
        else:
            print(data[i])
            accumulator += 4 * data[i]
    
    print( data[len(data) - 1] )
    return (h / 3) * ( data[0] + accumulator + data[len(data) - 1] )
    
    pass

def main():
    ########################################################################
    ####### Section 1
    ########################################################################

    tolerance = 1e-6
    f_str = "tanh(x)"

    # Part a.1; Initial Guess: 1.08
    #partA(f_pm1, 1.08, f_str, tolerance, 20)

    # Part a.2; initial Guess: 1.09
    #partA(f_pm1, 1.09, f_str, tolerance, 20)

    # Part b.1
    #partB(f_pm1, 1.08, 1.09, tolerance, f_str, 20)

    # Part b.2
    #partB(f_pm1, 1.09, 1.10, tolerance, f_str, 20)

    # Part b.3
    #partB(f_pm1, 1, 2.3, tolerance, f_str, 20)

    # Part b.4
    #partB(f_pm1, 1, 2.4, tolerance, f_str, 20)

    # Part c
    #partC(f_pm1, -5, 3, tolerance, f_str, 20)

    # Part d
    #partD(f_pm1, -10, 5, tolerance, 20)
    

    ########################################################################
    ####### Section 2
    ########################################################################
    '''
    x0 = np.array([1, 1, 1])
    pm2(x0)
    '''

    #######################################################################
    ####### Problem 3
    #######################################################################
    #first test case
    '''
    f = lambda x: x**3+x**2+x
    F = lambda x: x**3+x**2+x

    #second test case
    f1 = lambda x: x**3
    F1 = lambda x: x**3

    #third test case
    f2 = lambda x: 2*x**3+10*x**2
    F2 = lambda x: 2*x**3+10*x**2

    #intervals
    a, b = 0.0, 2

    #exact solutions for each equation 
    exact = quad(F, a, b)

    exact1 = quad(F1, a, b)

    exact2 = quad(F2, a, b)


    #printing exact value, simpsons approximation and error
    nvalues = [2**i for i in range(1,5)]
    for n in nvalues:
        errsimp = abs(exact[0] - simpson(f,a,b,n)) 
        print(f"# of intervals={n}, Exact value: {exact[0]}, Simpson's approximation = {simpson(f,a,b,n)}, Simpson's error  = {errsimp}")

    print("\n")


    for n in nvalues:
        errsimp = abs(exact1[0] - simpson(f1,a,b,n))
        print(f"# of intervals={n}, Exact value: {exact1[0]}, Simpson's approximation = {simpson(f1,a,b,n)}, Simpson's error  = {errsimp}")

    print("\n")

    for n in nvalues:
        errsimp = abs(exact2[0] - simpson(f2,a,b,n))
        print(f"# of intervals={n}, Exact value: {exact2[0]}, Simpson's approximation = {simpson(f2,a,b,n)}, Simpson's error  = {errsimp}")

    print("\n")
    '''

    #######################################################################
    ####### Problem 4
    #######################################################################
    #data = [34, 32, 29, 33, 37, 40, 41, 36, 38, 39]
    #x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    #test = simpsons_with_data(data, 0, 90, 10)
    #sp = simpson(data, x)
    #print("Simpson's approx: ", test)
    #print("Scipy: ", sp)
    section4()
    
    
    return 0


if __name__ == "__main__":
    main()
