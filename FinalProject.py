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
from mpmath import *
import bisection
import secant
import newton
import center_difference as cd
import gaussian_elimination_w_pp as pp
from scipy.integrate import quad
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


def negativeFArrow(xn):
    return np.matrix([[-f(xn[0], xn[1], xn[2])], [-g(xn[0], xn[1], xn[2])], [-m(xn[0], xn[1], xn[2])]])


def gaussianElimination(A, b):
    return pp.part_piv_ge(A, b)


def f_pm1(x):
    return np.tanh(x)


def f(x, y, z):
    return x ** 2 + y ** 2 + z ** 2 - 10


def g(x, y, z):
    return x + 2 * y - 2


def m(x, y, z):
    return x + 3 * z - 9


def partA(f, initial_guess, f_str, tolerance, maxSteps, precision="0.6E"):
    print("\nInitial guess: ", initial_guess)
    solution, itera, x, fx, m = newtonMethod(f, initial_guess, tolerance, maxSteps)

    x_vals = np.linspace(-10, 10)
    for i in range(0, len(x)):
        abline_values = [m[i] * (n - x[i]) + fx[i] for n in x_vals]
        plt.plot(x_vals, abline_values, '--', label='iteration: {0}'.format(i + 1))

    plt.title("Root Approximation of " + f_str + ",\n with initial guess x = {0}".format(initial_guess))
    plt.plot(np.zeros(len(x_vals)), np.linspace(-5, 5), '--', c="lightgray")
    plt.plot(x_vals, np.zeros(len(x_vals)), '--', c='lightgray')
    plt.plot(x_vals, np.tanh(x_vals))
    plt.ylim(-1, 1)
    plt.legend(loc='best')
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
    # print(df.to_string())

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
    print(df)

    # Plot data
    ts = df['Discharge (cf/s)'].plot(figsize=(14, 5), title='Hoover Damn', xlabel="Time", ylabel="Discharge (cf/s)",
                                     legend=True)
    ts.plot()
    plt.show()

    # Calculate total discharge rate in seconds
    per_second = df.sum(numeric_only=True)[0]

    """
    # Calculate total discharge rate in years - Convert cubic ft/sec --> cubic ft/yr
    # Conversion Rate: 60 secs/min -> 60 mins/hr -> 24 hrs/day -> 365 days/yr
    per_year = per_second * 60 * 60 * 24 * 365

    print(f"\nThe Hoover Dam has discharged {per_year} cubic feet of water in the past "
          f"calendar year.")
    """
    print(f"\nThe Hoover Dam has discharged {per_second} cubic feet of water per second in the past "
          f"calendar year.")


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
    
def main():
    ########################################################################
    ####### Section 1
    ########################################################################
    """
    tolerance = 1e-6
    f_str = "tanh(x)"

    # Part a.1; Initial Guess: 1.08
    partA(f_pm1, 1.08, f_str, tolerance, 20)

    # # Part a.1; Initial Guess: 1.08
    # partA(1.08, f_str, tolerance, 20)
    # Part b.2
    partB(f_pm1, 1.09, 1.10, tolerance, f_str, 20)

    # Part b.3
    partB(f_pm1, 1, 2.3, tolerance, f_str, 20)

    # Part b.4
    partB(f_pm1, 1, 2.4, tolerance, f_str, 20)

    # Part c
    partC(f_pm1, -5, 3, tolerance, f_str, 20)

    # Part d
    partD(f_pm1, -10, 5, tolerance, 20)
    """

    ########################################################################
    ####### Section 2
    ########################################################################
    '''
    x0 = np.array([2, 0, 2])
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
    
    #section4()
    
    
    return 0


if __name__ == "__main__":
    main()
