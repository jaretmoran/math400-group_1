import sys
import numpy as np
import matplotlib.pyplot as plt

# global variables
tolerance = 1e-6
precision = "0.6E"
functionString = "tanh(x)"
maxSteps = 20
h=0.0125


def f(x):
    return np.tanh(x)


def centerDifference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def newtonsMethod(xN):
    iterationCounter = 0
    # lists used for the graphs
    x = []
    fx = []
    m = []
    print("\nIteration\t\t x value \t\t\t f(x)")
    print(0, "\t\t\t %{0:>10} \t\t %{0:>10}".format(precision) % (xN, f(xN)))
    while abs(f(xN)) > tolerance and iterationCounter < maxSteps:
        print(iterationCounter + 1, "\t\t\t ", end=" ")
        try:
            # data for graphs
            x.append(xN)
            fx.append(f(xN))
            derivativeAtXn = centerDifference(f, xN, h)
            m.append(derivativeAtXn) # append slope of tangent line; slope = to derivative at xN

            # newton's method to get xNPlusOne
            xNPlusOne = xN - f(xN) / derivativeAtXn
        except:
            raise Exception("Division by zero. f'(x) = 0")

        print("%{0:>10} \t\t %{0:>10}".format(precision) % (xNPlusOne, f(xNPlusOne)))
        xN = xNPlusOne # new xN is xNPlusOne's value
        iterationCounter += 1
    return m, x, fx


def secantMethod(x0, x1):
    iterationCounter = 0  # set iteration counter to zero

    xNMinusOne = x0
    xN = x1
    while abs(f(xN)) > tolerance and iterationCounter < maxSteps:
        try:
            # secant method
            xNPlusOne = xN - f(xN) * ((xN - xNMinusOne) / (f(xN) - f(xNMinusOne)))
            xNMinusOne = xN
            xN = xNPlusOne
        except:
            raise Exception("Division by zero. f(xN) - f(xNMinusOne) = 0")

        iterationCounter += 1

    print("Number of iterations = ", iterationCounter)
    print("An estimate of the root is %{0:>10} with f(x) = %{0:>10}".format(precision) % (xN, f(xN)))


def bisectionMethod(initialInterval):
    # initial values of f(a) and f(b)
    a0, b0 = initialInterval[0], initialInterval[1]

    if f(a0) * f(b0) > 0:
        print("Error! f must have different signs at the endpoints. Aborting")
        sys.exit(1)

    print("Initial Interval: ",
          "a = %{0}".format(precision) % (a0),
          " b = %{0}".format(precision) % (b0),
          " f(a) = %{0}".format(precision) % (f(a0)),
          " f(b) = %{0}".format(precision) % (f(b0)), "\n")

    iterationCounter = 0
    aN, bN = a0, b0
    xNPlusOne = None
    while abs(bN - aN) > 2 * tolerance:  # while the size  of interval is larger than the tolerance
        xNPlusOne = float(aN + bN) / 2.0  # midpoint of interval

        print("aN = %{0:>10}".format(precision) % (aN),
              " bN = %{0:>10}".format(precision) % (bN),
              " xNPlusOne = %{0:>10}".format(precision) % (xNPlusOne),
              " f(aN) = %{0:>10}".format(precision) % (f(aN)),
              " f(bN) = %{0:>10}".format(precision) % (f(bN)),
              " f(xNPlusOne) = %{0:>10}".format(precision) % (f(xNPlusOne)))

        if f(xNPlusOne) * f(aN) < 0:  # if different signs with f(aN)
            bN = xNPlusOne # assign midpoint to b
        else:
            aN = xNPlusOne # assign midpoint to a

        iterationCounter += 1

        # found the root
        if f(xNPlusOne) == 0:
            break

    print("Number of iterations = ", iterationCounter)
    print("An estimate of the root is %{0:>10}".format(precision) % (xNPlusOne))

    # plot graph
    x = np.linspace(-10, 10)
    plt.title("Graph of f(x) = " + functionString)
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.grid()
    plt.plot(x, f(x))  # plot f(x)
    plt.plot(x, np.zeros(len(x)), '--', c="gray")  # plot x-axis
    plt.show()


def bisectionWithNewton(initialInterval, s=0.1):
    # lists used for print statements
    x = []
    fx = []
    intervals = []
    method = []

    # initial values
    a, b = initialInterval[0], initialInterval[1]
    xN = a

    if f(a) * f(b) > 0:
        print("Error! f must have different signs at the endpoints. Aborting")
        sys.exit(1)

    iterationCounter = 0
    while abs(f(xN)) > tolerance and iterationCounter < maxSteps:
        intervals.append(abs(b - a))

        # Bisection Step: occurs if approximate root jumps out of interval
        # or interval is greater than s * length of initial interval
        if xN < initialInterval[0] or xN > initialInterval[1] or \
                abs(b - a) > s * (initialInterval[1] - initialInterval[0]):
            xNPlusOne = float(a + b) / 2.0
            method.append("Bisection")

            if f(xNPlusOne) * f(a) < 0:  # if different signs with f(aN)
                b = xNPlusOne  # assign midpoint to b
            else:
                a = xNPlusOne  # assign midpoint to a

            # found the root
            if f(xNPlusOne) == 0:
                break

            xN = xNPlusOne

        # Newton Step
        else:
            method.append("Newton")
            derivativeAtXn = centerDifference(f, xN, h)
            xNPlusOne = xN - f(xN) / derivativeAtXn
            xN = xNPlusOne

        x.append(xN)
        fx.append(f(xN))
        iterationCounter += 1

    # print out table summarizing results
    print("\nInterval Length \t\t Method \t\t\t\t x-value \t\t\t f(x)")
    print("---------------------------------------------------------------------------")
    for i in range(len(x)):
        print("%{0} \t\t".format(precision) % (intervals[i]), end=" ")
        print(method[i].ljust(10, " ") + "\t\t", end=" ")
        print("%{0:>10} \t\t".format(precision) % (x[i]), end=" ")
        print("%{0:>10} \t\t".format(precision) % (fx[i]))
    print(f"\nNumber of iterations: {iterationCounter}")
    print(f"xN: {xN}, f(xN): {f(xN)}")

# --------------- Part A ---------------
# initialGuess = 1.08
# # initialGuess = 1.09
# print("Initial Guess: ", initialGuess)
# m, x, fx = newtonsMethod(initialGuess)
# xValues = np.linspace(-10, 10)
# # for each tangent line, calculate the y values and plot it
# for i in range(len(x)):
#     # point slope formula for tangent line y values
#     tangentLineYValues = [m[i] * (n - x[i]) + fx[i] for n in xValues]
#     plt.plot(xValues, tangentLineYValues, '--', label='iteration: {0}'.format(i + 1))
# plt.title("Root Approximation of " + functionString + ",\n with initial guess x = {0}".format(initialGuess))
# plt.plot(np.zeros(len(xValues)), np.linspace(-5, 5), '--', c="lightgray")
# plt.plot(xValues, np.zeros(len(xValues)), '--', c='lightgray')
# plt.plot(xValues, np.tanh(xValues)) # plots the function tanh
# plt.ylim(-5, 5)
# plt.legend()
# plt.show()
# --------------- Part A ---------------

# --------------- Part B ---------------
# initial guesses
# (i)
# x0 = 1.08
# x1 = 1.09

# (ii)
# x0 = 1.09
# x1 = 1.1

# (iii)
# x0 = 1
# x1 = 2.3

# (iv)
# x0 = 1
# x1 = 2.4
#
# secantMethod(x0, x1)
# --------------- Part B ---------------

# --------------- Part C ---------------
# initialInterval = [-5, 3]
# bisectionMethod(initialInterval)
# --------------- Part C ---------------

# --------------- Part D ---------------
# initialInterval = [-10, 15]
# bisectionWithNewton(initialInterval)
# --------------- Part D ---------------