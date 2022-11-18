import numpy as np


# performs center difference calculation specific to a variable
def centerDifferenceMultivariate(function, x, var, h=0.0125):
    result = 0

    if var == 'x':
        result = function(x[0] + h, x[1], x[2]) - function(x[0] - h, x[1], x[2])
    elif var == 'y':
        result = function(x[0], x[1] + h, x[2]) - function(x[0], x[1] - h, x[2])
    elif var == 'z':
        result = function(x[0], x[1], x[2] + h) - function(x[0], x[1], x[2] - h)

    result /= (2 * h)

    return result


# function f from the problem statement
def f(x, y, z):
    return x ** 2 + y ** 2 + z ** 2 - 10


# function g from the problem statement
def g(x, y, z):
    return x + 2 * y - 2


# function h from the problem statement (named m to avoid confusion with h variable)
def m(x, y, z):
    return x + 3 * z - 9


def jacobian(xn, h=0.0125):
    # using center difference as numerical differentiation to compute the Jacobian matrix
    return np.matrix([
        [centerDifferenceMultivariate(f, xn, 'x', h), centerDifferenceMultivariate(f, xn, 'y', h),
         centerDifferenceMultivariate(f, xn, 'z', h)],
        [centerDifferenceMultivariate(g, xn, 'x', h), centerDifferenceMultivariate(g, xn, 'y', h),
         centerDifferenceMultivariate(g, xn, 'z', h)],
        [centerDifferenceMultivariate(m, xn, 'x', h), centerDifferenceMultivariate(m, xn, 'y', h),
         centerDifferenceMultivariate(m, xn, 'z', h)]
    ])


def negativeFArrow(xn):
    return np.matrix([[-f(xn[0], xn[1], xn[2])], [-g(xn[0], xn[1], xn[2])], [-m(xn[0], xn[1], xn[2])]])


# A represents Jacobian, b represents negativeFArrow; the reuslt is
def partialPivotingGaussianElimination(A, b):
    nrow = A.shape[0]
    ncol = A.shape[1]

    # Gaussian Elimination
    for i in range(ncol - 1):  # loop over columns
        col_i = abs(A[i:nrow, i])  # isolate column i
        pivot = max(col_i)
        t, = np.where(col_i == pivot)[0]  # find pivot i and its index
        t = t + i

        # interchanging i^th row with pivot row
        temp = A[i, :].copy()
        tb = b[i].copy()
        A[i, :] = A[t, :]
        b[i] = b[t]
        A[t, :] = temp
        b[t] = tb

        aii = A[i, i]

        # perform the elimination step
        for j in range(i + 1, nrow):  # loop over rows below column i
            m = -A[j, i] / aii  # multiplier
            A[j, i] = 0
            A[j, i + 1:nrow] = A[j, i + 1:nrow] + m * A[i, i + 1:nrow]
            b[j] = b[j] + m * b[i]

    # back substitution step below

    # initialize vector of zeros (this will eventually be result of back substitution)
    x = np.zeros((nrow))

    nrow -= 1
    x[nrow] = b[nrow] / A[nrow, nrow]

    # calculate values of solution x
    for i in range(nrow - 1, -1, -1):
        dot = A[i, i + 1:nrow + 1] @ x[i + 1:nrow + 1]
        x[i] = (b[i] - dot) / A[i, i]

    return x


def newtonsMethodMultivariate(xN):
    steps = 0

    # print initial value of xN
    print(f"\nx{steps} = {xN}")

    xNPlusOne = None

    # perform each step of Newton's method, utilizing Gaussian Elimination
    # and partial pivoting
    while True:
        steps += 1
        xNPlusOne = xN + partialPivotingGaussianElimination(jacobian(xN), negativeFArrow(xN))

        print(f"x{steps} = {xNPlusOne}")

        # test if maximum norm of the difference between successive iterates
        # is less than 5 * 10^-6 (steps == 100 is a precaution against invalid input)
        # if np.linalg.norm(xn - xn_previous, np.inf) < 5e-6 or steps == 100:
        if np.linalg.norm(xNPlusOne - xN, np.inf) < 5e-6 or steps == 100:
            break

        xN = xNPlusOne

    return xNPlusOne, steps


x0 = np.array([2, 0, 2])  # initial vector
x, stepCount = newtonsMethodMultivariate(x0)

# summary of results
print("\n\nNumber of steps: ", stepCount)
print("(x, y, z) =", x)

# should be very close to 0 as finding root
print("f(x, y, z) = ", f(x[0], x[1], x[2]))
print("g(x, y, z) = ", g(x[0], x[1], x[2]))
print("h(x, y, z) = ", m(x[0], x[1], x[2]))