import numpy as np
from scipy.integrate import quad


def simpson(f, a, b, n):
    # calculating step size
    h = float(b - a) / n
    
    # Finding sum 
    s = f(a) + f(b)
    
    for i in range(1,n):
        k = a + i*h
        
        # if i!=3k
        if i%3 == 0:
            for i in np.arange(1, (n/3)):
                s += 2 * f(k)
        else:
            s += 3 * f(k)
    
    # Finding approximate value
    s *= 3 * h / 8
    
    return s
        
        

#first test case
f = lambda x: x**3+x**2+x
F = lambda x: x**3+x**2+x

#second test case
f1 = lambda x: x**3
F1 = lambda x: x**3

#third test case
f2 = lambda x: 2*x**3+10*x**2
F2 = lambda x: 2*x**3+10*x**2

#intervals
a, b = 0.0, 0.5

#exact solutions for each equation 
exact = quad(F, a, b)

exact1 = quad(F1, a, b)

exact2 = quad(F2, a, b)


#printing exact value, simpsons approximation and error

n = 3
errsimp = abs(exact[0] - simpson(f,a,b,n)) 
print(f"# of intervals={n}, Exact value: {exact[0]}, Simpson's approximation = {simpson(f,a,b,n)}, Simpson's error  = {errsimp}")

print("\n")

errsimp = abs(exact1[0] - simpson(f1,a,b,n))
print(f"# of intervals={n}, Exact value: {exact1[0]}, Simpson's approximation = {round(simpson(f1,a,b,n), 16)}, Simpson's error  = {round(errsimp, 16)}")

print("\n")

errsimp = abs(exact2[0] - simpson(f2,a,b,n))
print(f"# of intervals={n}, Exact value: {exact2[0]}, Simpson's approximation = {round(simpson(f2,a,b,n), 16)}, Simpson's error  = {round(errsimp, 15)}")

print("\n")

######
#Numerical Experiment for Order of Accuarcy

f3 = lambda x: x**4
F3 = lambda x: x**4

a, b = 0.0, 2

exact3 = quad(F3, a, b)

errsimp = abs(exact3[0] - simpson(f3,a,b,n)) 
print(f"# of intervals={n}, Exact value: {exact3[0]}, Simpson's approximation = {simpson(f3,a,b,n)}, Simpson's error  = {round(errsimp, 9)}")
    
print("Which shows Simpson's Rule is accurate up to degree 4.")

print("Simpson's rule has order of accuracy: O(h^4)")

