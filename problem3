import numpy as np
from scipy.integrate import quad

class integration:
    
    def simpson(f, a, b, n):
        h = float(b-a)/n
        k = 0.0
        x = a + h
        for i in np.arange(1, n/2 + 1):
            k += 4*f(x)
            x += 2*h

        x = a + 2*h
        for i in np.arange(1, n/2):
            k += 2*f(x)
            x += 2*h
        return (h/3)*(f(a)+f(b)+k)


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
a, b = 0.0, 2

#exact solutions for each equation 
exact = quad(F, a, b)

exact1 = quad(F1, a, b)

exact2 = quad(F2, a, b)


#printing exact value, simpsons approximation and error
nvalues = [2**i for i in range(1,5)]
for n in nvalues:
    errsimp = abs(exact[0] - integration.simpson(f,a,b,n)) 
    print(f"# of intervals={n}, Exact value: {exact[0]}, Simpson's approximation = {integration.simpson(f,a,b,n)}, Simpson's error  = {errsimp}")
    
print("\n")


for n in nvalues:
    errsimp = abs(exact[0] - integration.simpson(f1,a,b,n))
    print(f"# of intervals={n}, Exact value: {exact1[0]}, Simpson's approximation = {integration.simpson(f1,a,b,n)}, Simpson's error  = {errsimp}")

print("\n")

for n in nvalues:
    errsimp = abs(exact2[0] - integration.simpson(f2,a,b,n))
    print(f"# of intervals={n}, Exact value: {exact2[0]}, Simpson's approximation = {integration.simpson(f2,a,b,n)}, Simpson's error  = {errsimp}")

print("\n")
