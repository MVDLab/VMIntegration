"""Scipy optimization evaluation

Compare optimize.minimize(method="SLSQP") vs. optimize.least_squares()

simple problem: 

find min of f(x) = x1 * x4 (x1 + x2 + x3) + x3

contraints
inequality
x1*x2*x3*x4 >= 25

equality
x1^2+x2^2+x3^2+x4^2 = 40

bounds
1 <= x1,x2,x3,x4 <= 5

init_guess = (1,5,5,1)

"""

# python standard
import functools

# commmon 3rd party  
import numpy as np
import scipy.optimize



c = 0

def objective(x):
    """ function to minimize """
    global c

    print(f"objective call #{c}: \t{x}")
    c += 1 
    
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


def constraintA(x):
    """ inequality constraint expected to return value >= 0 when satisfied """

    # equivelent to x[0]*x[1]*x[2]*x[3]
    return functools.reduce(lambda a,b: a*b, x) - 25


def constraintB(x):
    """ equality constraint expected to return value == 0 when satisfied """

    # equivelent to x[0]**2+x[1]**2+x[2]**2+x[3]**2 
    return sum(x**2) - 40



if __name__=="__main__":

    x0 = [1, 5, 5, 1]
    bnds = ((1, 5),)* 4

    res = scipy.optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        )

    print(res)
    c = 0

    res = scipy.optimize.least_squares(
        objective,
        x0,
        bounds=list(zip(*bnds)),
        )

    print(res)
    
    