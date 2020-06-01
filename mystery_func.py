from sympy import Symbol, diff, lambdify
import numpy as np

def get_mystery_function():
    x = Symbol('x')
    mystery_func = (x-np.random.randint(-5000, 5000))**2 + np.random.randint(-500, 500)*x + np.random.randint(-5000, 5000)
    mystery_func_prime = mystery_func.diff(x)
    mystery_func = lambdify(x, mystery_func)
    mystery_func_prime = lambdify(x, mystery_func_prime)

    return mystery_func, mystery_func_prime