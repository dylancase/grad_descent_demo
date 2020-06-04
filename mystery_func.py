from sympy import Symbol, diff, lambdify
import numpy as np
from scipy.optimize import minimize

def get_mystery_function():
    x = Symbol('x')
    sign = np.random.choice([1, -1])
    mystery_func = (np.random.randint(1, 20)*x+np.random.randint(1000, 5000)*sign)**2 + np.random.randint(1, 50)*x + np.random.randint(1, 40)
    mystery_func_prime = mystery_func.diff(x)
    mystery_func = lambdify(x, mystery_func)
    mystery_func_prime = lambdify(x, mystery_func_prime)

    return mystery_func, mystery_func_prime

def get_higher_order_mystery_function():
    '''x = Symbol('x')
    sign = np.random.choice([1, -1])
    mystery_func = (np.random.randint(10, 200)*x+np.random.randint(100000, 500000)*sign)**2 + np.random.randint(1, 500)*x + np.random.randint(1, 400)
    mystery_func_prime = mystery_func.diff(x)
    mystery_func = lambdify(x, mystery_func)
    mystery_func_prime = lambdify(x, mystery_func_prime)'''
    x = Symbol('x')
    mystery_func = (x-7)**4 - 5 * x**3 + 5 * x**2 - 6
    mystery_func_prime = mystery_func.diff(x)
    mystery_func = lambdify(x, mystery_func)
    mystery_func_prime = lambdify(x, mystery_func_prime)

    return mystery_func, mystery_func_prime

def get_actual_minimum(mystery_func):
    fit = minimize(mystery_func, x0 = 0)
    print(f"Minimum of {fit['fun']:.2e} at x = {fit['x'][0]:.2f}")
    return fit['fun'], fit['x'][0]

def try_guesses(mystery_func, n_guesses=10, deriv = None):
    guesses = []
    our_minimum = None
    for _ in range(n_guesses):
        x = float(input("Enter a guess: "))
        current_val = mystery_func(x)
        if deriv:
            current_derivative = deriv(x)
            print(f'Current value: {current_val:.2e}, current derivative: {current_derivative:.2e}')
        else:
            print(f'{current_val:e}')
        guesses.append(x)
        if not our_minimum or current_val < our_minimum:
            our_minimum = current_val
    return our_minimum, guesses

def get_multivariable_function():
    pass