from sympy import Symbol, diff, lambdify
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
    mystery_func = x**4 + 5 * x**3 - 3 * x**2 - 20*x + 50
    mystery_func_prime = mystery_func.diff(x)
    mystery_func = lambdify(x, mystery_func)
    mystery_func_prime = lambdify(x, mystery_func_prime)

    return mystery_func, mystery_func_prime

def get_actual_minimum(mystery_func):
    fit = minimize(mystery_func, x0 = 0)
    print(f"Minimum of {fit['fun']:.2f} at x = {fit['x'][0]:.2f}")
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

def plot_guesses(guesses, function, title = None):
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.scatter(guesses, [function(guess) for guess in guesses], color = 'r')
    x = np.linspace(np.min(guesses), np.max(guesses))
    ax.plot(x, function(x))
    ax.set_xlabel('x', size = 20)
    ax.set_ylabel('f(x)', size = 20)
    if title:
        ax.set_title(title)
    return ax

def plot_near_min(guesses, function, actual_min, x_min, x_range = 100, y_range=10000):
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.scatter(guesses, [function(guess) for guess in guesses], color = 'r')
    x = np.linspace(np.min(guesses), np.max(guesses), 1000000)
    ax.plot(x, function(x))
    ax.set_xlim(x_min - x_range, x_min + x_range)
    ax.set_ylim(actual_min-y_range/100, actual_min + y_range)
    return ax