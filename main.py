from Fit import *

mode = 'sine'  # iris, sine, diabetes, friedman_function

if mode == 'iris':
    run_iris()
elif mode == 'sine':
    run_sine()
elif mode == 'diabetes':
    run_diabetes()
elif mode == 'friedman_function':
    run_friedman_function()
else:
    print(f"Błąd, Nieprawidłowy tryb: {mode}")
