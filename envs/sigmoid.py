import numpy as np
import matplotlib.pyplot as plt

mg = 2.7468
max_thrust = 15.7837

def func(x):
    tanh1 = mg * (1 + np.tanh(3 * x + 2)) / 2
    tanh2 = (max_thrust - mg) * (1 + np.tanh(3 * x - 2)) / 2
    return tanh1 + tanh2

# Compute values
x_values = np.linspace(-2, 2, 200)
func_values = func(x_values)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, func_values, color='b')
plt.axhline(y=max_thrust, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
