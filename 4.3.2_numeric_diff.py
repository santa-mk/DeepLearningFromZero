import numpy as np
import matplotlib.pylab as plt

def numeric_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def plot_tangent(function, target_x, x):
    katamuki = numeric_diff(function, target_x)
    target_y = function(target_x)
    seppen = target_y - katamuki * target_x
    y = katamuki * x + seppen
    
    ## dot and lines
    plt.plot(target_x, target_y, marker='.')
    plt.vlines(target_x, -1, target_y, "m", linestyle=":")
    plt.hlines(target_y, 0, target_x, "m", linestyle=":")
    plt.plot(x, y)

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

## diffs
# diffs = numeric_diff(function_1, x)
# plt.plot(x, diffs)
# plt.show()

## numeric diff 5
plot_tangent(function_1, 5, x)
plot_tangent(function_1, 10, x)
plt.show()