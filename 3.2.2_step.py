import numpy as np

def step_function_num(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function(x):
    y = x > 0
    return y.astype(np.int)

print("step(0) : " + str(step_function_num(0)))
print("step(1) : " + str(step_function_num(1)))

array = np.array([-1.0, 1.0, 2.0])
print("np array")
print(array)
print("np array stepped")
print(step_function(array))
