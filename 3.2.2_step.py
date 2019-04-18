def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print("step(0) : " + str(step_function(0)))
print("step(1) : " + str(step_function(1)))
