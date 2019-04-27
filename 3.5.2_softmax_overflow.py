import numpy as np

def softmax_overflow(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def softmax(a):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max) ## care overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1010, 1000, 990])

print("overflow")
y = softmax_overflow(a)
print(y)

print("no overflow")
y2 = softmax(a)
print(y2)