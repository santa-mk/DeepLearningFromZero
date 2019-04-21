import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array(([5, 6], [7, 8]))

C = np.dot(A, B)
D = np.dot(B, A)

print(C)
print(D)
