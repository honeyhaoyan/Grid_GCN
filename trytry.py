import numpy as np
a = np.array([[1,2,3]])
print(a.shape)
print(a)
b = np.repeat(a,3,axis = 0)
print(b)