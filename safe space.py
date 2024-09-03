import numpy as np
size = 50
A = np.triu(np.ones((size, size)))
A = 2*A

b_list = [i for i in range(1,51)]
b = np.array(b_list)




x = np.linalg.solve(A, b)
print(x)
# print(A@x_star)
