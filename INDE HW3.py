import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import random

matdata = scipy.io.loadmat(r"C:\Users\peter\Downloads\LogReg_data_2024.mat")
A = matdata['A0']
b = matdata['b']
A_other = matdata['A']

m = len(b)
alpha = 1
Beta = 0.8
alpha = alpha * Beta
c_1 = 0.1
w_0 = np.zeros([10,1])
c_0 = np.zeros([1,1])
x = np.concatenate((w_0,c_0))

def stochGrad(b,m,x,s):
    w = x[:-1]
    c = x[-1]

    input_numbers = range(len(b))
    stochIdxs = random.sample(input_numbers, s)
    A_other_sub= A_other[stochIdxs].copy()
    b_sub = b[stochIdxs].copy()

    ones_vec = np.ones((s,1))
    p = ones_vec / (1 +math.e**(-A_other_sub @ w - b_sub * c))
    c_comp = -1/m * np.matmul(b_sub.T ,(ones_vec-p))
    w_comp = -1/m * np.matmul(A_other_sub.T, (ones_vec-p))

    return np.concatenate((w_comp, c_comp))

def f(x,m):
    w= x[:-1]
    c = x[-1]
    num = 0
    for i in range(m):
        a_i = np.reshape(A[i], (10, 1))
        stuff = float(np.matmul(a_i.T, w)) + float(c)

        num += math.log(1 + math.e **(-float(b[i]) * stuff))

    return 1/m * num


# make_BFGS_graphs()
# make_nesterov_graphs()

iter_lst = []
grad_lst = []
iter = 1

s = 100
alpha = 10 ** (-2)

def grad(b,m,x):
    w = x[:-1]
    c = x[-1]
    w_len = len(b)
    ones_vec = np.ones((w_len,1))
    p = ones_vec / (1 +math.e**(-A_other @ w - b * c))
    c_comp = -1/m * np.matmul(b.T ,(ones_vec-p))
    w_comp = -1/m * np.matmul(A_other.T, (ones_vec-p))
    return np.concatenate((w_comp, c_comp))

for i in range(500000):
    if iter % 100 == 0:
        grad_lst.append(np.linalg.norm(np.linalg.norm(grad(b, m, x))))
        iter_lst.append(iter)
    iter+=1
    print(iter)
    x_new = x + alpha* - stochGrad(b,m,x,s)
    x=x_new

plt.scatter(iter_lst, grad_lst)
# Set labels for the axes
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
# Set a title for the plot
plt.title('Gradient Norm vs. Iteration')
# Show the plot
plt.show()