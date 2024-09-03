import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA



matdata = scipy.io.loadmat(r"C:\Users\peter\Downloads\Lasso_data.mat")
b = matdata['b']
A = matdata['A']
xs =  matdata['xs']
tau = float(matdata['tau'])

eigenvalues, eigenvectors = LA.eigh(A.T @ A)
max_eigenval = max(eigenvalues)
x = np.zeros([1024,1])


def subgradient_1_norm(x):
    g= np.zeros((x.shape))
    for i in range(len(x)):
        x_val = x[i]
        if x_val < 0:
            g[i] = -1
        if x_val > 0:
            g[i] = 1
        elif x_val == 0:
            g[i] = 0
    return g


def subgradient_norm_sq(A,b,x):
    return A.T @ A @ x - A.T @ b


def sub_f(x):
    return subgradient_norm_sq(A,b,x) + tau * subgradient_1_norm(x)

def f(x):
    return 1/2 * np.linalg.norm(A@x - b)**2 + tau * np.linalg.norm(x, ord =1)


def stop_condition_subgradient(f_val_lst,i):
    f_best = min(f_val_lst)
    f_val = f_val_lst[i-1]
    return float(np.abs(f_best-f(xs))/f(xs)), float(np.abs(f_val-f(xs))/f(xs))
def stop_condition_ISTA(x):
    return float(np.abs(f(x)-f(xs))/ f(xs))


# subgradient method _____________________________________________________

error_lst = []
iter_lst = []
f_lst = [f(x)]


for i in range(1,5000):
    if stop_condition_subgradient(f_lst,i)[0] < 10**(-3):
        break
    error_lst.append(stop_condition_subgradient(f_lst,i)[0])
    print(i)
    iter_lst.append(i)
    alpha = 0.01/i
    # print('sub', list(alpha*sub_f(x)))
    x_new = x - alpha*sub_f(x)
    x=x_new
    # print('x', x)
    # print('f', f(x))
    f_lst.append(f(x))



print(np.sort(error_lst))
plt.xlabel('Iteration')
plt.ylabel('Relative Function Value Error')
# Set a title for the plot
plt.title('Subgradient Method Relative Function Value Error vs Iterations')
plt.plot(iter_lst, error_lst)
plt.show()



# ISTA _____________________________________________________
def prox_1_norm(u, t):
    u_new  = u.copy()
    for i in range(len(u)):
        if u[i] >= t:
            u_new[i] = u[i] - t
        elif -t <u[i] < t:
            u_new[i] = 0
        elif u[i] <= -t:
            u_new[i] = u[i] + t

    return u_new
error_lst = []
iter_lst = []
x = np.zeros([1024,1])
for i in range(1,5000):
    if stop_condition_ISTA(x) < 10**(-3):
        break
    error_lst.append(stop_condition_ISTA(x))
    iter_lst.append(i)
    alpha = 1/max_eigenval
    x_new = x - alpha*subgradient_norm_sq(A,b,x)
    x_new = prox_1_norm(x_new, alpha)
    x=x_new
    f_lst.append(f(x))
plt.xlabel('Iteration')
plt.ylabel('Relative Function Value Error')
# Set a title for the plot
plt.title('ISTA Method Relative Function Value Error vs Iterations')

plt.plot(iter_lst, error_lst)
plt.show()


