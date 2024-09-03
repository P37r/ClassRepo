import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

A = pd.read_csv(r"C:\Users\peter\Downloads\A0 - Sheet1 (1).csv")
b = pd.read_csv(r"C:\Users\peter\Downloads\b - Sheet1 (1).csv")
A_other = pd.read_csv(r"C:\Users\peter\Downloads\A_other - Sheet1 (2).csv")
A_other = A_other.to_numpy()

b = b.to_numpy()
A = A.to_numpy()

m = len(b)
alpha = 1
Beta = 0.8
alpha = alpha * Beta
c_1 = 0.1
w_0 = np.zeros([10,1])
c_0 = np.zeros([1,1])
x = np.concatenate((w_0,c_0))
x_prev = x.copy()

def find_root(x):
    stuff = math.sqrt(x**4 - 2*x**2 + 5)
    root1 = 1/2 * (x**2 - stuff - 1)
    root2 = 1/2 * (x**2 + stuff - 1)
    if 0 <=root1 <=1:
        return root1
    elif 0 <=root2 <=1:
        return root2
    else:
        print("ERROR!")

def grad(A,b,m,x):
    w = x[:-1]
    c = x[-1]
    w_len = len(b)
    ones_vec = np.ones((w_len,1))
    p = ones_vec / (1 +math.e**(-A_other @ w - b * c))
    c_comp = -1/m * np.matmul(b.T ,(ones_vec-p))
    w_comp = -1/m * np.matmul(A_other.T, (ones_vec-p))
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

def Armijo(A,b,m,x):
    alpha = 1
    #must define d_k as -gradient
    #must get the gradient
    gradient = grad(A,b,m,x)
    x_new = x + alpha* -gradient
    while f(x_new,m) > f(x,m) + c_1 * alpha * -1 * gradient.T @ gradient:
        alpha *=Beta
        x_new = x + alpha * -gradient
    return alpha

iter_lst = []
grad_lst = []

def Nesterov(x,x_prev):
    iter = 0
    B = 0
    p = 0
    L = 1 / 4 * max(np.linalg.eigvals(np.matmul(A.T, A)))
    while np.linalg.norm(grad(A,b,m,x)) > 10**(-8):
        iter_lst.append(iter)
        grad_lst.append(np.linalg.norm(grad(A,b,m,x)))
        iter+=1

        y_k = x + B * (x - x_prev)

        x_prev = x
        x = y_k  - 1/L * grad(A,b,m,y_k)
        p_prev = p
        p = find_root(p)
        B = p_prev**2 * p

def BFGS(x):
    iter = 0
    I = np.identity(11)
    B = I.copy()
    B_lst = []
    while np.linalg.norm(grad(A, b, m, x)) > 10 ** (-8):
        iter_lst.append(iter)
        gradient = grad(A, b, m, x)
        grad_lst.append(np.linalg.norm(gradient))
        iter += 1
        p_k = -1 * np.linalg.inv(B) @ gradient
        step = Armijo(A, b, m, x)
        x_prev = x
        x = x + step * p_k
        gradient_prev = gradient
        gradient = grad(A, b, m, x)

        y_k = gradient - gradient_prev
        s_k = x - x_prev
        rho_k = float(1 / np.matmul(y_k.T, s_k))

        B = (I - rho_k * (y_k @ s_k.T)) @ B @ (I - rho_k * (y_k @ s_k.T)) + rho_k * y_k @ y_k.T
        B_lst.append(B)
    return B_lst
def make_nesterov_graphs():
    Nesterov(x, x_prev)
    plt.plot(iter_lst, grad_lst)
    # Set labels for the axes
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    # Set a title for the plot
    plt.title('Gradient Norm vs. Iteration')
    # Show the plot
    plt.show()

def make_BFGS_graphs():
    B_lst = BFGS(x)
    obj_val_lst = []
    for i in range(1,len(B_lst)):
        obj_val = np.linalg.norm(B_lst[i] - B_lst[i-1])
        obj_val_lst.append(obj_val)


    #BFGS gradient norm
    plt.plot(iter_lst, grad_lst)
    # Set labels for the axes
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    # Set a title for the plot
    plt.title('Gradient Norm vs. Iteration')
    # Show the plot
    plt.show()

    #BFGS obj val
    plt.plot(iter_lst[1:], obj_val_lst)
    # Set labels for the axes
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    # Set a title for the plot
    plt.title('Objective Value vs. Iteration')
    # Show the plot
    plt.show()

# make_BFGS_graphs()
make_nesterov_graphs()