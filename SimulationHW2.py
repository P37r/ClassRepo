import random
import math
import matplotlib.pyplot as plt
import statistics as stat
from collections import defaultdict
from scipy.linalg import sqrtm
import numpy as np

def exp(lam,T):
    t=0
    i=0
    s = {}
    people = {}
    num_people = 0
    while t<= T:
        s[i] = t
        people[i] = num_people
        U = random.random()
        t = t - 1/lam * math.log(U)
        i+=1
        num_people +=random.randint(20, 40)
    return s,people
####problem 30
def func(t):
    if 0<=t<=5:
        return t/5
    if 5<t<=10:
        return 1 + 5* (t-5)
# is first 10 time units t or I?
# method 1

#average 70
def thinning_alg(lam_func, T,lam):
    t = 0
    I = 0
    S= {}
    while t<= T:
        U = random.random()
        t -= 1/lam * math.log(U)
        if t > T:
            return S
        U = random.random()
        if U <= lam_func(t)/lam:
            I+=1
            S[I] = t
    return S

#method 2
def non_homo(lam_func,T):
    t_dict = {}
    for i in range(T+1):
        t_dict[i] = i
    lam_dict = {}
    for i in range(T+1):
        if i != 0 or i!= 5:
            lam_dict[i] = lam_func(i)

    S= {}
    t=0
    J=1
    I=0

    while 1==1:
        U = random.random()
        X = -1/(lam_dict[J]) * math.log(U)

        while t + X > t_dict[J]:
            if J == T:
                return S
            X = (X-t_dict[J] + t) * lam_dict[J] / lam_dict[J+1]
            t = t_dict[J]
            J+=1

        t+=X
        U = random.random()
        if U <= lam_func(t)/lam_dict[J]:
            I+=1
            S[I] = t

# Problem 33
def circ_pois(lam, r):
    n=0
    x_lst=[]
    r_lst=[]
    u_lst=[]
    theta_lst = []
    while sum(x_lst) <= math.pi * r**2:
        U = random.random()
        x = -1/lam * math.log(U)
        x_lst.append(x)
        n+=1
    if n ==1:
        return "no points in C(" + str(r) + ")"

    for i in range(1,len(x_lst)):
        x_sub = x_lst[:i]
        r = math.sqrt(sum(x_sub)/math.pi)
        r_lst.append(r)
        U = random.random()
        u_lst.append(U)

    for u in u_lst:
        theta_lst.append(2*math.pi * u)

    return r_lst, theta_lst

res = circ_pois(1,5)
fig = plt.figure()

ax = fig.add_subplot(projection='polar')
c = ax.scatter(res[1], res[0])
plt.title("2-D Poisson Process within a Circle")
plt.show()

# Problem 4
def hyper_exp(p,lam1,lam2):
    U = random.random()
    if U < p:
        Y = -1/lam1 * math.log(U)
    else:
        Y = -1/lam2 * math.log(U)
    return Y

def exp_mod(lam1,lam2,T,p):
    t=0
    i=0
    s = {}
    y_lst = []
    y = 0
    while t<= T:
        y_lst.append(i)
        s[i] = t
        Y = hyper_exp(p,lam1,lam2)
        t += Y
        i+=1
    return s,y_lst

# problem 4 part a
x_dict,y = exp_mod(1,2,100,0.6)
x= []

for i in range(len(x_dict.values())):
    x.append(x_dict[i])

plt.step(x,y)
plt.title('N(t) with H2 hyperexponential parameter')
plt.xlabel('time')
plt.ylabel('N(t)')

plt.show()

# problem 4 part b
for i in range(50):
    x_dict, y = exp_mod(1, 2, 100, 0.6)
    plt.step(x_dict.values(), y)
plt.title('50 simulation of N(t) with H2 hyperexponential parameter')
plt.xlabel('time')
plt.ylabel('N(t)')
plt.show()
# problem 4 part c
def find_index(lst, time):
    idx = 0
    for i in lst:
        if i>time:
            return idx
        idx+=1

def find_ISD(func,arg1,arg2,arg3,arg4):
    result_dict = defaultdict(lambda: [])
    for i in range(1000):
        x= []
        x_dict = func(arg1, arg2, arg3, arg4)[0]
        for i in range(len(x_dict.values())):
            x.append(x_dict[i])

        idx_50 = find_index(x, 50)
        idx_90 = find_index(x, 90)

        result_dict[50].append(idx_50)
        result_dict[90].append(idx_90)

    IDE_dict = {}

    for i in result_dict:
        sub_lst = result_dict[i]
        m = stat.mean(sub_lst)
        v = stat.variance(sub_lst)
        IDE_dict[i] = v / m

    print(IDE_dict)
# FIND ISD 1
# find_ISD(exp_mod,1,2,100,0.6)
# Problem 6
def h(t,c):
    return 0.8 * math.e**(-1.2*(t-c))

def lam_hawkes(t,lam_0,tau_lst):
    result_lst = []
    for tau in tau_lst:
        result_lst.append(h(t,tau))
    return lam_0 + sum(result_lst)

def hawks_sim(lam_func,lam_0,T,dummy):
    t = 0
    I = 0
    S= {0:0}
    tau_lst = []
    lam_t = {0:0}

    while t<= T:
        h_t_lst = []
        # h(x) is maximized at h(t), given a t, and as tau increases, h(t-tau) increases
        # h_t_lst contains all the max values of h(t-tau_i) given a t.
        for tau in tau_lst:
            h_t_lst.append(h(t, tau))
        lam = lam_0 + sum(h_t_lst)

        U = random.random()
        X = -1/lam * math.log(U)
        t += X
        lam_val = lam_func(t, lam_0, tau_lst)
        U_2 = random.random()
        if U_2 <= lam_val/lam:
            I+=1
            S[I] = t
            tau_lst = S.values()
            lam_t[I] = lam_val

    S.popitem()
    return S, lam_t,tau_lst
def lam_hawkes_graph(t,lam_0,full_tau_lst):
    full_tau_lst = list(full_tau_lst)
    for i in range(len(full_tau_lst)):
        if full_tau_lst[i] > t:
            tau_splice = full_tau_lst[:i]
            break
        elif i == len(full_tau_lst)-1:
            tau_splice = full_tau_lst[:i]
            break
    result_lst = []
    for tau in tau_splice:
        result_lst.append(h(t,tau))
    return lam_0 + sum(result_lst)

result = hawks_sim(lam_hawkes,1,100,0)

S = result[0]
lam = result[1]
full_tau_lst = result[2]

def n_t_graph(t,dict):
    lb =  0
    for i in range(lb, len(dict.keys())):
        if dict[i] > t:
            return i-1
        elif i == len(dict.keys()):
            return i-1

x = [i/100 for i in range(10000)]
x.append(100)

y_lam = []
y_N_t = []

for i in x:
    y_lam.append(lam_hawkes_graph(i,1,full_tau_lst))

for i in x:
    y_N_t.append(n_t_graph(i,S))

plt.plot(x,y_lam)
plt.title("lam(t) graph")
plt.xlabel("time")
plt.ylabel("lam(t)")

plt.show()

plt.step(x,y_N_t)
plt.title("N(t) graph")
plt.xlabel("time")
plt.ylabel("N(t)")

plt.show()
####Chapter 6 problem 7

A = np.array([[3,-2,1],[-2,5,3],[1,3,4]])

#calculate sqrt using diagonalization
evalues, evectors = np.linalg.eigh(A)
lam = np.diag(evalues)
new_lam = np.sqrt(np.round(lam,15))
sqrt_matrix = evectors @ new_lam @ evectors.T

Z = np.array([random.normalvariate(0,1),random.normalvariate(0,1),random.normalvariate(0,1)])
mu_lst = np.array([1,2,3])