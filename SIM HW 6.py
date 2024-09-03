import random
import math
from math import comb
from scipy.stats import bernoulli
import numpy as np


def q1(trials,theta):
    result_lst = []
    for i in range(trials):
        S_n = np.random.binomial(40,17/40)
        if S_n>16:
            result_lst.append( math.e ** (-theta * S_n) * (0.2 * (math.e ** theta) + 1 - 0.2) ** 40)
        else:
            result_lst.append(0)
    return sum(result_lst)/trials



def tau(x):
    S_n = 0
    k = 0
    while  S_n <= x:
        k+=1
        x = random.normalvariate(1, math.sqrt(0.5))
        S_n+=x
    return S_n

def q2(trials):
    numerator = 0
    denominator = 0
    for i in range(trials):
        x = random.uniform(0,1)
        f_x = math.e** (x + x**2)
        g_x = 1
        h_x = x

        numerator+= h_x * f_x / g_x
        denominator += f_x / g_x
    return numerator/denominator
def q2_resample(trials):
    a_lst = []
    for i in range(trials):
        a_lst.append(random.random())
    b_lst = []
    for a in a_lst:
        b_lst.append(math.e**(a + a**2))

    b_lst_sum = sum(b_lst)
    b_lst = [i/ b_lst_sum for i in b_lst]

    sample = np.random.choice(a_lst, 1000, True, b_lst)

    return np.average(sample)


def tau(x):
    S_n = 0
    k = 0
    while  S_n <= x:
        k+=1
        X = random.normalvariate(0.1, math.sqrt(0.5))
        S_n+=X
    return S_n
def q3(trials):
#     # -2 * -0.1/0.5 = -0.4
    theta = 0.4
    x= 10
    result_lst = []
    for i in range(trials):
        delta = tau(x) - x
        result_lst.append(math.e**(-theta * delta))





    return math.e**(-theta * x) * np.average(result_lst)

def q3_perc(trials):
    result_lst = []
    for i in range(trials):
        print(i)
        result_lst.append(q3(trials))

    perc_2_5 = np.percentile(result_lst, 2.5)
    perc_97_5 =  np.percentile(result_lst, 97.5)

    return (perc_2_5, perc_97_5)
def q4(B,num,cutoff):
    init_guess_y = B/2

    y = init_guess_y
    # 1 = C_y * math.e**(-y*x)
    x_lst = []
    y_lst = []

    for i in range(num):
        U_1 = random.random()
        U_2 = random.random()
        print('y!',y)
        print((-math.e**(-B*y)/y - (-1/y)))
        C_y = 1/ (-math.e**(-B*y)/y - (-1/y))
        x = math.log(1 - (U_1 * y/ C_y)) / -y
        print((-math.e**(-B*x)/x - (-1/x)))
        C_x = 1/ (-math.e**(-B*x)/x - (-1/x))

        y = math.log(1 - (U_2 * x/ C_x)) / -x

        x_lst.append(x)
        y_lst.append(y)

    x_lst = x_lst[cutoff:]
    y_lst = y_lst[cutoff:]

    xy_lst = []
    for i in range(len(x_lst)):
        xy_lst.append(x_lst[i] * y_lst[i])


    print(x_lst)

    return np.average(x_lst), np.average(xy_lst)
def q5(num,cutoff):
    x = 2
    y = 3

    x_lst = []
    y_lst = []
    z_lst = []
    for i in range(num):
        U_z = random.random()
        U_y = random.random()
        U_x = random.random()
        z= -1/(x+y+1) * math.log(U_z)
        y = -1 / (x + z + 1) * math.log(U_y)
        x = -1 / (y + z + 1) * math.log(U_x)
        if i >cutoff:
            x_lst.append(x)
            y_lst.append(y)
            z_lst.append(z)

    xyz_lst = []
    for i in range(len(x_lst)):
        xyz_lst.append(x_lst[i] * y_lst[i] * z_lst[i])
    return np.average(xyz_lst)

# print(q1(1000,math.log(68/23)))
# print(q2_resample(1000))
# print(q2(1000))
# print(q3_perc(1000))
# print(q3(1000))
# print(q4(1, 11000,1000))
# print(q5(11000,1000))
