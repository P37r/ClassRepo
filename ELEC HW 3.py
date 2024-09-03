import random
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matdata = scipy.io.loadmat(r"C:\Users\peter\Downloads\data_1.mat")
matdata = matdata['someData']
channel1 = list(matdata[:, 1])[:50000]
channel1.sort()


def knn(lst, q, k, n):
    for i in range(len(lst)):
        if lst[i] > q or i == len(lst) - 1:
            if i - k >= 0:
                neighborhood = lst[i - k:i + k]
            else:
                neighborhood = lst[i:i + k]
            if i + k > len(lst):
                neighborhood = lst[len(lst) + 1 - k - 1: len(lst) + 1]
            dist_dict = {}
            dist_abs_lst = []
            for j in neighborhood:
                dist_dict[np.abs(q - j)] = q - j
                dist_abs_lst.append(np.abs(q - j))

            dist_abs_lst.sort()
            l = []
            # print(dist_abs_lst)
            for i in range(k):
                l.append(dist_dict[dist_abs_lst[i]])
            l.sort()
            # print('l', l)
            V = l[k - 1] - l[0]

            break

    return k / (n * V)


def est(var_lst, k, n, c):
    lst = split(var_lst, n, c)

    prob_dict = {}
    for q in lst:
        prob_dict[q] = knn(var_lst, q, k, n)
    return prob_dict


def split(var_lst, n, c):
    max_adj = max(var_lst) + c
    min_adj = min(var_lst) - c
    diff = max_adj - min_adj
    lst = []
    num = 0
    for i in range(n + 1):
        lst.append(min_adj + num)
        num += diff / n
    return lst


def knn2D(x, y, k, n, twoD_data, length):
    dist_lst = []
    for i in range(length):
        for j in range(length):
            data_x, data_y = twoD_data[i][j]
            dist_lst.append((data_x - x) ** 2 + (data_y - y) ** 2)

    dist_lst.sort()
    r = dist_lst[k - 1]
    V = math.pi * r ** 2
    return k / (n * V)


def est2(var_lst, var_lst2, k, n, c, twoD_Data):
    x_segments = split(var_lst, n, c)
    y_segments = split(var_lst2, n, c)

    var_len = len(var_lst)
    prob_dict = {}
    for i in x_segments:
        for j in y_segments:
            prob_dict[tuple([i, j])] = knn2D(i, j, k, n, twoD_Data, var_len)

    return prob_dict


def entropy(dict):
    num = 0
    for i in dict:
        p_i = dict[i]
        num += p_i * math.log(p_i)
    return -num


#                                                                                               Calculate Entropy
n_num = 100
k_num = 5000
division_num = 500
channel1_dist = est(channel1, k_num, division_num, 0.2)
entropy_val = entropy(channel1_dist)
print('entropy: ', entropy_val)

#                                                                                               Calculate I(X;Y)
n_num = 100
k_num = 20
division_num = 100


x_lst = []
y_lst = []

for i in range(n_num):
    num1 = random.normalvariate(0, 1)
    num2 = random.normalvariate(0, 1)
    x_lst.append(num1)
    y_lst.append(num2)

x_lst.sort()
y_lst.sort()

n = len(x_lst)
twoD_data = []

for i in range(n):
    little_lst = []
    for j in range(n):
        num1 = random.normalvariate(0, 1)
        num2 = random.normalvariate(0, 1)
        little_lst.append([num1, num2])
    twoD_data.append(little_lst)

x_dist = est(x_lst, k_num, division_num, 0.2)
y_dist = est(y_lst, k_num, division_num, 0.2)

# joint stuff

joint_dist = est2(x_lst, y_lst, k_num, division_num, 0.2, twoD_data)
joint_norm_factor = sum(joint_dist.values())

for point in joint_dist:
    joint_dist[point] /= joint_norm_factor

x_norm_factor = sum(x_dist.values())
y_norm_factor = sum(y_dist.values())

for point in x_dist:
    x_dist[point] /= x_norm_factor

for point in y_dist:
    y_dist[point] /= y_norm_factor

for point in joint_dist:
    joint_dist[point] /= joint_norm_factor

val = 0

for i in x_dist:
    for j in y_dist:
        p_x_y = joint_dist[tuple([i, j])]
        val += p_x_y * math.log(p_x_y / (x_dist[i] * y_dist[j]))

print('mutual information estimate:', val)
