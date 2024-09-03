import random
import numpy as np
import math
from scipy.stats import norm
from scipy.stats import poisson
from collections import defaultdict

def control_sim(est_lst,control_lst):
    var_control = np.var(control_lst)
    cov = np.cov(est_lst, control_lst)[0][1]
    c_star = - cov / var_control

    control_result_lst = []
    var_control_mean = np.average(control_lst)
    for i in range(len(est_lst)):
        control_result_lst.append(est_lst[i] + c_star * (control_lst[i] - var_control_mean))

    print('conditional control est variance', np.var(control_result_lst))

def p18():

    #raw estimator
    raw_estimator_lst = []

    for _ in range(10000):
        Y = random.normalvariate(1,1)
        X = random.normalvariate(Y, 2)
        if X >1:
            raw_estimator_lst.append(1)
        else:
            raw_estimator_lst.append(0)
    print('raw estimtor var' , np.var(raw_estimator_lst))

    conditional_est_lst = []
    for i in range(10000):
        Y = random.normalvariate(1, 1)
        conditional_est_lst.append(1- norm.cdf(5, loc=Y, scale=2))
    print('conditional est var', np.var(conditional_est_lst))

    conditional_est_anti_lst = []
    for i in range(5000):
        Y = random.normalvariate(1, 1)
        Y_2 = 1-(Y-1)
        val1 = 1- norm.cdf(5, loc=Y, scale=2)
        val2 = 1- norm.cdf(5, loc=Y_2, scale=2)
        conditional_est_anti_lst.append((val1 + val2)/2)

    print('conditional est antithetic var', np.var(conditional_est_anti_lst))

    conditional_est_lst = []
    control_lst= []
    for i in range(10000):
        Y = random.normalvariate(1, 1)
        conditional_est_lst.append(1 - norm.cdf(5, loc=Y, scale=2))
        control_lst.append(math.e**Y)
    print("part h")
    control_sim(conditional_est_lst,control_lst)

def p19(n):
    raw_lst = []

    for _ in range(n):
        U = random.random()
        X = poisson.rvs(15/(0.5 + U))
        if X >= 20:
            raw_lst.append(1)
        else:
            raw_lst.append(0)

    print('raw est variance', np.var(raw_lst))

    conditional_est_lst = []
    control_lst = []
    for _ in range(n):
        U = random.random()
        val = 1 - poisson.cdf(19, mu =15/(0.5+U))
        conditional_est_lst.append(val)
        control_lst.append(1-val)

    control_sim(conditional_est_lst, control_lst)



    conditional_est_anti_lst = []
    for _ in range(int(n/2)):
        U = random.random()
        U_2 = 1 - U
        va1 = 1 - poisson.cdf(19, mu=15 / (0.5 + U))
        val2 = 1 - poisson.cdf(19, mu=15 / (0.5 + U_2))
        conditional_est_anti_lst.append((va1 + val2)/2)
    print('conditional est anti variance', np.var(conditional_est_anti_lst))


def gen_exp_T(lam):
    U = random.random()
    return -1/lam * math.log(U), U
def single_server_sim(T,lam1,lam2):
    t = 0
    n = 0
    A = {}
    D = {}
    N = {}
    N_A = 0
    N_D = 0

    U_arriv_lst = []
    U_depart_lst = []

    l = gen_exp_T(lam1)
    arrival_time = l[0]
    U_arriv_lst.append(l[1])
    t_A = arrival_time
    t_D = math.inf
    while N_D <=10:
        if (t_A <= t_D and t_A <= T):
            t = t_A
            N_A +=1
            if N_A <= 10:
                N[N_A] = n
            n +=1
            l = gen_exp_T(lam1)
            arrival_time = l[0]
            U_arriv_lst.append(l[1])
            t_A += arrival_time
            if n ==1:
                l = gen_exp_T(lam2)
                t_D = t + l[0]
                U_depart_lst.append(l[1])
            A[N_A] = t

        elif (t_D < t_A and t_D <= T):
            N_D +=1
            t = t_D
            n = n-1
            if n ==0:
                t_D = math.inf
            else:
                l = gen_exp_T(lam2)
                t_D = t +l[0]
                U_depart_lst.append(l[1])
            D[N_D] = t

        elif min(t_A,t_D) > T and n >0:
            t = t_D
            n -=1
            N_D += 1
            if n>0:
                l = gen_exp_T(lam2)
                U_depart_lst.append(l[1])
                t_D = t + l[0]
            D[N_D] = t

        elif min(t_A,t_D) > T and n ==0:
            T_p = max(t-T,0)
            return A, D, T_p

    time_in_sys = {}
    for i in range(1,11):
        time_in_sys[i] = D[i] - A[i]

    I_dict = {}
    for i in range(1,10):
        I_dict[i] = A[i+1] - A[i]

    for i in range(len(U_arriv_lst)):
        U_arriv_lst[i] = 1-U_arriv_lst[i]

    for i in range(len(U_depart_lst)):
        U_depart_lst[i] = 1 - U_depart_lst[i]

    for i in range(50):
        U_arriv_lst.append(1- random.random())
    for i in range(50):
        U_depart_lst.append(1- random.random())
    return A, D, time_in_sys, I_dict, N,U_depart_lst, U_arriv_lst

def single_server_sim2(T, U_arriv, U_depart, lam1, lam2):
    t = 0
    n = 0
    A = {}
    D = {}
    N = {}
    N_A = 0
    N_D = 0

    arrival_time = -1/lam1 * math.log(U_arriv[N_A])
    t_A = arrival_time
    t_D = math.inf
    while N_D <= 10:
        if (t_A <= t_D and t_A <= T):
            t = t_A
            N_A += 1
            if N_A <= 10:
                N[N_A] = n
            n += 1
            arrival_time = -1/lam1 * math.log(U_arriv[N_A])
            t_A += arrival_time
            if n == 1:
                departure_time = -1/lam2 * math.log(U_depart[N_D])
                t_D = t + departure_time
            A[N_A] = t

        elif (t_D < t_A and t_D <= T):
            N_D += 1
            t = t_D
            n = n - 1
            if n == 0:
                t_D = math.inf
            else:
                depart_time=  -1/lam2 * math.log(U_depart[N_D])
                t_D = t + depart_time
            D[N_D] = t

        elif min(t_A, t_D) > T and n > 0:
            t = t_D
            n -= 1
            N_D += 1
            if n > 0:
                depart_time = -1/lam2 * math.log(U_depart[N_D])
                t_D = t + depart_time
            D[N_D] = t

        elif min(t_A, t_D) > T and n == 0:
            T_p = max(t - T, 0)
            return A, D, T_p

    time_in_sys = {}
    for i in range(1, 11):
        time_in_sys[i] = D[i] - A[i]


    return A, D, time_in_sys, N

def q24():
    var_lst = []
    for i in range(500):
        result = single_server_sim(1000000, 2,1)

        U_lst1 = result[6]
        U_lst2 = result[5]

        d1 = result[2]
        d2 = single_server_sim2(1000000000, U_lst1, U_lst2, 2, 1)[2]

        d1_sum = sum(d1.values())
        d2_sum = sum(d2.values())
        var_lst.append((d1_sum + d2_sum)/2)

    print('antithetic variance', np.var(var_lst))

    # part c
    var_lst = []
    service_lst = []
    I_lst = []
    N_dict = defaultdict(lambda : [])
    for _ in range(1000):
        result = single_server_sim(100000000, 2, 1)
        time_in_sys = sum(result[2].values())
        service_sum = sum(result[1].values())
        N = result[4]
        I_sum = sum(result[3].values())
        var_lst.append(time_in_sys)
        service_lst.append(service_sum)
        I_lst.append(I_sum)
        for i in N:
            N_dict[i].append(N[i])

    print("raw est",np.var(var_lst))
    print('part c')
    control_sim(var_lst,service_lst)

    print('part d')
    compare_lst = []


    for i in range(len(service_lst)):
        compare_lst.append(service_lst[i] - I_lst[i])
    control_sim(compare_lst,service_lst)

    len_val = len(N_dict[2])
    E_S_i = []
    for j in range(len_val):
        sum_lst = []
        for i in N_dict:
            sum_lst.append((N_dict[i][j]) + 1)
        E_S_i.append(sum(sum_lst))

    print('sum(E[T_i|N_i]) est variance', np.var(E_S_i))

# p18()
# p19(10000)
q24()

