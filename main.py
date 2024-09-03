import random
import math
import numpy as np
import matplotlib.pyplot as plt

def gen_exp_T(lam):
    U = random.random()
    return -1/lam * math.log(U)
def arriv_func(t):
    if t<0:
        return "error"
    if t <= 5:
        return 4 + 3 * t
    elif t <= 10:
        return 19 - 3 * (t-5)

    else:
        return arriv_func(t - 10)
def single_server_sim(T):
    break_time_lst = []
    t = 0
    n = 0
    A = {}
    D = {}

    #exp service lam = 25
    #break uniform 0,3

    arrival_time = gen_exp_T(arriv_func(t))
    t_A = arrival_time
    t_D = math.inf
    break_total = 0
    while "cat" == "cat":
        if t_A <= t_D and t_A <= T:
            if n!=0:
                t = t_A
            n +=1
            arrival_time = gen_exp_T(arriv_func(t))
            t_A += arrival_time
            if n ==1:
                t_D = t + gen_exp_T(25)

        elif t_D < t_A and t_D <= T:
            t = t_D
            n = n-1

            if n ==0:
                t_D = math.inf
                break_total = 0

                while break_total < arrival_time:
                    break_time = random.uniform(0, 0.3)
                    break_total += break_time
                break_time_lst.append(break_total)
                t+=break_total

            else:
                t_D = t +gen_exp_T(25)

        elif min(t_A,t_D) > T and n >0:
            t = t_D
            n -=1
            if n>0:
                t_D = t + gen_exp_T(25)

        elif min(t_A,t_D) > T and n ==0:
            T_p = max(t-T,0)
            return A, D, T_p, sum(break_time_lst)
def problem1():
    waiting_time_lst = []
    for i in range(500):
        waiting_time_lst.append(single_server_sim(100)[3])
    print('average resting time', sum(waiting_time_lst) / len(waiting_time_lst))

def insurance_sim():
    capital_val = 25000
    t = 0
    while t <365:
        U = random.random()
        arrival_time = -1/10 * math.log(U)
        t += arrival_time
        claim = gen_exp_T(0.001)
        payment = arrival_time * 11000
        capital_val += payment - claim

        if capital_val < 0:
            return False
    return True

def problem2(trials):
    neg_count = 0
    for i in range(trials):
        if not insurance_sim():
            neg_count +=1
    return (trials - neg_count)/trials





#stock problem,
# purchase option in the next 20 days
# S_0 = 100
# option price = 100
# mu = -0.05, sigma = 0.3

def y(x):
    return 1/(1+ 0.33267*x)
def norm_approx(x):
    return 1- 1/math.sqrt(2 * math.pi) * (0.4361836 * y(x) -0.1201676 * y(x)**2 + 0.9372980 * y(x)**3) * \
           math.e**(-x**2 / 2)

def long_expression(p_m,alpha,i,sig, K,mu):
    b_i =  (i * mu - math.log(K / p_m)) / (sig * math.sqrt(i))
    return K + p_m * math.e ** (i * alpha) * norm_approx(sig*math.sqrt(i) + b_i) - K * norm_approx(b_i)

def stock_price(mu, sig, K):
    alpha = mu + sig**2 / 2
    p_dict = {20:100}
    for i in range(20,0,-1):
        X = random.normalvariate(-0.05, 0.3)
        p_dict[i-1] = p_dict[i] * math.e ** X

    for m in range(20,0,-1):
        satisfy_all_conditions = True
        if p_dict[m] > K:
            for i in range(1,m+1):
                if p_dict[m] <= long_expression(p_dict[m],alpha,i,sig, K,mu):
                    satisfy_all_conditions = False
                    break
            if satisfy_all_conditions:
                return m, p_dict[m] - K

    return 0,0


def problem3():
    result_lst = []
    for i in range(10000):
        result_lst.append(stock_price(-0.05,0.3,100)[1])
    return sum(result_lst) / len(result_lst)


import numpy as np

# Parameters
arrival_rate = 10
mu1 = 1
mu2 = 1 / 2
p = 0.6
simulation_count = 1000
simulation_time = [50, 100]


def hyper_exp(mu1, mu2, p):
    U = random.random()
    U_2 = random.random()
    if U <p:
        return -1/mu1 * math.log(U_2)
    else:
        return -1 / mu2 * math.log(U_2)
def find_customers_in_system(tau_dict, S_dict, t):

    print(S_dict)
    still_in_system = 0
    left_system = 0
    idx = len(tau_dict.keys())
    # print(idx)
    # print(S_dict)
    for i in tau_dict:
        if tau_dict[i] > t:
            idx = i-1
            break
    for i in range(1,idx+1):
        time_cust_left = tau_dict[i] + S_dict[i]
        if time_cust_left <= t:
            left_system +=1


    return idx-left_system
def lam_func(t):
    return 5 * math.sin(0.5*t) + 5
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

def simulate_system(lam, T,homo,div):
    tau = 0
    S= 0
    tau_dict= {}
    S_dict = {}
    i= 0

    if homo:
        while tau < T:
            i+=1
            U = random.random()
            tau+= -1/lam * math.log(U)
            tau_dict[i] = tau
    else:
        tau_dict = thinning_alg(lam_func, T, 10)

    for i in tau_dict:
        S = hyper_exp(1, 1 / 2, 0.6)
        S_dict[i] = S

    num_cust_50 = find_customers_in_system(tau_dict,S_dict, 50)
    num_cust_100 = find_customers_in_system(tau_dict,S_dict, 100)

    x_lst = []
    y_lst = []

    if div!= 0:
        for i in range(T*div):
            x_lst.append(i/div)
            y_lst.append(find_customers_in_system(tau_dict,S_dict, i/div))

    return num_cust_50, num_cust_100, x_lst, y_lst

def problem7():
    homo_50 = []
    homo_100 = []
    non_homo_50 = []
    non_homo_100 = []
    #
    # for _ in range(10000):
    #     homo = simulate_system(10, 100, True,0)
    #     no_homo = simulate_system(10, 100, False,0)
    #
    #     homo_50.append(homo[0])
    #     homo_100.append(homo[1])
    #     non_homo_50.append(no_homo[0])
    #     non_homo_100.append(no_homo[1])
    #
    # iter_lst = [homo_50, homo_100, non_homo_50,non_homo_100]
    # name_lst = ['homo_50', 'homo_100', 'non_homo_50', 'non_homo_100']
    #
    # for i in range(len(iter_lst)):
    #     name = name_lst[i]
    #     lst = iter_lst[i]
    #     print(name)
    #     print('mean', np.mean(lst))
    #     print('variance', np.var(lst))
    #     print("--------------")

    result = simulate_system(10, 100, True, 4)
    x = result[2]
    # y = result[3]
    #
    # plt.plot(x, y)
    # plt.title("single path")
    # plt.xlabel("time")
    # plt.ylabel("number of customers in system")
    # plt.show()

    #average plot
    d = {}
    for i in range(1000):
        print(i)
        d[i] = simulate_system(10, 100, False, 4)[3]
    len_x_lst = len(d[0])
    y_avg_lst= []


    for i in range(len_x_lst):
        # print("cat")
        avg_lst = []
        for j in d:
            avg_lst.append(d[j][i])
        mean = np.mean(avg_lst)
        y_avg_lst.append(mean)

    plt.plot(x, y_avg_lst)
    plt.title("average of 1000 simulated paths")
    plt.xlabel("time")
    plt.ylabel("average number of customers in system")
    plt.show()


# problem1()
# print(problem2(10000))
# print(problem3())
# problem7()

result = simulate_system(10, 100, True, 0)
# print(result)