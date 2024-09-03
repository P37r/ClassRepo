import random
import numpy as np
import math
def CTMC(Q,i,T):
    t= 0
    time_dict = {0:i}
    while t< T:
        min_val= math.inf
        for j in range(len(Q[i])):
            if Q[i][j] >0:
                U = random.random()
                time = -1 /Q[i][j] * math.log(U)
                if time < min_val:
                    min_val = time
                    min_j = j
        t+= min_val
        i=min_j
        time_dict[t]= i
        # print(t)
    time_dict.popitem()
    return time_dict[max(time_dict.keys())], time_dict

Q_p1 = \
    [[-3,3,0,0,0,0],
    [2,-5,3,0,0,0],
    [0,4,-7,3,0,0],
    [0,0,4,-7,3,0],
     [0,0,0,4,-7,3],
    [0,0,0,0,4,-4]]

Q_p1_nu = \
    [[3,3,0,0,0,0],
    [2,5,3,0,0,0],
    [0,4,7,3,0,0],
    [0,0,4,7,3,0],
     [0,0,0,4,7,3],
    [0,0,0,0,4,4]]

# print(CTMC(Q_p1,0,10)[1])


def exp_van(lam,T):
    t=0
    i=0
    s = {}
    while t<= T:
        s[t] = i
        U = random.random()
        t = t - 1/lam * math.log(U)
        i+=1
    return s

def uniformiztion(Q,T,i):
    # pick lambda >= nu
    lam = 8
    Q = np.array(Q)
    P = Q/ lam
    diag = -1 * P.diagonal()
    diag2 = []
    for j in diag:
        diag2.append(1-j)
    np.fill_diagonal(P,diag2)


    n_t_lst = exp_van(lam,T)
    states_lst = list(n_t_lst.values())[1:]
    markov_lst = []
    for z in range(len(states_lst)):
        U =random.random()
        for idx in range(1,len(P[i])+1):
            if U < sum(P[i][:idx]):
                i = idx - 1
                markov_lst.append(i)
                break

    time_state_lst = zip(list(n_t_lst.keys())[1:], markov_lst)
    d = {}
    for tup in time_state_lst:
        x,y = tup
        d[x] = y

    return d[max(d.keys())], d

def q1():
    result_lst = []
    result_lst_unif= []
    for i in range(1000):
        result_lst.append(CTMC(Q_p1,0,10)[0])
        result_lst_unif.append(uniformiztion(Q_p1, 10, 0)[0])

    print('E[X(10)] =',np.mean(result_lst))
    print('Var[X(10)] =',np.var(result_lst))
    print('unif: E[X(10)] =', np.mean(result_lst_unif))
    print('unif: Var[X(10)] =', np.var(result_lst_unif))


Q_p2 = [[0, 1/2],
        [1/4,0]]

def exp(T,X_t):
    t=0
    i=0
    s = {}
    while t< T:
        s[i] = t
        time_lst = []
        time_star = 0
        # Find value of X(t)
        for time in X_t:
            time_lst.append(time)
            if time >t:
                time_star = time_lst[-2]
                break

        #If the loop never assigns t_star then t is larger than any of the markov times so set time = max
        if time_star == 0:
            time_star = max(X_t.keys())

        U = random.random()

        if X_t[time_star] == 0:
            t -= 1/1 * math.log(U)
        else:
            t -= 1 / 2 * math.log(U)
        i += 1
    return s
def q2_helper(T,i):
    Q_p2 = [[-1/2, 1 / 2],
            [1 / 4, -1/4]]

    X_t = CTMC(Q_p2,i,T)[1]
    return exp(T,X_t)

def q2():
    result_lst = []
    for i in range(500):
        result_lst.append(len(q2_helper(50,0).keys())-1)
    return np.var(result_lst)/np.mean(result_lst)

# print(q2())
def infection_sim():
    i = 100
    N = 1000
    lam = 1
    mu  = 0.4
    T = 100
    t=0

    infected_dict = {}
    while t < T:
        birth_rate = lam*(N-i)*i/N
        death_rate = mu*i

        U_1 = random.random()
        U_2 = random.random()

        birth_time = -1/birth_rate * math.log(U_1)
        death_time = -1/death_rate * math.log(U_2)

        if birth_time< death_time:
            t+=birth_time
            i+=1
        else:
            t+= death_time
            i-=1

        infected_dict[t] = i

    return infected_dict
def q3():
    result_lst = []
    for i in range(1000):
        print(i)
        infected_lst = list(infection_sim().values())[-1]
        result_lst.append(infected_lst)

    return np.mean(result_lst),np.var(result_lst)


def q4():
    var_lst = []
    n=0
    while "cat" == "cat":
        X = random.normalvariate(0,1)
        n+=1
        var_lst.append(X)
        S=np.std(var_lst)
        if n >= 100 and S/math.sqrt(n) < 0.1:
            return n, np.mean(var_lst), np.var(var_lst)

# print(q4())

# #lam = 4
def gen_exp_T(lam):
    U = random.random()
    return -1/lam * math.log(U)
def single_server_sim(T,lam1,lam2):
    t = 0
    n = 0
    A = {}
    D = {}
    N_A = 0
    N_D = 0

    #exp service lam = 25
    #break uniform 0,3

    arrival_time = gen_exp_T(lam1)
    t_A = arrival_time
    t_D = math.inf
    while "cat" == "cat":
        if (t_A <= t_D and t_A <= T) and n <=3:
            t = t_A
            N_A +=1
            n +=1
            arrival_time = gen_exp_T(lam1)
            t_A += arrival_time
            if n ==1:
                t_D = t + gen_exp_T(lam2)
            A[N_A] = t

        elif (t_D < t_A and t_D <= T) or n>3:
            N_D +=1
            t = t_D
            n = n-1
            if n ==0:
                t_D = math.inf
            else:
                t_D = t +gen_exp_T(lam2)
            D[N_D] = t

        elif min(t_A,t_D) > T and n >0:
            t = t_D
            n -=1
            N_D += 1
            if n>0:
                t_D = t + gen_exp_T(lam2)
            D[N_D] = t

        elif min(t_A,t_D) > T and n ==0:
            T_p = max(t-T,0)
            return A, D, T_p

def q5_helper(T,lam1,lam2,n):
    n_lst = []
    wait_sum_lst = []
    for j in range(n):
        arrival_dict,depart_dict,l = single_server_sim(T,lam1,lam2)
        wait_dict = {}
        for i in arrival_dict:
            wait_dict[i] = depart_dict[i] - arrival_dict[i]
        wait_sum = sum(wait_dict.values())
        wait_sum_lst.append(wait_sum)
        n_lst.append(len(wait_dict.keys()))

    theta = sum(wait_sum_lst) / sum(n_lst)
    return theta, wait_sum_lst, n_lst

def MSE_est(T,lam1,lam2,n):
    Y_lst = []
    for _ in range(100):
        theta, wait_sum_lst, n_lst = q5_helper(T, lam1, lam2, n)
        d_bar = sum(wait_sum_lst) / len(wait_sum_lst)
        n_bar = sum(n_lst) / len(n_lst)
        D_i = 0
        N_i = 0
        for i in range(len(wait_sum_lst)):
            random_N = random.choice(n_lst)
            random_D = random.choice(wait_sum_lst)

            D_i += random_D
            N_i += random_N
        Y = (D_i / N_i - d_bar / n_bar) ** 2
        Y_lst.append(Y)

    MSE = sum(Y_lst) / len(Y_lst)
    return MSE
# q1()
print(q2())
# print(q3())
# print(q4())
# print("theta!", q5_helper(8,4,4.2,1000)[0])
# print("MSE", MSE_est(8,4,4.2,1000))
