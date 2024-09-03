# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA, FastICA
#
# S_1_lst = []
# S_2_lst = []
# X_1_lst = []
# X_2_lst = []
# time = []
# for i in range(0,25*1000):
#     t= i/1000
#     S_1 = math.cos(t)
#     S_2 = math.e**(-t) - 5*math.e**(-t/5)
#
#     X_1 = 0.8*S_1 + 0.5*S_2
#     X_2 = 0.3*S_1 - 0.4*S_2
#
#
#     S_1_lst.append(S_1)
#     S_2_lst.append(S_2)
#     X_1_lst.append(X_1)
#     X_2_lst.append(X_2)
#     time.append(t)
#
#
# plt.title('S_1 and S_2')
# plt.plot(time,S_1_lst, label = 'S1')
# plt.plot(time,S_2_lst, label = 'S2')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
#
# plt.title('X_1 and X_2')
# plt.plot(time,X_1_lst, label = 'X1')
# plt.plot(time,X_2_lst, label = 'X2')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
#
#
# X_1_lst = np.array(X_1_lst)
# X_2_lst = np.array(X_2_lst)
#
# X_observed = np.column_stack((X_1_lst, X_2_lst))
# ica = FastICA(n_components=2)  # You can specify the number of components if known
#
# # Fit the model to the observed signals
# ica.fit(X_observed)
#
# # Transform the observed signals to estimated source signals
# S_estimated = ica.transform(X_observed)
#
# S_1_est = S_estimated[:, 0]
# S_2_est = S_estimated[:, 1]
#
# plt.title('S_1 and S_2 estimated from ICA')
# plt.plot(time,S_1_est, label = 'S1')
# plt.plot(time,S_2_est, label = 'S2')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.show()
#



import numpy as np


A = np.random.randint(1,10, [4,4])/10

print(A)


B = np.random.randint(1,10, [4,4])/10

print(B)

print(A@B)