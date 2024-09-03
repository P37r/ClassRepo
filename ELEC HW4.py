import numpy as np
import scipy
import matplotlib.pyplot as plt


matdata = scipy.io.loadmat(r"C:\Users\peter\Downloads\problem2data.mat")
data = matdata['X']


x_orig = data[:,0]
y_orig= data[:,1]

mu_x = np.average(x_orig)
mu_y = np.average(y_orig)

x = x_orig -mu_x
y = y_orig - mu_y

cov_matrix = np.cov(x,y)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

                                                                                        # Graph original data with PCs
plt.figure(figsize=(8, 6))
plt.scatter(x_orig, y_orig, color='blue', alpha=0.5)  # Creating scatter plot
plt.title('Original Data with Principal Components')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(True)
for vector in eigen_vectors.T:
    # Scaling the vector to fill the whole figure
    scale = max(plt.xlim()[1], plt.ylim()[1])
    scaled_vector = vector * scale
    plt.plot([0, scaled_vector[0]], [0, scaled_vector[1]], color='red', linewidth=2)
plt.show()

oneD_projection = []
# print('data 0', data[0])
# print('evec', eigen_vectors[:,0])
for i in range(len(x)):
    oneD_projection.append(data[i] @ eigen_vectors[:,0])
                                                                                            # Graph the 1-D projection
zero_lst = list(np.zeros([len(oneD_projection),1]))

plt.scatter(oneD_projection, zero_lst)
plt.title('1-D Projection')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
# Reprojection

oneD_projection= np.array(oneD_projection)

data_hat = []
for i in range(len(x)):
    data_hat.append(oneD_projection[i] * eigen_vectors[:, 0])


data_hat = np.array(data_hat)
x_hat = data_hat[:,0]
y_hat= data_hat[:,1]


                                                                        # Plot both the actual and the approximated data
plt.figure(figsize=(8, 6))
plt.scatter(x_orig, y_orig, color='blue', alpha=0.5, label = 'original data')  # Creating scatter plot
plt.scatter(x_hat, y_hat, color='red', alpha=0.5, label = 'approximated data')  # Creating scatter plot
plt.legend()

plt.title('Original Data and Data Approximation Using PCA')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(True)
for vector in eigen_vectors.T:
    # Scaling the vector to fill the whole figure
    scale = max(plt.xlim()[1], plt.ylim()[1])
    scaled_vector = vector * scale
    plt.plot([0, scaled_vector[0]], [0, scaled_vector[1]], color='red', linewidth=2)
plt.show()



