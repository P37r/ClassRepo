import numpy as np
import matplotlib.pyplot as plt
import scipy
matdata = scipy.io.loadmat(r"C:\Users\peter\Downloads\data_1.mat")

matdata = matdata['someData']

channel1 = matdata[:, 1]
channel2 = matdata[:, 2]

corr = scipy.stats.spearmanr(channel1, channel2)
print('correlation between channel 1 and channel 2:', corr)

data = np.load(r"C:\Users\peter\Downloads\x1.npy")

binwidth = 10
plt.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))
plt.show()

x_lst = []
y_lst = []
for i in range(len(data)):
    x_lst.append(i)
    y_lst.append(data[i])

plt.figure(figsize=(12, 6))  # Increase the figure size
plt.plot(x_lst, y_lst)

# Set labels for the axes
plt.xlabel('time')
plt.ylabel('y')

# Set a title for the plot
plt.title('Time Series')

# Show the plot
plt.show()