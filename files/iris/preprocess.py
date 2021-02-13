import numpy as np

x = np.loadtxt("iris.data").T
y = np.loadtxt("iris-y.data")

indices = np.random.choice(x.shape[1], 150, replace=False)

x_train = x[:, indices[:120]]
x_test = x[:, indices[120:]]

y_train = y[indices[:120]].reshape(1, 120)
y_test = y[indices[120:]].reshape(1, 30)

np.savetxt("train-x.txt", x_train)
np.savetxt("test-x.txt", x_test)
np.savetxt("train-y.txt", y_train)
np.savetxt("test-y.txt", y_test)
