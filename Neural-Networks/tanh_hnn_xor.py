import numpy as np

def f(x):
    return np.tanh(x)


def grad(y):
    return 1 - y**2

n_hn = 2
np.random.seed(9)
weights_1 = 2 * np.random.random((n_hn,2)) - 1
weights_2 = 2 * np.random.random((1,n_hn)) - 1

np.random.seed(54)
bias_1 = 2 * np.random.random((n_hn,1)) - 1
bias_2 = 2 * np.random.random((1,1)) - 1

lr = 0.01
input = np.array([[1,1],
                  [1,0],
                  [0,1],
                  [0,0]]).T

output = np.array([[0,1,1,0]])

for j in range(20000):
    intermediate = f(np.dot(weights_1, input) + bias_1)
    trial = f(np.dot(weights_2, intermediate) + bias_2)
    E = output - trial
    col = weights_2.T * grad(intermediate)

    weights_2 += np.dot(E * grad(trial), intermediate.T) * lr
    weights_1 += np.dot(E * grad(trial) * col, input.T) * lr
    bias_1 += np.sum(E * grad(trial) * col, axis=1, keepdims=True) * lr
    bias_2 += np.sum(E * grad(trial), axis=1, keepdims=True) * lr
print(trial)
