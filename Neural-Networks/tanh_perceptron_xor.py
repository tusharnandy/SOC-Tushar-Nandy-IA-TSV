import numpy as np

# AN ATTEMPT TO USE Hyperbolic tan IN MY PERCEPTRON

def f(x):
    return np.tanh(x)


def grad(y):
    return 1 - y**2


act_input = np.array([[1,1],
                      [1,0],
                      [0,1],
                      [0,0]])

act_output = np.array([[0,1,1,0]]).T

w = 2 * np.random.random((2, 1)) - 1

for i in range(300000):

    input = act_input
    output = f(np.dot(input, w))
    error = act_output - output
    change = error * grad(output)

    w += np.dot(input.T, change)

print(f"Output after training: \n{output}")
print(w)
