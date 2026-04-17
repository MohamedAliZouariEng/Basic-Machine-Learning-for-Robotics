import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

def gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.random.randn(2,1)
    X_b = np.c_[np.ones((m,1)), X]
    cost_history = []

    for iteration in range(iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y.reshape(-1,1))
        theta -= learning_rate * gradients 
        cost = (1/m) * np.sum(X_b.dot(theta) - y.reshape(-1,1) ** 2)
        cost_history.append(cost)

    return theta, cost_history

def plot_regression_line(X, y, theta, learning_rate):
    plt.plot(X, y, 'b.')
    plt.plot(X, theta[0] + theta[1] * X, 'r-', label=f'LR={learning_rate}')
    plt.title(f'Linear Regression (Learning Rate = {learning_rate})')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()


def plot_cost_history(cost_histories, learning_rates):
    for i, cost_history in enumerate(cost_histories):
        plt.plot(cost_history, label=f'LR={learning_rates[i]}')
    plt.title("Cost History for Different Learning Rates")
    plt.xlabel('Iterations')
    plt.ylabel('Cost(MSE)')
    plt.legend()
    plt.show()

learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]

iterations = 15
cost_histories = []

for learning_rate in learning_rates:
    theta, cost_history = gradient_descent(X, y, learning_rate, iterations)
    cost_histories.append(cost_history)

    plot_regression_line(X, y, theta, learning_rate)

plot_cost_history(cost_histories, learning_rates)