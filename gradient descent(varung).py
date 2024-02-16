# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:21:47 2024

@author: varung
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate a simple dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Set learning rate and number of iterations
learning_rate = 0.01
n_iterations = 1000

# Initialize random values for the coefficients
theta = np.random.randn(2, 1)

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# The resulting theta contains the coefficients
print("Intercept and Coefficient (Theta):", theta)

# Predictions using the learned parameters
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)

# Plot the original data and the linear regression line
plt.scatter(X, y)
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
