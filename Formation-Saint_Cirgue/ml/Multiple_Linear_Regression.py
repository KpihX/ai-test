# %% [markdown]
# # Simple Linear Regression

# %%
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1/ Dataset

# %%
m, n = 100, 2
x, y = make_regression(n_samples=m, n_features=n, noise=10)
plt.scatter(x[:, 0], y, label='y = f(x_1)')
plt.scatter(x[:, 1], y, c='orange', label="y = f(x_2)")
plt.legend()

# %%
print(x.shape)
print(y.shape)

# %%
y = y.reshape(y.shape[0], 1)
print(y.shape)

# %%
# X matrix
X = np.hstack((x, np.ones(x[:, 0].shape + (1,))))
print(X)

# %%
theta = np.random.randn(3, 1)
print(theta.shape)
print(theta)

# %% [markdown]
# ## Model

# %%
def model(X, theta):
    return X.dot(theta)

# %%
Y = model(X, theta)
print(Y)

# %%
plt.scatter(x[:, 0], y, label='y = f(x_1)')
plt.plot(x[:, 0], Y, c='red', label='Y = f(x_1)')
plt.legend()

# %%
plt.scatter(x[:, 1], y, label="y = f(x_2)")
plt.plot(x[:, 1], Y, c='red', label='Y = f(x_2)')
plt.legend()

# %%
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta)-y)**2)

# %%
cost_function(X, y, theta)

# %%
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

# %%
def gradient_descent(X, y, theta, learn_rate, n_iter):
    cost_history = np.zeros(n_iter)
    for i in range(0, n_iter):
        theta = theta - learn_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history

# %%
n_iter = 1000
theta_final, cost_history = gradient_descent(X, y, theta, learn_rate=0.01, n_iter=n_iter)
print(theta_final)

# %%
predictions = model(X, theta_final)

plt.scatter(x[:, 0], y, label="y = f(x_1)")
plt.scatter(x[:, 0], predictions, c='red', label='Y = f(x_1)')
plt.legend()

# %%
plt.scatter(x[:, 1], y, label="y = f(x_2)")
plt.scatter(x[:, 1], predictions, c='red', label='Y = f(x_2)')
plt.legend()

# %% [markdown]
# ## Courbe d'apprentissage

# %%
plt.plot(np.arange(800, n_iter+1), cost_history[799:], c='green')

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:, 0], x[:, 1], y)
ax.scatter(x[:, 0], x[:, 1], predictions)
plt.show()

# %%


# %% [markdown]
# ## Coefficient de détermination

# %%
def determ_coef(y, predictions):
    u = ((y - predictions)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

r = determ_coef(y, predictions)
print(r)


