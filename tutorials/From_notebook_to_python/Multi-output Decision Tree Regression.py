#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Multi-output Decision Tree Regression

Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html#sphx-glr-auto-examples-tree-plot-tree-regression-multioutput-py
"""


# # Import modules

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


# # Create a random dataset

# In[2]:


rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))


# In[6]:


import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 15
plt.scatter(y[:,0], y[:, 1])
plt.figure()


# # Fit regression model

# In[7]:


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)


# # Predict

# In[20]:


X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
print(len(X_test))
y_1 = regr_1.predict(X_test)
print(len(y_1))
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)


# # Plotting

# In[22]:


# Plot the results
plt.figure()
s = 25
plt.scatter(y[:, 0], y[:, 1], c = "k", s=s, label="data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="b", s=s, label="max_depth=2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s, label="max_depth=5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="g", s=s, label="max_depth=8")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()


# In[ ]:




