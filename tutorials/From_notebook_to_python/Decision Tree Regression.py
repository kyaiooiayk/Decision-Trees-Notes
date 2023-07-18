#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Decision Tree Regression

Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
"""


# # Import modules

# In[13]:


# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from matplotlib import rcParams


# # Create data & noise

# In[25]:


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y_nf = np.sin(X).ravel()
# from [[],[]] to []
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# In[17]:


rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 15
plt.plot(X,y)
plt.plot(X,y_nf)


# # Modelling

# In[8]:


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)


# In[9]:


# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)


# # Plotting

# In[21]:


# Plot the results
rcParams['figure.figsize'] = 15, 5
rcParams['font.size'] = 15

plt.figure()

plt.scatter(X, y, s=20, c="r", label="data")
plt.plot(X_test, y_1, "g-*", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, "b-*", label="max_depth=5", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# In[ ]:


"""
We can see that if the maximum depth of the tree (controlled by the max_depth parameter) is set too high, the 
decision trees learn too fine details of the training data and learn from the noise, i.e. they overfit.
"""

