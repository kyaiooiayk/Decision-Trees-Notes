#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
What? How to visualise decision tree

IMPORTANT NOTE: If working on MAC please use this command: brew install graphviz

Reference: XGBoost with python, Jason Brownle
           https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
"""


# In[12]:


# Import python modules
import os
from numpy import loadtxt
from xgboost import plot_tree
from matplotlib import pyplot
from xgboost import XGBClassifier
from IPython.display import Markdown, display


# In[13]:


# Additional cosmetic function
def myPrint(string, c = "blue"):    
    """My version of the python-native print command.
    
    Print in bold and red tect
    """
    colorstr = "<span style='color:{}'>{}</span>".format(c, '**'+ string + '**' )    
    display(Markdown(colorstr))
    
def printPythonModuleVersion():    
    """printPythonModuleVersion
    Quickly list the python module versions
    """
    myPrint("Checking main python modules version")
    import scipy
    print('scipy: %s' % scipy.__version__)
    import numpy
    print('numpy: %s' % numpy.__version__)    
    import matplotlib
    print('matplotlib: %s' % matplotlib.__version__)    
    import pandas
    print('pandas: %s' % pandas.__version__)
    import statsmodels
    print('statsmodels: %s' % statsmodels.__version__) 
    import sklearn
    print('sklearn: %s' % sklearn.__version__)
    import xgboost
    print('xgboostn: %s' % xgboost.__version__)

printPythonModuleVersion()


# In[14]:


myPrint("Loading dataset")
dataset = loadtxt('../DATASETS/pima-indians-diabetes.csv', delimiter=",") 
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]     


# In[15]:


# fit model on training data
model = XGBClassifier()
model.fit(X, y)


# In[20]:


# plot single tree
rcParams['font.size'] = 20
rcParams['figure.figsize'] = 20, 20

plot_tree(model, num_trees = 4, rankdir='LR')
pyplot.show()


# In[ ]:




