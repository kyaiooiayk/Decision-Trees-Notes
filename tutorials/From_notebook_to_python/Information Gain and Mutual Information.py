#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Definitions" data-toc-modified-id="Definitions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Definitions</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Information-gain" data-toc-modified-id="Information-gain-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Information gain</a></span></li><li><span><a href="#Mutual-Information" data-toc-modified-id="Mutual-Information-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Mutual Information</a></span><ul class="toc-item"><li><span><a href="#Relation-btw-the-two" data-toc-modified-id="Relation-btw-the-two-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Relation btw the two</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Information Gain and Mutual Information
# 
# </font>
# </div>

# # Definitions
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - **Information gain** is the reduction in entropy or surprise by transforming a dataset and is often used in training decision trees. Information gain is calculated by comparing the entropy of the dataset before and after a transformation.
# 
# - **Mutual information** calculates the statistical dependence between two variables and is the name given to information gain when applied to variable selection.
#     
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


from math import log2


# # Information gain
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Information Gain, or IG for short, measures the reduction in entropy or surprise by splitting a dataset according to a given value of a random variable. 
# 
# - A larger information gain suggests a lower entropy group or groups of samples, and hence less surprise.
# 
# - You might recall that information quantifies how surprising an event is in bits. Lower probability events have more information, higher probability events have less information. Entropy quantifies how much information there is in a random variable, or more specifically its probability distribution. A skewed distribution has a low entropy, whereas a distribution where events have equal probability has a larger entropy.
# 
#     - **Low probability** events are more surprising thus have a larger amount of information -> smaller entropy?
#     - **High probablity** events are less surprising thus have a smaller amouunt of information -> larger entropy?
# 
# </font>
# </div>

# In[2]:


# calculate the entropy for the split in the dataset
def entropy(class0, class1):
    return -(class0 * log2(class0) + class1 * log2(class1))


# split of the main dataset
class0 = 13 / 20
class1 = 7 / 20
# calculate entropy before the change
s_entropy = entropy(class0, class1)
print('Dataset Entropy: %.3f bits' % s_entropy)

# split 1 (split via value1)
s1_class0 = 7 / 8
s1_class1 = 1 / 8
# calculate the entropy of the first group
s1_entropy = entropy(s1_class0, s1_class1)
print('Group1 Entropy: %.3f bits' % s1_entropy)

# split 2  (split via value2)
s2_class0 = 6 / 12
s2_class1 = 6 / 12
# calculate the entropy of the second group
s2_entropy = entropy(s2_class0, s2_class1)
print('Group2 Entropy: %.3f bits' % s2_entropy)

# calculate the information gain
gain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)
print('Information Gain: %.3f bits' % gain)


# # Mutual Information
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable. 
# 
# - The mutual information between two random variables X and Y can be stated formally as follows: `I(X ; Y) = H(X) – H(X | Y)`
# 
# - It measures the average reduction in uncertainty about x that results from learning the value of y; or **vice versa**, the average amount of information that x conveys about y.
# 
# </font>
# </div>

# ## Relation btw the two
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# -  Notice the similarity in the way that the mutual information is calculated and the way that information gain is calculated; they are equivalent:
# 
#     - `I(X ; Y) = H(X) – H(X | Y)`
#     - `IG(S, a) = H(S) – H(S | a)`
# 
# - As such, mutual information is sometimes used as a synonym for information gain. Technically, they calculate the same quantity if applied to the same data.
# 
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://machinelearningmastery.com/information-gain-and-mutual-information/
# 
# </font>
# </div>

# # Requirements
# <hr style="border:2px solid black"> </hr>

# In[3]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv -m')


# In[ ]:




