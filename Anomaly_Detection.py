#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection
# 
# implement the anomaly detection algorithm and apply it to detect failing servers on a network. 
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from utils import *



def estimate_gaussian(X): 
   m, n = X.shape

   mu = 1 / m * np.sum(X, axis = 0)
   var = 1 / m * np.sum((X - mu) ** 2, axis = 0)

   return mu, var


#

def select_threshold(y_val, p_val): 
   best_epsilon = 0
   best_F1 = 0
   F1 = 0

   step_size = (max(p_val) - min(p_val)) / 1000

   for epsilon in np.arange(min(p_val), max(p_val), step_size):

    
       predictions = (p_val < epsilon)

       tp = np.sum((predictions == 1) & (y_val == 1))
       fp = sum((predictions == 1) & (y_val == 0))
       fn = np.sum((predictions == 0) & (y_val == 1))

       prec = tp / (tp + fp)
       rec = tp / (tp + fn)

       F1 = 2 * prec * rec / (prec + rec)
     
       if F1 > best_F1:
           best_F1 = F1
           best_epsilon = epsilon

   return best_epsilon, best_F1
# ### 2.4 High dimensional dataset
# 

# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()



# 
# Let's check the dimensions of these new variables to become familiar with the data



print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)


# #### Anomaly detection 
# 
# Now, let's run the anomaly detection algorithm on this new dataset.
# 
# The code below will use your code to 
# * Estimate the Gaussian parameters ($\mu_i$ and $\sigma_i^2$)
# * Evaluate the probabilities for both the training data `X_train_high` from which you estimated the Gaussian parameters, as well as for the the cross-validation set `X_val_high`. 
# * Finally, it will use `select_threshold` to find the best threshold $\varepsilon$. 

# In[ ]:


# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))


