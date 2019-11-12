#!/usr/bin/env python
# coding: utf-8

# # Predict geothermal heat flux: KNN regression

# References:
# * http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
# * https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# * https://github.com/amirkdv/ghf-greenland-gbrt
# 
# Ensure R17_global_test.csv and R17_global_train.csv is in the same directory as this notebook.
# 
# Install mlxtend to use SequentialFeatureSelector (pip install mlxtend).

# In[1]:


import numpy as np
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
from timeit import default_timer as timer
import math


# Load the data.

# In[2]:


X_idx = [3,4,5,6,7,8,9,10,12,13,17,19,20,21,22,23,24] #feature indices
y_idx = 16 # GHF index

# Load labels
labels = []
with open('R17_global_train.csv') as f:
    labels = np.array(f.readline().strip().split(',')).take(X_idx)

# Load data
data_train = np.loadtxt('R17_global_train.csv', delimiter=',', skiprows=1)
X_train = data_train.take(X_idx, axis=1)
y_train = data_train.take(y_idx, axis=1)

data_test = np.loadtxt('R17_global_test.csv', delimiter=',', skiprows=1)
X_test = data_test.take(X_idx, axis=1)
y_test = data_test.take(y_idx, axis=1)

# Scale X
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(labels)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


# Use sequential feature selection to decide what features to use. Grid search to determine best hyperparameter values.

# In[3]:


knn = KNeighborsRegressor()
sfs1 = SFS(estimator=knn, 
           k_features='best',
           forward=False, 
           floating=True,
           cv=5)

pipe = Pipeline([('sfs', sfs1), 
                 ('knn', knn)])

param_grid = [
  {'sfs__estimator__n_neighbors': range(1, len(X_idx)),
   'sfs__estimator__weights': ['distance', 'uniform'],
   'sfs__estimator__metric': ['euclidean', 'manhattan', 'chebyshev']}
  ]

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid,  
                  n_jobs=-1, 
                  cv=5,
                  iid=True,
                  refit=True)
start = timer()
gs = gs.fit(X_train, y_train)
end = timer()
print('Grid search completed in %.2f minutes' % ((end - start) / 60.0))

print('Best score:', gs.best_score_)

# Save indexes best features
feature_idx = gs.best_estimator_.steps[0][1].k_feature_idx_
print('Best Features:', [labels[i] for i in feature_idx])
print('Best Params:', gs.best_params_)


# Fit a KNN model using the best hyperparameter values.

# In[4]:


# Create subsets of train/test data using best features from SFS
best_knn = gs.best_estimator_.named_steps['sfs'].estimator
X_train_selected = X_train.take(feature_idx, axis=1)
X_test_selected = X_test.take(feature_idx, axis=1)

knn = KNeighborsRegressor(n_neighbors=best_knn.n_neighbors,
                          weights=best_knn.weights,
                          metric = best_knn.metric)

print(knn)
knn.fit(X_train_selected, y_train)
print('R^2:', knn.score(X_test_selected, y_test))
pred = knn.predict(X_test_selected)
rmse = (1 / np.average(y_test)) * math.sqrt(np.average((y_test - pred)**2))
print ('RMSE:', rmse)


# In[ ]:




