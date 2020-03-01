# from google.colab import drive
# drive.mount('/content/drive')
# !pip install shap
# !pip install pyitlib
# import os
# os.path.abspath(os.getcwd())
# os.chdir('/content/drive/My Drive/Protein project')
# os.path.abspath(os.getcwd())

#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on Mar 1, 2020
@author: Chengning Zhang
"""
import warnings
warnings.filterwarnings("ignore")
from __future__ import division ###for float operation
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score ##tp / (tp + fn)
from sklearn.metrics import precision_score #tp / (tp + fp)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
#from pyitlib import discrete_random_variable as drv
import time 
import timeit 
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted ### Checks if the estimator is fitted by verifying the presence of fitted attributes (ending with a trailing underscore)
#from sklearn.utils.multiclass import unique_labels, not necessary, can be replaced by array(list(set()))


class PUgenerator:
  def __init__(self):
      pass

  def fit(self,X,y,n_L,n_U,random_state = 42,pos_label = '1'):
    """ Implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        n_L : Scalar
              Number of labeled examples
        n_U : Scalar
              Number of Unlabeled examples
        random_state : Scalar
              set seed for Pu data generation
        pos_label : 

        Returns
        -------
        
      """
    # print("just arrived in the fit function--",id(X))
    X,y = check_X_y(X,y)
    data = np.concatenate((X, y.reshape(-1,1)), axis=1)
    # print("after check_X_y --",id(X))
    n,p = X.shape
    y = y.astype(str)
    # row_inx_0 = [row for row in range(n) if y[row] == '0']
    row_inx_1 = [row for row in range(n) if y[row] == pos_label]
    
    # sample labeled positive from X_1_total
    X_1_total = X[row_inx_1,:]
    np.random.seed(random_state) # set seed for np.random
    row_inx_L = np.random.choice(X_1_total.shape[0], n_L,replace=True)
    X_labeled = X_1_total[row_inx_L,:]

    # sample from unlabeled X
    np.random.seed(random_state*3) # set seed for random
    np.random.shuffle(data) # save memory space than permutation, since no copy needed
    # print("after permutation--",id(X))
    np.random.seed(random_state*2)
    row_inx_U = np.random.choice(n, n_U,replace=True) # set seed for np.random
    X_Unlabeled = data[row_inx_U,0:p]
    y_Unlabeled = data[row_inx_U,-1]
    y_Unlabeled = np.where(y_Unlabeled == pos_label,'1','0')
    self.X_1abeled_ , self.X_Unlabeled_, self.prevalence_ ,self.X_true_, self.X_1_true, self.p_, self.y_Unlabeled_ = X_labeled,X_Unlabeled, len(row_inx_1)/n, X, X_1_total,p,y_Unlabeled
    return self

  def value_count(self):
    X_true = pd.DataFrame(self.X_true_)
    X_L = pd.DataFrame(self.X_1abeled_)
    X_U = pd.DataFrame(self.X_Unlabeled_)
    X_1_true = pd.DataFrame(self.X_1_true)

    X_true_count = X_true.apply(pd.Series.value_counts)
    X_L_count = X_L.apply(pd.Series.value_counts)
    X_U_count = X_U.apply(pd.Series.value_counts)
    X_1_true_count = X_1_true.apply(pd.Series.value_counts)
    return X_true_count,X_L_count,X_U_count,X_1_true_count

  def plot_dist(self):
    X_true_count,X_L_count,X_U_count,X_1_true_count = self.value_count()
    
    # X_true_count  vs X_U_count
    X_true_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Population, f(x)",sharex = False)
    X_U_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Unlabeled, f(x|s=0)",sharex = False)
    
    #print(15*'__')
    # X_1_true_count vs X_L_count
    X_1_true_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Positive population, f(x|y=1)",sharex = False)
    X_L_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Labeled, f(x|s=1)",sharex = False)
