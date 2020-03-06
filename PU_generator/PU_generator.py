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
  """ Generate PU data from fully labeled data set.
      Labelling assumption: SCAR, i.e, e(x) = P(s=1|y=1,x) = p(s=1|y=1) = c
      Scenarios: case_control or single-training
  """
  def __init__(self):
      pass

  def fit(self,X,y,n_L = None,n_U = None,random_state = 42,pos_label = '1',case_control = True, c = None,n_T = None):
    """ Implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        n_L : Scalar
              Number of labeled examples, for case_control = T
        n_U : Scalar
              Number of Unlabeled examples, for case_control = T
        random_state : Scalar
                        set seed for Pu data generation
        pos_label : default is '1'.
                    make other labels be '0'.
        case_control : Bool
                      Case control scenario or single training data scenario
        c : scalar
            P(s=1|y=1), only needed when case_control = F
        n_T : scalar
            Sample size of single training dataset, for case_control = F
        Returns
        -------
        self
      """
    # print("just arrived in the fit function--",id(X))
    X,y = check_X_y(X,y)
    y = y.astype(str)
    # f(x) = alpha f+(x) + (1-alpha) f-(x), true population
    data = np.concatenate((X, y.reshape(-1,1)), axis=1) 
    np.random.seed(random_state*3) # set seed for random
    np.random.shuffle(data) # save memory space than permutation, since no copy needed
    # print("after check_X_y --",id(X))
    n,p = X.shape
    # f+(x)
    X_1_true = data[data[:,-1] == pos_label ,0:p]
    
    # case control scenario
    if case_control: 
      # sample labeled positive from f+(x)
      np.random.seed(random_state) # set seed for np.random
      row_inx_L = np.random.choice(X_1_true.shape[0], n_L,replace=True)
      X_labeled = X_1_true[row_inx_L,:]
      # sample unlabeled X from f(x)
      np.random.seed(random_state*2)
      row_inx_U = np.random.choice(n, n_U,replace=True) # set seed for np.random
      X_Unlabeled = data[row_inx_U,0:p]
      y_Unlabeled = data[row_inx_U,-1]
      y_Unlabeled = np.where(y_Unlabeled == pos_label,'1','0')
      self.X_1abeled_ , self.X_Unlabeled_, self.prevalence_ ,self.X_true_, self.X_1_true_, self.p_, self.y_Unlabeled_ = X_labeled,X_Unlabeled, X_1_true.shape[0]/n, X, X_1_true, p, y_Unlabeled
    
    else: 
      # sample single training data.
      np.random.seed(random_state*2)
      row_inx_Total = np.random.choice(n, n_T,replace=True) # set seed for np.random
      data_T = data[row_inx_Total,:] # data_T is single training set
      data_T_P = data_T[data_T[:,-1] == pos_label ,:]
      data_T_N = data_T[data_T[:,-1] != pos_label ,:]
      # sample positive labeled.
      np.random.seed(random_state) # set seed for np.random
      row_inx_L = np.random.choice(data_T_P.shape[0],int(data_T_P.shape[0]*c) ,replace=False)
      data_T_P_L = data_T_P[row_inx_L,:]
      data_T_P_U = data_T_P[list(set(range(data_T_P.shape[0])).difference(row_inx_L) ) ]
      # Unlabeled = data_T_P_U + data_T_N
      data_T_U = np.concatenate((data_T_P_U,data_T_N), axis = 0)
      X_Unlabeled = data_T_U[:,0:p]
      y_Unlabeled = data_T_U[:,-1]
      y_Unlabeled = np.where(y_Unlabeled == pos_label,'1','0')
      self.X_1abeled_, self.X_Unlabeled_, self.X_T_, self.prevalence_, self.X_true_, self.X_1_true_,self.p_, self.y_Unlabeled_ = data_T_P_L[:,0:p], X_Unlabeled, data_T[:,0:p], X_1_true.shape[0]/n, X ,X_1_true, p, y_Unlabeled 
      # X_T_ 
    self.case_control_ = case_control
    return self

  def value_count(self):
    # x_labeled vs x_1_true,  
    # x_T or x_Unlabeled vs X_true_
    X_L = pd.DataFrame(self.X_1abeled_) # fl(x)
    X_1_true = pd.DataFrame(self.X_1_true_) # f+(x)
    X_true = pd.DataFrame(self.X_true_) # f(x)
    if self.case_control_:
      XU_or_XT = pd.DataFrame(self.X_Unlabeled_) # fU(x) or fT(x)
    else:
      XU_or_XT = pd.DataFrame(self.X_T_)

    X_true_count = X_true.apply(pd.Series.value_counts)
    X_L_count = X_L.apply(pd.Series.value_counts)
    X_1_true_count = X_1_true.apply(pd.Series.value_counts)
    XU_or_XT_count = XU_or_XT.apply(pd.Series.value_counts)

    return X_true_count,X_L_count,XU_or_XT_count,X_1_true_count

  def plot_dist(self):
    X_true_count,X_L_count,XU_or_XT_count,X_1_true_count = self.value_count()
    # x_T or x_Unlabeled vs X_true_
    X_true_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Population, f(x)",sharex = False)
    if self.case_control_:
      XU_or_XT_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Unlabeled, f(x|s=0)",sharex = False)
    else:
      XU_or_XT_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Single-training, f(x)",sharex = False)
    # x_labeled vs x_1_true,
    X_1_true_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Positive population, f(x|y=1)",sharex = False)
    X_L_count.apply(lambda x: 100 * x / x.sum() ).plot(kind='bar',subplots=True,layout=(int(self.p_/2), 2),title = "Labeled, f(x|s=1)",sharex = False)
    

