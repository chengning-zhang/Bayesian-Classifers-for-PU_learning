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


def get_cv(X, y, case_control = True, nl = None, nu = None, c = None,n_T = None, random_state = 42, pos_label = '1', cls = None, M = None, runs=10, verbose = True):  
  """ Get CLL, accuracy, precision, recall under multiple runs
      Parameters
      ----------
      cls:  ___{PNB, PTAN,PSTAN, WNB, WTAN}___
            Model class
      X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix.
      Y : {array-like, sparse matrix}, shape (n_samples,)
            Target.
      M : {array-like, sparse matrix}, shape (n_features, n_features)
            Contact matrix
      nl : Scalar
            Number of labeled examples
      nu : Scalar
            Number of Unlabeled examples
      runs : Scalar
            Number of runs, all metrics are average of all runs.
      random_state : Scalar
            Random state on PU generator, make results reproducible
      Pos_label : str type
            Positive label in data. Default is '1'
      case_control : bool

      Returns
      -------
      CLL, accuracy, precision, recall of all runs.
      
  """
  X,y = check_X_y(X,y)
  Accuracy = []
  Precision = []
  Recall = []
  CLL = []
  for i in range(runs):
    # generate PU data, with different seed i*random_state
    pu_object = PUgenerator()
    pu_object.fit(X,y,case_control,nl,nu,c,n_T,random_state*(i+1),pos_label) # generate case control or single training data
    model = cls()
    model.fit(pu_object.X_1abeled_,pu_object.X_Unlabeled_, pu_object.prevalence_,M, case_control)
    Accuracy.append(accuracy_score(pu_object.y_Unlabeled_,model.predict(pu_object.X_Unlabeled_)) )
    CLL.append(model.Conditional_log_likelihood(pu_object.y_Unlabeled_,model.predict_proba(pu_object.X_Unlabeled_)) )
    Precision.append(precision_score(pu_object.y_Unlabeled_,model.predict(pu_object.X_Unlabeled_), 
         average='macro') )
    Recall.append(recall_score(pu_object.y_Unlabeled_,model.predict(pu_object.X_Unlabeled_), 
         average='macro' ) )
    if verbose:
        print("accuracy in %s -th run is %s" % (i+1,Accuracy[i]) )
        print("CLL in %s -th run is %s" % (i+1,CLL[i]))
        print("Precision in %s -th run is %s" % (i+1,Precision[i]) )
        print("Recall in %s -th run is %s" % (i+1,Recall[i]) )
        if not case_control:
          print("estimated p(s=1|y=1) in %s -th run = %s" % (i+1, model.c_ ))
        print(10*'__')
  return np.array(Accuracy), np.array(CLL), np.array(Precision),np.array(Recall)
