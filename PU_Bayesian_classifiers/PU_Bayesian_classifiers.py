# from google.colab import drive
# drive.mount('/content/drive')
# !pip install shap
# !pip install pyitlib
# import os
# os.path.abspath(os.getcwd())
# os.chdir('/content/drive/My Drive/Protein project')
# os.path.abspath(os.getcwd())

import warnings
warnings.filterwarnings("ignore")
from __future__ import division  # for float operation
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score  # tp / (tp + fn)
from sklearn.metrics import precision_score # tp / (tp + fp)
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



class Bayes_net_PU(BaseEstimator, ClassifierMixin): 
    """
    Bayesian network implementation for Postitive Unlabeled examples
    API inspired by SciKit-learn.
    """
    def predict_proba(self, X): ### key prediction methods, all other prediction methods will use it first.
      raise NotImplementedError

    def predict(self, X):
      """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,1)
            Predicted target values for X
      """
      Prob_1 = self.predict_proba(X) 
      return(np.where(Prob_1 > 0.5, '1', '0'))

    def Conditional_log_likelihood(self,y_true,y_pred_prob): 
      """Calculate the conditional log likelihood.
      :param y_true: The true class labels. e.g ['1','1',.....'0','0']
      :param y_pred_prob: np.array shows prob of class '1' for each instance.
      :return: CLL. A scalar.
      """
      cll = []
      for i in range(len(y_pred_prob)):
        cll.append(y_pred_prob[i] if y_true[i] == '1' else 1-y_pred_prob[i] )

      cll = [np.log2(ele) for ele in cll]
      cll = np.array(cll)
      return(sum(cll))
 
    def plot_tree_structure(self,mapping = None,figsize = (5,5)):
      check_is_fitted(self)
      parent = self.parent_
      egdes = [(k,v) for v,k in parent.items() if k is not None]
      G = nx.MultiDiGraph()
      G.add_edges_from(egdes)
      #mapping=dict(zip(range(8),['b0','b1','b2','b3','b4','b5','b6','b7']))
      plt.figure(figsize=figsize)
      nx.draw_networkx(G,nx.shell_layout(G))




class PNB(Bayes_net_PU):
  name = "PNB"
  def __init__(self, alpha = 1):
      self.alpha = alpha

  def fit(self,X_L, X_u, pri, M = None):  
    """ Implementation of a fitting function.
        Parameters
        ----------
        X_l : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input positive labeled samples.
        X_u : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input unlabeled samples.
        pri : scalar.
            The prevalence probability (p(y = 1))
        Returns
        -------
        self : object
            Returns self.
      """

    X_L = check_array(X_L)
    X_u = check_array(X_u)
    if X_L.shape[1] != X_u.shape[1]:
      raise ValueError('labeled data and unlabeled data have different number of features ')
    # 1: Learned from positive examples, P(xij|1) = N_L(xij)/N_L.  N_L(xij) 
    # 2: Learned from Unlabeled examples, N_U(xij)
    # 3: P(xi = j|c = 0), Listprob0, calculated from previous list
    n_L,p = X_L.shape
    n_u,p = X_u.shape
    #
    List_count_1 = {} 
    List_prob_1 = {} # {x0:{'1': p(x0 =1|y=1), '2': p(x0 =2|y=1), 'else': }, x1:{},   ... x7:{} }
    #
    List_count_U = {} 
    #
    List_prob_0 = {} # P(xi = j|c=0)
    K = {} # X_i_L and X_i_u contains all possible values of x_i, there are not other values, different from supervised setting.
    for i in range(p):
      x_i_L = X_L[:,i]
      x_i_u = X_u[:,i]
      x_i_L_counter = Counter(x_i_L) # may be not need key error
      x_i_u_counter = Counter(x_i_u)
      x_i_values = set(x_i_L_counter.keys()).union(x_i_u_counter.keys()) # all possible values of x_i
      K[i] = len(list(x_i_values))
      # part 1
      x_i_L_prob = {key: (value + self.alpha) / (n_L + self.alpha * (K[i]) ) for key,value in x_i_L_counter.items()} # p(x|s=1) = p(x|y=1)
      x_i_L_prob.update({key: (0 + self.alpha) / (n_L + self.alpha * (K[i]) )  for key in list(x_i_values) if key not in list(x_i_L_counter.keys()) } )
      List_prob_1[i] = x_i_L_prob
      List_count_1[i] = x_i_L_counter
      # part 2
      List_count_U[i] = x_i_u_counter
      # part 3
      x_i_0_prob = {key: max([0,x_i_u_counter[key] - x_i_L_prob[key] * pri * n_u]) for key in list(x_i_values)} # numeritor, can be negative, make it >=0
      x_i_0_prob = {key:(self.alpha + value)/ (K[i]*self.alpha + n_u * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
      x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
      List_prob_0[i] = x_i_0_prob
      # x_i_0_prob = {key: value/sum(np.array(list(x_i_0_prob.values()))) for key,value in x_i_0_prob.items()    }
    self.is_fitted_ = True  
    self.n_features_, self.K_, self.List_count_1_,self.List_prob_1_, self.List_count_U_, self.List_prob_0_, self.prevalence_ = p, K, List_count_1,List_prob_1,List_count_U,List_prob_0, pri
    return self

  def predict_proba(self,X): 
    """
        Return probability estimates for the test vector X. Usually it would be X_unlabeled
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, )
            Returns the probability of the samples for each class in
            the model. 
    """
    check_is_fitted(self)
    X = check_array(X)
    Prob_1 = []
    for ins in X:
      P1 = self.prevalence_ # don't need copy, immutable
      P0 = 1 - P1
      for i in range(self.n_features_):
        P1 = P1 * (self.List_prob_1_[i][ins[i]]) 
        P0 = P0 * (self.List_prob_0_[i][ins[i]]) 
        # normalize proba
      P = P1 + P0
      P1 = P1/P; P0 = P0/P
      Prob_1.append(P1)

    Prob_1 = np.array(Prob_1) # for shap 
    return Prob_1

    



class PTAN(Bayes_net_PU):
    name = "PTAN"
    def __init__(self, alpha = 1,starting_node = 0):
      self.starting_node = starting_node
      self.alpha = alpha
    
    def get_mutual_inf(self,X_L, X_u, pri, M = None):
      """get PU conditional mutual inf of all pairs of features, part of training
        Parameters
        ----------
        X_l : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input positive labeled samples.
        X_u : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input unlabeled samples.
        pri : scalar.
            The prevalence probability (p(y = 1))
        Returns
        -------
        np.array matrix.
      """
      X_L = check_array(X_L)
      X_u = check_array(X_u)
      if X_L.shape[1] != X_u.shape[1]:
        raise ValueError('labeled data and unlabeled data have different number of features ')
      n_L,p = X_L.shape
      n_u,p = X_u.shape
      M = np.zeros((p,p)) # will not change global M, since new memory assigned for this local M
      # part 1: proba that can be estimated from labeled examples. 1 P(xij,xkl|1), 2 p(xj|1), 3 p(xkl|1). P(xij,xkl|1) = N_L(xi=j,xk=l)/N_L
      # part 2: P(xij,xkl) from U, P(xij,xkl) = N_U(xij,xkl) / n_U
      # part 3: p(xij,xkl|0),p(xij|0),p(xkl|0), same as PNB, from previous list
      #
      # List_prob_xi_xj_1 = {} # p(xij,xkl|c =1) = N_L(xij,xkl) / N_L and  p(xij|c =1) = N_L(xij)/N_L
      # List_count_xi_xj_1 = {} # N_L(xij,xkl) and N_L(xij)
      # 
      # List_prob_xi_xj_U = {} # P(xij,xkl) = N_U(xij,xkl)/n_u
      # List_count_xi_xj_U = {} # N_U(xij,xkl) and N_U(xij)
      # 
      # List_prob_xi_xj_0 = {} # p(xij,xkl|0),and p(xij|0) obtained from previous lists
      K = {}
      X_values = {}
      for i in range(p):
        x_i_L = X_L[:,i]
        x_i_u = X_u[:,i]
        x_i_L_counter = Counter(x_i_L) # may be not need key error
        x_i_u_counter = Counter(x_i_u) # N_U(xi = j)
        x_i_values = list(set(x_i_L_counter.keys()).union(x_i_u_counter.keys()))
        K_i = len(list(x_i_values))
        K[i] = K_i 
        X_values[i] = x_i_values
        # part 1, p(xij|1) and N_L(xi = j)
        x_i_L_prob = {key: (value + self.alpha) / (n_L + self.alpha * (K[i]) ) for key,value in x_i_L_counter.items()} # p(xi= j|s=1) = p(x|y=1)
        x_i_L_prob.update({key: (0 + self.alpha) / (n_L + self.alpha * (K[i]) )  for key in x_i_values if key not in list(x_i_L_counter.keys()) } )
        # List_prob_xi_xj_1[(i,i)] = x_i_L_prob
        # List_count_xi_xj_1[(i,i)] = x_i_L_counter
        # part 2, learn from U, N_U(xij) ,N_U(xij,xkl)
        xi_prob_U = {key: (self.alpha + value) / (K_i*self.alpha + n_u)   for key,value in x_i_u_counter.items()} # P(xij)
        # List_prob_xi_xj_U[(i,i)] = xi_prob_U 
        # List_count_xi_xj_U[(i,i)] = x_i_u_counter
        # part 3, p(xi =j | y=0)
        x_i_0_prob = {key: max([0,x_i_u_counter[key] - x_i_L_prob[key] * pri * n_u]) for key in x_i_values} # N_U(xi =j) - N_u*p(xij, y =1) = N_U(xij,y=0)   numeritor, can be negative, make it >=0
        x_i_0_prob = {key:(self.alpha + value)/ (K[i]*self.alpha + n_u * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
        x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
        # List_prob_xi_xj_0[(i,i)] = x_i_0_prob
        for j in range(i+1,p):
          x_j_L = X_L[:,j]
          x_j_u = X_u[:,j]
          x_j_L_counter = Counter(x_j_L) # may be not need key error
          x_j_u_counter = Counter(x_j_u)
          x_j_values = list(set(x_j_L_counter.keys()).union(x_j_u_counter.keys()))
          K_j = len(list(x_j_values))
          x_j_L_prob = {key: (value + self.alpha) / (n_L + self.alpha * (K_j) ) for key,value in x_j_L_counter.items()} # p(xj= sth|s=1) = p(x|y=1)
          x_j_L_prob.update({key: (0 + self.alpha) / (n_L + self.alpha * (K_j) )  for key in x_j_values if key not in list(x_j_L_counter.keys()) } )
          # part 3, p(xi =j | y=0)
          x_j_0_prob = {key: max([0,x_j_u_counter[key] - x_j_L_prob[key] * pri * n_u]) for key in x_j_values} # numeritor, can be negative, make it >=0
          x_j_0_prob = {key:(self.alpha + value)/ (K_j*self.alpha + n_u * (1-pri) ) for key,value in x_j_0_prob.items()} # add psudo count and divied by dem
          x_j_0_prob = {key: value/(sum(np.array(list(x_j_0_prob.values()))))   for key,value in x_j_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
          
          # part 1 P(xij,xkl|1) = N_L(xi=j,xk=l)/N_L and N_L(xi=j,xk=l)
          xi_xj_count_1 = {(v1,v2): X_L[(X_L[:,i] == v1) & (X_L[:,j] == v2) ].shape[0]  for v1 in x_i_values for v2 in x_j_values} # N_L(xi = j, xk = l)
          xi_xj_prob_1 = {key: (self.alpha + value) / (K_i*K_j*self.alpha + n_L)   for key,value in xi_xj_count_1.items()} # p(xij,xkl|1)
          # List_prob_xi_xj_1[(i,j)] = xi_xj_prob_1
          # List_count_xi_xj_1[(i,j)] = xi_xj_count_1
          # part 2, learn from U,  N_U(xij,xkl)
          xi_xj_count_U = {(v1,v2): X_u[(X_u[:,i] == v1) & (X_u[:,j] == v2) ].shape[0]   for v1 in x_i_values for v2 in x_j_values} # N_U(xi = j, xk = l)
          xi_xj_prob_U = {key: (self.alpha + value) / (K_i*K_j*self.alpha + n_u)   for key,value in xi_xj_count_U.items()} # P(xij,xkl)
          # List_prob_xi_xj_U[(i,j)] = xi_xj_prob_U 
          # List_count_xi_xj_U[(i,j)] = xi_xj_count_U
          # part 3, p(xi = j,xk =l |0) 
          xi_xj_prob_0 = {(v1,v2): max([0, xi_xj_count_U[(v1,v2)] - xi_xj_prob_1[(v1,v2)] * pri * n_u ])   for v1 in x_i_values for v2 in x_j_values}# numeritor, can be negative, make it >=0
          xi_xj_prob_0 = {key: (self.alpha + value)/ (K_j*K_i*self.alpha + n_u * (1-pri) )   for key,value in xi_xj_prob_0.items()} # add psudo count and divied by dem
          xi_xj_prob_0 = {key: value/(sum(np.array(list(xi_xj_prob_0.values()))))   for key,value in xi_xj_prob_0.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
          # List_prob_xi_xj_0[(i,j)] = xi_xj_prob_0
          # M[i,j]
          M[i,j] = sum( np.array([pri* xi_xj_prob_1[(v1,v2)]* np.log( xi_xj_prob_1[(v1,v2)]/(x_i_L_prob[v1]* x_j_L_prob[v2]) ) + 
          (xi_xj_prob_U[(v1,v2)] - pri* xi_xj_prob_1[(v1,v2)] )* np.log(xi_xj_prob_0[(v1,v2)] / ( x_i_0_prob[v1]*x_j_0_prob[v2] ) )
          for v1 in x_i_values for v2 in x_j_values] ) )
          M[j,i] = M[i,j]
          # for bug, x1, x3
          # if i == 1 and j == 3:
          #  part1 = [pri* xi_xj_prob_1[(v1,v2)]* np.log( xi_xj_prob_1[(v1,v2)]/(x_i_L_prob[v1]* x_j_L_prob[v2]) )  
          #            for v1 in x_i_values for v2 in x_j_values]
          #  part2 = [(xi_xj_prob_U[(v1,v2)] - pri* xi_xj_prob_1[(v1,v2)] )* np.log(xi_xj_prob_0[(v1,v2)] / ( x_i_0_prob[v1]*x_j_0_prob[v2] ) )
          #            for v1 in x_i_values for v2 in x_j_values]
    
      # self.n_L_,self.n_features_, self.n_U_, self.M_,self.List_prob_xi_xj_1_, self.List_count_xi_xj_1_,self.List_prob_xi_xj_U_,self.List_count_xi_xj_U_,self.List_prob_xi_xj_0_,self.K_,self.prior_  = n_L,p,n_u,M,List_prob_xi_xj_1,List_count_xi_xj_1,List_prob_xi_xj_U,List_count_xi_xj_U,List_prob_xi_xj_0,K,pri 
      # self.part1,self.part2 = part1,part2
      # return n_L,p,n_u,M,List_prob_xi_xj_1,List_count_xi_xj_1,List_prob_xi_xj_U,List_count_xi_xj_U,List_prob_xi_xj_0,K,pri 
      return n_L,p,n_u,M,K,X_values

    def Findparent(self,X_L, X_u, pri, M = None):
      # n_L,p,n_u,M,List_prob_xi_xj_1,List_count_xi_xj_1,List_prob_xi_xj_U,List_count_xi_xj_U,List_prob_xi_xj_0,K,pri = self.get_mutual_inf(X_L, X_u, pri)
      n_L,p,n_u,M,K,x_values = self.get_mutual_inf(X_L, X_u, pri)
      np.fill_diagonal(M,0)  
      V = range(p) # set of all nodes
      st = self.starting_node
      Vnew = [st]  # vertex that already found their parent. intitiate it with starting node. TAN randomly choose one
      parent = {st:None} # use a dict to show nodes' interdepedency
      while set(Vnew) != set(V): # when their are still nodes whose parents are unknown.
        index_i = [] # after for loop, has same length as Vnew, shows the closest node that not in Vnew with Vnew.  
        max_inf = [] # corresponding distance
        for i in range(len(Vnew)):  # can be paralelled 
          vnew = Vnew[i]
          ListToSorted = [e for e in M[:,vnew]] # 
          index = sorted(range(len(ListToSorted)),key = lambda k: ListToSorted[k],reverse = True)
          index_i.append([ele for ele in index if ele not in Vnew][0]) 
          max_inf.append(M[index_i[-1],vnew])
      
        index1 = sorted(range(len(max_inf)),key = lambda k: max_inf[k],reverse = True)[0] ## relative position, Vnew[v1,v2] index_i[v4,v5] max_inf[s1,s2] index1 is the position in those 3 list
        Vnew.append(index_i[index1]) # add in that node
        parent[index_i[index1]] = Vnew[index1] # add direction, it has to be that the new added node is child, otherwise some nodes has 2 parents which is wrong.
      
      #return parent,n_L,p,n_u,M,List_prob_xi_xj_1,List_count_xi_xj_1,List_prob_xi_xj_U,List_count_xi_xj_U,List_prob_xi_xj_0,K,pri
      return parent,n_L,p,n_u,M,K,x_values

    
    def fit(self,X_L, X_u, pri, M = None):  # this is based on trainning data !!!
      parent,n_L,p,n_u,M,K,x_values = self.Findparent(X_L, X_u, pri)
      # part 1: proba that can be estimated from labeled examples. 1 P(xij|1,xkl), 2 p(x_root|1) = N_L(x_root)/N_L,  P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
      # part 2: learn from U, N_U(xij,xkl), and N_U(xkl)
      # part 3: p(xij|0,xkl),p(x_root|0) from previous list
      #
      List_prob_1 = {} # 1 P(xij|1,xkl), 2 p(x_root|1)  
      List_count_1 = {} # N_L(xij,xpal) and N_L(xij)
      # 
      List_count_U = {} # N_U(xij,xkl) and N_U(xij)
      # 
      List_prob_0 = {} # p(xij|0,xkl),p(x_root|0)
      # for root node
      root_i = self.starting_node
      x_i_values = x_values[root_i]
      # part 1 
      x_i_L = X_L[:,root_i]
      x_i_L_counter = Counter(x_i_L)
      x_i_L_prob = {key: (x_i_L_counter[key]+self.alpha)/(K[root_i]*self.alpha + n_L ) for key in x_i_values}
      List_prob_1[root_i] = x_i_L_prob
      List_count_1[root_i] = x_i_L_counter
      # part 2 
      x_i_u = X_u[:,root_i]
      x_i_u_counter = Counter(x_i_u)
      List_count_U[root_i] = x_i_u_counter
      # part 3 
      x_i_0_prob = {key: max([0,x_i_u_counter[key] - x_i_L_prob[key] * pri * n_u]) for key in x_i_values} # N_U(xi =j) - N_u*p(xij, y =1) = N_U(xij,y=0)   numeritor, can be negative, make it >=0
      x_i_0_prob = {key:(self.alpha + value)/ (K[root_i]*self.alpha + n_u * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
      x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
      List_prob_0[root_i] = x_i_0_prob
      #
      for i in [e for e in range(0,p) if e != root_i]:
        x_i_values = x_values[i]
        x_i_parent_Value = x_values[parent[i]]
        # part 1, P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
        List_count_1[i] = {v2: {v1:X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value} # {pva1: {'1': , '2':, '3': }, pval2:{}}
        List_prob_1[i] = {v2: {v1:(X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] + self.alpha)/ (X_L[(X_L[:,parent[i]] == v2)].shape[0] + self.alpha*K[i]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 2 
        List_count_U[i] = {v2: {v1:X_u[(X_u[:,i] == v1) & (X_u[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 3 
        x_i_0_prob = {v2: {v1: List_count_U[i][v2][v1] - List_prob_1[i][v2][v1]*pri* sum(list(List_count_U[i][v2].values())) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1: max([0,x_i_0_prob[v2][v1] ]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:(x_i_0_prob[v2][v1] + self.alpha)/(self.alpha*K[i] + (1-pri)*sum(list(List_count_U[i][v2].values())) ) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:x_i_0_prob[v2][v1]/sum(list(x_i_0_prob[v2].values()))  for v1 in x_i_values} for v2 in x_i_parent_Value} # normalize 
        List_prob_0[i] = x_i_0_prob
      self.is_fitted_ = True  
      self.parent_ = parent
      self.n_features_, self.K_, self.List_count_1_,self.List_prob_1_, self.List_count_U_, self.List_prob_0_, self.prevalence_ = p, K, List_count_1,List_prob_1,List_count_U,List_prob_0, pri
      return self
      
    def predict_proba(self,X):	
      check_is_fitted(self)
      X = check_array(X)
      Prob_1 = []
      root_i = self.starting_node
      for ins in X:
        P1 = self.prevalence_
        P0 = 1 - P1
        # root_i
        P1 = P1 * (self.List_prob_1_[root_i][ins[root_i]])
        P0 = P0 * (self.List_prob_0_[root_i][ins[root_i]])
        for i in [e for e in range(0,self.n_features_) if e != root_i]:
          pValue = ins[self.parent_[i]]
          P1 = P1 * (self.List_prob_1_[i][pValue][ins[i]])
          P0 = P0 * (self.List_prob_0_[i][pValue][ins[i]])
        P = P1 + P0
        P1 = P1/P; P0 = P0/P
        Prob_1.append(P1)
      #
      Prob_1 = np.array(Prob_1)
      return Prob_1





class PSTAN(Bayes_net_PU):
    name = "PSTAN"
    def __init__(self, alpha = 1,starting_node = 0):
      self.starting_node = starting_node
      self.alpha = alpha

    def Findparent(self, M):
      M = M.copy() # to avoid change global M
      np.fill_diagonal(M,0)  
      p = int(M.shape[0]) 
      V = range(p) # set of all nodes
      st = self.starting_node
      Vnew = [st]  # vertex that already found their parent. intitiate it with starting node. TAN randomly choose one
      parent = {st:None} # use a dict to show nodes' interdepedency
      while set(Vnew) != set(V): # when their are still nodes whose parents are unknown.
        index_i = [] # after for loop, has same length as Vnew, shows the closest node that not in Vnew with Vnew.  
        max_inf = [] # corresponding distance
        for i in range(len(Vnew)):  # can be paralelled 
          vnew = Vnew[i]
          ListToSorted = [e for e in M[:,vnew]] # does not need int(e)
          index = sorted(range(len(ListToSorted)),key = lambda k: ListToSorted[k],reverse = True)
          index_i.append([ele for ele in index if ele not in Vnew][0]) 
          max_inf.append(M[index_i[-1],vnew])
      
        index1 = sorted(range(len(max_inf)),key = lambda k: max_inf[k],reverse = True)[0] ## relative position, Vnew[v1,v2] index_i[v4,v5] max_inf[s1,s2] index1 is the position in those 3 list
        Vnew.append(index_i[index1]) # add in that node
        parent[index_i[index1]] = Vnew[index1] # add direction, it has to be that the new added node is child, otherwise some nodes has 2 parents which is wrong.
      
      return parent

    
    def fit(self,X_L, X_u, pri, M):  # this is based on trainning data !!!
      X_L = check_array(X_L)
      X_u = check_array(X_u)
      if X_L.shape[1] != X_u.shape[1]:
        raise ValueError('labeled data and unlabeled data have different number of features ')
      n_L,p = X_L.shape
      n_u,p = X_u.shape
      parent = self.Findparent(M)
      # part 1: proba that can be estimated from labeled examples. 1 P(xij|1,xkl), 2 p(x_root|1) = N_L(x_root)/N_L,  P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
      # part 2: learn from U, N_U(xij,xkl), and N_U(xkl)
      # part 3: p(xij|0,xkl),p(x_root|0) from previous list
      #
      List_prob_1 = {} # 1 P(xij|1,xkl), 2 p(x_root|1)  
      List_count_1 = {} # N_L(xij,xpal) and N_L(xij)
      # 
      List_count_U = {} # N_U(xij,xkl) and N_U(xij)
      # 
      List_prob_0 = {} # p(xij|0,xkl),p(x_root|0)
      K = {}
      # for root node
      root_i = self.starting_node
      x_i_L = X_L[:,root_i]
      x_i_L_counter = Counter(x_i_L)
      x_i_u = X_u[:,root_i]
      x_i_u_counter = Counter(x_i_u)
      x_i_values = list(set(x_i_L_counter.keys()).union(x_i_u_counter.keys()))
      K[root_i] = len(list(x_i_values))
      # part 1 
      x_i_L_prob = {key: (x_i_L_counter[key]+self.alpha)/(K[root_i]*self.alpha + n_L ) for key in x_i_values}
      List_prob_1[root_i] = x_i_L_prob
      List_count_1[root_i] = x_i_L_counter
      # part 2 
      List_count_U[root_i] = x_i_u_counter
      # part 3 
      x_i_0_prob = {key: max([0,x_i_u_counter[key] - x_i_L_prob[key] * pri * n_u]) for key in x_i_values} # N_U(xi =j) - N_u*p(xij, y =1) = N_U(xij,y=0)   numeritor, can be negative, make it >=0
      x_i_0_prob = {key:(self.alpha + value)/ (K[root_i]*self.alpha + n_u * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
      x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
      List_prob_0[root_i] = x_i_0_prob
      #
      for i in [e for e in range(0,p) if e != root_i]:
        x_i_values = list(set(X_L[:,i]).union(X_u[:,i]))
        x_i_parent_Value = list(set(X_L[:,parent[i]]).union(X_u[:,parent[i] ] ) )
        K[i] = len(x_i_values)
        # part 1, P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
        List_count_1[i] = {v2: {v1:X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value} # {pva1: {'1': , '2':, '3': }, pval2:{}}
        List_prob_1[i] = {v2: {v1:(X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] + self.alpha)/ (X_L[(X_L[:,parent[i]] == v2)].shape[0] + self.alpha*K[i]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 2 
        List_count_U[i] = {v2: {v1:X_u[(X_u[:,i] == v1) & (X_u[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 3 
        x_i_0_prob = {v2: {v1: List_count_U[i][v2][v1] - List_prob_1[i][v2][v1]*pri* sum(list(List_count_U[i][v2].values())) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1: max([0,x_i_0_prob[v2][v1] ]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:(x_i_0_prob[v2][v1] + self.alpha)/(self.alpha*K[i] + (1-pri)*sum(list(List_count_U[i][v2].values())) ) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:x_i_0_prob[v2][v1]/sum(list(x_i_0_prob[v2].values()))  for v1 in x_i_values} for v2 in x_i_parent_Value} # normalize 
        List_prob_0[i] = x_i_0_prob
      self.is_fitted_ = True  
      self.parent_ = parent
      self.n_features_, self.K_, self.List_count_1_,self.List_prob_1_, self.List_count_U_, self.List_prob_0_, self.prevalence_ = p, K, List_count_1,List_prob_1,List_count_U,List_prob_0, pri
      return self
      
    def predict_proba(self,X):	
      check_is_fitted(self)
      X = check_array(X)
      Prob_1 = []
      root_i = self.starting_node
      for ins in X:
        P1 = self.prevalence_
        P0 = 1 - P1
        # root_i
        P1 = P1 * (self.List_prob_1_[root_i][ins[root_i]])
        P0 = P0 * (self.List_prob_0_[root_i][ins[root_i]])
        for i in [e for e in range(0,self.n_features_) if e != root_i]:
          pValue = ins[self.parent_[i]]
          P1 = P1 * (self.List_prob_1_[i][pValue][ins[i]])
          P0 = P0 * (self.List_prob_0_[i][pValue][ins[i]])
        P = P1 + P0
        P1 = P1/P; P0 = P0/P
        Prob_1.append(P1)
      #
      Prob_1 = np.array(Prob_1)
      return Prob_1


class PESTAN(Bayes_net_PU):
  name = "PESTAN"
  def __init__(self,alpha = 1):
    self.alpha = alpha

  def fit(self,X_L, X_u, pri, M): 
    X_L = check_array(X_L)
    X_u = check_array(X_u)
    if X_L.shape[1] != X_u.shape[1]:
      raise ValueError('labeled data and unlabeled data have different number of features ')
    n_L,p = X_L.shape
    n_u,p = X_u.shape
    models = []
    ## train p PSTAN base models
    for i in range(p):
      model = PSTAN(self.alpha, starting_node= i)
      model.fit(X_L, X_u, pri, M)
      models.append(model)
  
    self.models_, self.n_features_ = models, p
    self.is_fitted_ = True
    return self

  def predict_proba(self,X):	   
    check_is_fitted(self)
    X = check_array(X)

    Prob_1 = 0
    for model in self.models_:
      Prob_1 += model.predict_proba(X) # np array here 

    Prob_1 = Prob_1/(self.n_features_)
    return(Prob_1)

  
  

class PETAN(Bayes_net_PU):
  name = "PETAN"
  def __init__(self,alpha = 1):
    self.alpha = alpha

  def fit(self,X_L, X_u, pri, M): 
    X_L = check_array(X_L)
    X_u = check_array(X_u)
    if X_L.shape[1] != X_u.shape[1]:
      raise ValueError('labeled data and unlabeled data have different number of features ')
    n_L,p = X_L.shape
    n_u,p = X_u.shape
    models = []
    ## train p PTAN base models
    for i in range(p):
      model = PTAN(self.alpha, starting_node= i)
      model.fit(X_L, X_u, pri)
      models.append(model)
    
    #append STAN
    model = PSTAN(self.alpha, starting_node = 0) #
    model.fit(X_L, X_u, pri, M)
    models.append(model)    
    self.models_, self.n_features_ = models, p
    self.is_fitted_ = True
    return self

  def predict_proba(self,X):	   
    check_is_fitted(self)
    X = check_array(X)

    Prob_1 = 0
    for model in self.models_:
      Prob_1 += model.predict_proba(X) # np array here 

    Prob_1 = Prob_1/(self.n_features_+ 1)
    return(Prob_1)

  
