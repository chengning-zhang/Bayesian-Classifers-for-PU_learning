class PNB(Bayes_net_PU):
  name = "PNB"
  def __init__(self, alpha = 1):
      self.alpha = alpha

  def fit(self,X_L, X_u, pri, M = None, case_control = True):  
    """ Implementation of a fitting function.
        Parameters
        ----------
        X_l : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input positive labeled samples.
        X_u : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input unlabeled samples.
        pri : scalar.
            The prevalence probability (p(y = 1))
        M : None
            contact matrix.    
        case_control : Bool
            Case control scenario or single-training data scenario
        Returns
        -------
        self : object
            Returns self.
      """

    X_L = check_array(X_L)
    X_u = check_array(X_u)
    if X_L.shape[1] != X_u.shape[1]:
      raise ValueError('labeled data and unlabeled data have different number of features ')
    # 1: Learned from positive examples, P(xij|1) = N_L(xij)/N_L.  N_L(xij), same for both scenario
    # 2: Learned from Unlabeled examples, N_U(xij) or from U+L N_(U+L)(xij)
    # 3: P(xi = j|c = 0), Listprob0, calculated from previous list
    n_L,p = X_L.shape
    # n_u,p = X_u.shape
    if case_control:
      X_U_or_UL = X_u
    else:
      X_U_or_UL = np.concatenate((X_L,X_u),axis = 0)
    #
    n_U_or_UT = X_U_or_UL.shape[0]
    List_count_1 = {} 
    List_prob_1 = {} # {x0:{'1': p(x0 =1|y=1), '2': p(x0 =2|y=1), 'else': }, x1:{},   ... x7:{} }
    #
    List_count_U_or_UL = {} 
    #
    List_prob_0 = {} # P(xi = j|c=0)
    K = {} # X_i_L and X_i_u contains all possible values of x_i, there are not other values, different from supervised setting.
    for i in range(p):
      x_i_L = X_L[:,i]
      x_i_U_or_UL = X_U_or_UL[:,i]
      x_i_L_counter = Counter(x_i_L) # may be not need key error
      x_i_U_or_UL_counter = Counter(x_i_U_or_UL)
      x_i_values = set(x_i_L_counter.keys()).union(x_i_U_or_UL_counter.keys()) # all possible values of x_i
      K[i] = len(list(x_i_values))
      # part 1
      x_i_L_prob = {key: (value + self.alpha) / (n_L + self.alpha * (K[i]) ) for key,value in x_i_L_counter.items()} # p(x|s=1) = p(x|y=1)
      x_i_L_prob.update({key: (0 + self.alpha) / (n_L + self.alpha * (K[i]) )  for key in list(x_i_values) if key not in list(x_i_L_counter.keys()) } )
      List_prob_1[i] = x_i_L_prob
      List_count_1[i] = x_i_L_counter
      # part 2
      List_count_U_or_UL[i] = x_i_U_or_UL_counter
      # part 3
      x_i_0_prob = {key: max([0,x_i_U_or_UL_counter[key] - x_i_L_prob[key] * pri * n_U_or_UT]) for key in list(x_i_values)} # numeritor, can be negative, make it >=0
      x_i_0_prob = {key:(self.alpha + value)/ (K[i]*self.alpha + n_U_or_UT * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
      x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
      List_prob_0[i] = x_i_0_prob
      # x_i_0_prob = {key: value/sum(np.array(list(x_i_0_prob.values()))) for key,value in x_i_0_prob.items()    }
    self.case_control_ = case_control
    self.is_fitted_ = True  
    self.n_features_, self.K_, self.List_count_1_,self.List_prob_1_, self.List_count_U_or_UL_, self.List_prob_0_, self.prevalence_ = p, K, List_count_1,List_prob_1,List_count_U_or_UL,List_prob_0, pri
    return self

  def predict_proba(self,X): 
    """
        Return probability estimates for the test vector X. Usually it would be X_unlabeled
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        P(y=1|x) : array-like of shape (n_samples, )
            Returns the probability of the samples for positive class in
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

    
