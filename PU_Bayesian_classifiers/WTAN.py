class WTAN(Bayes_net_PU):
  name = "WTAN"
  def __init__(self,alpha = 1,starting_node = 0):
    self.alpha = alpha
    self.starting_node = starting_node

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
  
  def fit(self,X_L, X_u, pri, M,  case_control = True, model_class = LogisticRegression, **kwargs):
    """ Implementation of a fitting function.
        Get fitted model that predict p(s=1|x), not related to sampling scenario
        Parameters
        ----------
        X_l : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input positive labeled samples.
        X_u : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input unlabeled samples.
        pri : scalar
            The prevalence p(y=1)
        M : np.matrix, shpae (n_features, n_features)
            contact matrix
        case_control : Bool
            Case control scenario or single-training data scenario
        model_class : a sklearn estimator, preferred logistic regression 
                      since it gives calibrated proba, predict p(s=1|x)
        
        **kwargs : 
            extra parameters for model_class

        Returns self
        -------
        self
    """
    X_L = check_array(X_L)
    X_u = check_array(X_u)
    if X_L.shape[1] != X_u.shape[1]:
      raise ValueError('labeled data and unlabeled data have different number of features ')
    n_L,p = X_L.shape
    # parent 
    parent = self.Findparent(M)
    # fit model g(x) = p(s=1|x)
    X = np.concatenate((X_L,X_u), axis = 0)
    enc = preprocessing.OneHotEncoder(drop='first').fit(X)
    X = enc.transform(X).toarray()
    # X = pd.DataFrame(X).astype('category') # convert to categorical, for logistic regression to work
    y = np.concatenate( (np.repeat('1',X_L.shape[0] ), np.repeat('0',X_u.shape[0]) ),axis = 0)
    # 
    model = model_class(**kwargs)
    model.fit(X,y)
    # estimate p(s=1)
    p_s_1 = X_L.shape[0]/(X_L.shape[0]+X_u.shape[0])
    # estimate c
    if case_control:
      c = p_s_1/(pri*(1-p_s_1) + p_s_1)
    else: 
      c = p_s_1/pri
    # estimate w(x)
    inx = list(model.classes_ ).index('1')
    g_U = model.predict_proba( X[n_L:] )[:,inx] # let us assume it is already calibrated ,it that already calibrated? 
    w_U = ((1-c)/c) * (g_U/(1-g_U)) # maybe need to normalize
    w_U = w_U - min(w_U) # make non-negative
    w_U = w_U / max(w_U) # 0-1
    # learning the coef_, p(xij|1,xpal), p(xij|0,xpal)
    # extreme case: w_U correctly weight positive 1 and negative 0 in U, originally p(xij|1) = N_L(xij)/N_L, 
    # List_count_1 = {} 
    List_prob_1 = {} #
    #
    List_prob_0 = {} # P(xi = j|c=0)
    # for root node
    root_i = self.starting_node
    x_i_L_counter = Counter(X_L[:,root_i])
    x_i_values = list(set(X_L[:,root_i]).union(set(X_u[:,root_i])))
    X_i_U_1_counter = {val: w_U[X_u[:,root_i] == val].sum() for val in x_i_values}
    X_i_U_0_counter = {val: (1-w_U)[X_u[:,root_i] == val].sum() for val in x_i_values}
    # part 1, p(xi = j|1) = (N_L(xij) + sum_U_xij(w_U))/( n_L + sum(w_U))
    List_prob_1[root_i] = {key: (self.alpha + x_i_L_counter[key] + X_i_U_1_counter[key]) / (n_L + w_U.sum() + self.alpha*len(x_i_values) ) for key in x_i_values}
    # part 2, p(xi = j|1)
    List_prob_0[root_i] = {key: ( self.alpha + X_i_U_0_counter[key])/ ((1-w_U).sum() + self.alpha*len(x_i_values) ) for key in x_i_values}
    # for other nodes
    for i in [e for e in range(0,p) if e != root_i]:
      x_i_values = list(set(X_L[:,i]).union(X_u[:,i]))
      x_i_parent_Value = list(set(X_L[:,parent[i]]).union(X_u[:,parent[i] ] ) )
      # part 1, p(xij|1,xkl)
      List_prob_1[i] = {v2: {v1: (self.alpha + X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0]  +  w_U[(X_u[:,i] == v1) & (X_u[:,parent[i]] == v2)].sum()   ) / 
                             ( X_L[(X_L[:,parent[i]] == v2)].shape[0] +  w_U[(X_u[:,parent[i]] == v2)].sum()+ self.alpha*len(x_i_values)  ) 
                             for v1 in x_i_values} for v2 in x_i_parent_Value}
      # part 2 , p(xij|0,xkl)
      List_prob_0[i] = {v2: {v1: (self.alpha + (1-w_U)[(X_u[:,i] == v1) & (X_u[:,parent[i]] == v2)].sum()  ) / 
                              ( (1-w_U)[(X_u[:,parent[i]] == v2)].sum() + self.alpha*len(x_i_values)  )
                             for v1 in x_i_values} for v2 in x_i_parent_Value}
    
    
    self.case_control_ = case_control
    self.is_fitted_ = True  
    self.parent_ = parent
    self.case_control_ = case_control
    self.List_prob_1_, self.List_prob_0_, self.c_, self.n_features_, self.w_U_, self.prevalence_ = List_prob_1, List_prob_0, c, p, w_U, pri
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
    root_i = self.starting_node
    for ins in X:
      P1 = self.prevalence_ # don't need copy, immutable
      P0 = 1 - P1
      # root_i
      P1 = P1 * (self.List_prob_1_[root_i][ins[root_i]])
      P0 = P0 * (self.List_prob_0_[root_i][ins[root_i]])
      for i in [e for e in range(0,self.n_features_) if e != root_i]:
        pValue = ins[self.parent_[i]]
        P1 = P1 * (self.List_prob_1_[i][pValue][ins[i]])
        P0 = P0 * (self.List_prob_0_[i][pValue][ins[i]])
        # normalize proba
      P = P1 + P0
      P1 = P1/P; P0 = P0/P
      Prob_1.append(P1)
    #
    Prob_1 = np.array(Prob_1) # for shap 
    return Prob_1
