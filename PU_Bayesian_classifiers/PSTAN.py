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

    
    def fit(self,X_L, X_u, pri, M, case_control = True):  # this is based on trainning data !!!
      X_L = check_array(X_L)
      X_u = check_array(X_u)
      if X_L.shape[1] != X_u.shape[1]:
        raise ValueError('labeled data and unlabeled data have different number of features ')
      n_L,p = X_L.shape
      # n_u,p = X_u.shape
      if case_control:
        X_U_or_UL = X_u
      else:
        X_U_or_UL = np.concatenate((X_L,X_u),axis = 0)
      #
      n_U_or_UL = X_U_or_UL.shape[0]
      parent = self.Findparent(M)
      # part 1: proba that can be estimated from labeled examples. 1 P(xij|1,xkl), 2 p(x_root|1) = N_L(x_root)/N_L,  P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
      # part 2: learn from U, N_U(xij,xkl), and N_U(xkl)
      # part 3: p(xij|0,xkl),p(x_root|0) from previous list
      #
      List_prob_1 = {} # 1 P(xij|1,xkl), 2 p(x_root|1)  
      List_count_1 = {} # N_L(xij,xpal) and N_L(xij)
      # 
      List_count_U_or_UL = {} # N_U(xij,xkl) and N_U(xij)
      # 
      List_prob_0 = {} # p(xij|0,xkl),p(x_root|0)
      K = {}
      # for root node
      root_i = self.starting_node
      x_i_L = X_L[:,root_i]
      x_i_L_counter = Counter(x_i_L)
      x_i_U_or_UL = X_U_or_UL[:,root_i]
      x_i_U_or_UL_counter = Counter(x_i_U_or_UL)
      x_i_values = list(set(x_i_L_counter.keys()).union(x_i_U_or_UL_counter.keys()))
      K[root_i] = len(list(x_i_values))
      # part 1 
      x_i_L_prob = {key: (x_i_L_counter[key]+self.alpha)/(K[root_i]*self.alpha + n_L ) for key in x_i_values}
      List_prob_1[root_i] = x_i_L_prob
      List_count_1[root_i] = x_i_L_counter
      # part 2 
      List_count_U_or_UL[root_i] = x_i_U_or_UL_counter
      # part 3 
      x_i_0_prob = {key: max([0,x_i_U_or_UL_counter[key] - x_i_L_prob[key] * pri * n_U_or_UL]) for key in x_i_values} # N_U(xi =j) - N_u*p(xij, y =1) = N_U(xij,y=0)   numeritor, can be negative, make it >=0
      x_i_0_prob = {key:(self.alpha + value)/ (K[root_i]*self.alpha + n_U_or_UL * (1-pri) ) for key,value in x_i_0_prob.items()} # add psudo count and divied by dem
      x_i_0_prob = {key: value/(sum(np.array(list(x_i_0_prob.values()))))   for key,value in x_i_0_prob.items() } # normalize prob sum to 1, however, due to computation problem, it is not sum to 1
      List_prob_0[root_i] = x_i_0_prob
      #
      for i in [e for e in range(0,p) if e != root_i]:
        x_i_values = list(set(X_L[:,i]).union(X_U_or_UL[:,i]))
        x_i_parent_Value = list(set(X_L[:,parent[i]]).union(X_U_or_UL[:,parent[i] ] ) )
        K[i] = len(x_i_values)
        # part 1, P(xij|1,xkl) = N_L(xi=j,xk=l)/N_L(xkl)
        List_count_1[i] = {v2: {v1:X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value} # {pva1: {'1': , '2':, '3': }, pval2:{}}
        List_prob_1[i] = {v2: {v1:(X_L[(X_L[:,i] == v1) & (X_L[:,parent[i]] == v2)].shape[0] + self.alpha)/ (X_L[(X_L[:,parent[i]] == v2)].shape[0] + self.alpha*K[i]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 2 
        List_count_U_or_UL[i] = {v2: {v1:X_U_or_UL[(X_U_or_UL[:,i] == v1) & (X_U_or_UL[:,parent[i]] == v2)].shape[0] for v1 in x_i_values} for v2 in x_i_parent_Value}
        # part 3 
        x_i_0_prob = {v2: {v1: List_count_U_or_UL[i][v2][v1] - List_prob_1[i][v2][v1]*pri* sum(list(List_count_U_or_UL[i][v2].values())) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1: max([0,x_i_0_prob[v2][v1] ]) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:(x_i_0_prob[v2][v1] + self.alpha)/(self.alpha*K[i] + (1-pri)*sum(list(List_count_U_or_UL[i][v2].values())) ) for v1 in x_i_values} for v2 in x_i_parent_Value}
        x_i_0_prob = {v2: {v1:x_i_0_prob[v2][v1]/sum(list(x_i_0_prob[v2].values()))  for v1 in x_i_values} for v2 in x_i_parent_Value} # normalize 
        List_prob_0[i] = x_i_0_prob
      self.case_control_ = case_control
      self.is_fitted_ = True  
      self.parent_ = parent
      self.n_features_, self.K_, self.List_count_1_,self.List_prob_1_, self.List_count_U_, self.List_prob_0_, self.prevalence_ = p, K, List_count_1,List_prob_1,List_count_U_or_UL,List_prob_0, pri
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
