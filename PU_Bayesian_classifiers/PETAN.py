class PETAN(Bayes_net_PU):
  name = "PETAN"
  def __init__(self,alpha = 1):
    self.alpha = alpha

  def fit(self,X_L, X_u, pri, M, case_control = True): 
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
      model.fit(X_L, X_u, pri,case_control)
      models.append(model)
    
    #append STAN
    model = PSTAN(self.alpha, starting_node = 0) #
    model.fit(X_L, X_u, pri, M,case_control)
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

  
