class Bayes_net_PU(BaseEstimator, ClassifierMixin): 
    """
    Bayesian network implementation for Postitive Unlabeled examples
    API inspired by SciKit-learn.
    """

    def predict_proba(self, X): # key prediction methods, all other prediction methods will use it first.
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

