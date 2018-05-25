"""
Qi et al.
An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization.

Paper: http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-guojun.pdf
"""

from random import choice
import numpy as np
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh
from scipy.sparse.csgraph import laplacian
from base_metric import BaseMetricLearner


class SDML(BaseMetricLearner):
  def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True):
    
  def _prepare_inputs(self, X, W):
    self.X = X
    # set up prior M
    if self.use_cov:
      self.M = np.cov(X.T)
    else:
      self.M = np.identity(X.shape[1])
    L = laplacian(W, normed=False)
    self.loss_matrix = self.X.T.dot(L.dot(self.X))

  def metric(self):
    return self.M

  def fit(self, X, W, verbose=False):
    """
    X: data matrix, (n x d)
    W: connectivity graph, (n x n). +1 for positive pairs, -1 for negative.
    """
    self._prepare_inputs(X, W)
    P = pinvh(self.M) + self.balance_param * self.loss_matrix
    emp_cov = pinvh(P)
    # hack: ensure positive semidefinite
    emp_cov = emp_cov.T.dot(emp_cov)
    self.M, _ = graph_lasso(emp_cov, self.sparsity_param, verbose=verbose)
    return self

  @classmethod
  def prepare_constraints(self, labels, num_points, num_constraints):
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    b, d = np.empty((2, num_constraints), dtype=int)
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      b[i] = choice(np.nonzero(labels == al)[0])
      d[i] = choice(np.nonzero(labels != cl)[0])
    W = np.zeros((num_points,num_points))
    W[a,b] = 1
    W[c,d] = -1
    # make W symmetric
    W[b,a] = 1
    W[d,c] = -1
    return W
