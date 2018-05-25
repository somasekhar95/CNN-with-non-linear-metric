
from random import choice
import numpy as np
import scipy.linalg
from base_metric import BaseMetricLearner


class LSML(BaseMetricLearner):
  def __init__(self, tol=1e-3, max_iter=1000):
    self.tol = tol
    self.max_iter = max_iter

  def _prepare_inputs(self, X, constraints, weights, prior):
    self.X = X
    self.vab = np.diff(X[constraints[:,:2]], axis=1)[:,0]
    self.vcd = np.diff(X[constraints[:,2:]], axis=1)[:,0]
    if weights is None:
      self.w = np.ones(constraints.shape[0])
    else:
      self.w = weights
    self.w /= self.w.sum()  # weights must sum to 1
    if prior is None:
      self.M = np.cov(X.T)
    else:
      self.M = prior

  def metric(self):
    return self.M

  def fit(self, X, constraints, weights=None, prior=None, verbose=False):
    
    self._prepare_inputs(X, constraints, weights, prior)
    prior_inv = scipy.linalg.inv(self.M)
    s_best = self._total_loss(self.M, prior_inv)
    step_sizes = np.logspace(-10, 0, 10)
    if verbose:
      print 'initial loss', s_best
    for it in xrange(1, self.max_iter+1):
      grad = self._gradient(self.M, prior_inv)
      grad_norm = scipy.linalg.norm(grad)
      if grad_norm < self.tol:
        break
      if verbose:
        print 'gradient norm', grad_norm
      M_best = None
      for step_size in step_sizes:
        step_size /= grad_norm
        new_metric = self.M - step_size * grad
        w, v = scipy.linalg.eigh(new_metric)
        new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
        cur_s = self._total_loss(new_metric, prior_inv)
        if cur_s < s_best:
          l_best = step_size
          s_best = cur_s
          M_best = new_metric
      if verbose:
        print 'iter', it, 'cost', s_best, 'best step', l_best * grad_norm
      if M_best is None:
        break
      self.M = M_best
    else:
      print "Didn't converge after %d iterations. Final loss: %f" % (it, s_best)
    return self

  def _comparison_loss(self, metric):
    dab = np.sum(self.vab.dot(metric) * self.vab, axis=1)
    dcd = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
    violations = dab > dcd
    return self.w[violations].dot((np.sqrt(dab[violations]) -
                                   np.sqrt(dcd[violations]))**2)

  def _total_loss(self, metric, prior_inv):
    return (self._comparison_loss(metric) +
            _regularization_loss(metric, prior_inv))

  def _gradient(self, metric, prior_inv):
    dMetric = prior_inv - scipy.linalg.inv(metric)
    dabs = np.sum(self.vab.dot(metric) * self.vab, axis=1)
    dcds = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
    violations = dabs > dcds
    # TODO: vectorize
    for vab, dab, vcd, dcd in zip(self.vab[violations], dabs[violations],
                                  self.vcd[violations], dcds[violations]):
      dMetric += ((1-np.sqrt(dcd/dab))*np.outer(vab, vab) +
                  (1-np.sqrt(dab/dcd))*np.outer(vcd, vcd))
    return dMetric

  @classmethod
  def prepare_constraints(cls, labels, num_constraints):
    C = np.empty((num_constraints,4), dtype=int)
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      C[i,1] = choice(np.nonzero(labels == al)[0])
      C[i,3] = choice(np.nonzero(labels != cl)[0])
    C[:,0] = a
    C[:,2] = c
    return C


def _regularization_loss(metric, prior_inv):
  sign, logdet = np.linalg.slogdet(metric)
  return np.sum(metric * prior_inv) - sign * logdet
