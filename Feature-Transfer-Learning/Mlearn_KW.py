import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import pairwise_distances


class MatchOLearn_KW:

  def __init__(self, kernel = 'rbf', C=1.0, gamma=1.0, propensity=0.5, metric='l2', tolerance=1e-2):
    self.kernel = kernel
    self.C = C
    self.gamma = gamma
    self.propensity = propensity
    self.metric = metric
    self.tol = tolerance


  def fit(self, X, Q, A, match,learn, bandC, size=99):

      regr = linear_model.LinearRegression()
      regr.fit(X[:, learn], Q)
      res = Q - regr.predict(X[:, learn])

      XMatch = X[:,match]
      Xdist = pairwise_distances(XMatch,metric=self.metric)

      Qdist = pairwise_distances(Q.reshape(-1, 1), metric="l1")

      sign_dist = lambda p1, p2: np.sign(p1 - p2)
      sign_mat = np.asarray([[sign_dist(p1, p2) for p2 in res] for p1 in res])

      # only match opposite TRT
      trt_match_mat = abs(pairwise_distances(A.reshape(-1, 1), metric="l1")) / 2

      bandwidth = (len(XMatch) ** (-0.2)) * bandC
      gd = np.exp(- (Xdist ** 2) * bandwidth)
      gd = np.where((gd >= np.percentile(gd, size)) & (gd != 1), gd, 0)
      sampleweight =  Qdist * trt_match_mat * gd

      trt_mat = np.repeat(A.reshape(-1, 1), len(A), axis=1)
      newtrt_mat = trt_mat * trt_match_mat * sign_mat

      ind = np.where(sampleweight > 0)
      new_label = newtrt_mat[ind]

      ind1 = np.where(sampleweight > 0)[0]
      newX = X[ind1]
      newweight = sampleweight[ind]
      self.new_label = new_label
      if np.mean(self.new_label) == 1. or np.mean(self.new_label) == -1.: return self
      self.Weights = newweight
      self.clf = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel,tol=self.tol)
      self.clf.fit(newX[:,learn],self.new_label,sample_weight=self.Weights)

      return self

  def predict(self,X):
    tmp = np.mean(self.new_label)
    if tmp == 1. or tmp == -1.:
        classification = np.ones(X.shape[0]) * tmp
        return classification
    classification = self.clf.predict(X)

    return classification

  def estimate(self, X, Q, A, learn, normalize=True):
      tmp = np.mean(self.new_label)
      if tmp == 1. or tmp == -1.:
          classification = np.ones(len(Q)) * tmp
      else:
        classification = self.clf.predict(X[:,learn])
      if self.propensity == 'obs':
          logist = linear_model.LogisticRegression()
          logist.fit(X, A)
          prob = logist.predict_proba(X)[:, 1]
      else:
          prob = self.propensity
      PS = prob * A + (1 - A) / 2
      Q0 = Q[np.where(A == classification)]
      PS0 = PS[np.where(A == classification)]

      if normalize == False:
        est_Q = np.sum(Q0 / PS0) / len(Q)
      elif normalize == True:
          est_Q = np.sum(Q0 / PS0) / np.sum(1 / PS0)

      return est_Q

