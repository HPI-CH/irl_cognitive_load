import numpy as np
from sklearn.svm import NuSVC
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomNuSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.model = NuSVC(nu=self.nu, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking, probability=self.probability, tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, decision_function_shape=self.decision_function_shape, break_ties=self.break_ties, random_state=self.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self, X):
        self.fit(self.X, self.y)
        decision_function = self.model.decision_function(self.X)
        feature_importances = np.abs(np.dot(self.X.T, decision_function))
        self.feature_importances_ = feature_importances
        return feature_importances
    





