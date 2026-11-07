import numpy as np
from sklearn.svm import NuSVR
from sklearn.feature_selection import RFE
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin

class CustomNuSVR(NuSVR):
    """Customized NuSVR to use the RFE

    Parameters
    ----------
    NuSVR : Sklearn Class
        NuSVR from the sklearn's class.
    """
    def fit(self, X, y):
        super().fit(X, y)
        self.X =X
        self.y = y
       
        dual_coef = self.dual_coef_[0]  # shape: (n_support_vectors,)
        support_vectors = self.support_vectors_  # shape: (n_support_vectors, n_features)
        self.feature_importances_ = np.dot(dual_coef, support_vectors)  # shape: (n_features,)
        return self


    def predict(self, X):
        """Makes the prediction based on the given data

        Parameters
        ----------
        X : pd.DataFrame
           Data

        Returns
        -------
        np.array
            Predictions
        """
        return super().predict(X)