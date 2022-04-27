from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        data_by_classes = {}
        for index, sample in enumerate(X):
            if y[index] in data_by_classes.keys():
                data_by_classes[index].append(sample)
            else:
                data_by_classes[index] = [sample]

        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = [], [], [], [], []


        # TODO: take that from tirgul 6, (14)

        for key in data_by_classes.keys():
            self.classes_.append(key)
            self.mu_.append(np.mean(data_by_classes[key]))
            self.cov_.append(np.cov(data_by_classes[key]))
            self._cov_inv.append(np.linalg.inv(np.cov(data_by_classes[key])))
            self.pi_.append(len(data_by_classes[key]) / (len(y) - 1))


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred_y = np.zeros(len(X))
        likehood_matrix = self.likelihood(X)
        for i, x in enumerate(X):
            max_likehood_index = np.argmax(likehood_matrix[i])
            pred_y[i] = self.classes_[max_likehood_index]
        return pred_y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # TODO: should I go by this likehood (page 7 tirgul 6) or by (page 8 tirgule 6)?

        likehood_matrix = np.zeros((len(X), len(self.classes_)))   # sample rows, classes cols.

        for sample_row, x in enumerate(X):
            for class_col in range(len(self.classes_)):
                likehood_matrix[sample_row, class_col] = self.__compute_likehood(x, class_col)
        return likehood_matrix

    def __compute_likehood(self, sample, class_col):
        a_k = np.dot(self._cov_inv[class_col], self.mu_[class_col])
        b_k = np.log(self.pi_[class_col]) - 0.5 * np.dot(self.mu_[class_col], a_k)
        return np.dot(np.transpose(a_k), sample) + b_k

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self._predict(X), y)
