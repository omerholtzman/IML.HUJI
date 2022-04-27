from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.fitted_ = False
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.data_by_classes = {}
        for index, sample in enumerate(X):
            if str(y[index][0]) in self.data_by_classes.keys():
                self.data_by_classes[str(y[index][0])].append(sample)
            else:
                self.data_by_classes[str(y[index][0])] = [sample]

        self.classes_, self.mu_, self.vars_, self.pi_, self.vars_inv = [], [], [], [], []

        for key in self.data_by_classes.keys():
            n_k = len(self.data_by_classes[key])
            self.classes_.append(key)
            self.mu_.append(1 / (n_k - 1) * np.sum(self.data_by_classes[key]))
            self.pi_.append(len(self.data_by_classes[key]) / (len(y) - 1))
            self.vars_.append(np.diag(np.var(self.data_by_classes[key], ddof=1, axis=0)))
            self.vars_inv.append(np.linalg.inv(self.vars_[-1]))

        self.fitted_ = True

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

        likehood_matrix = np.zeros((len(X), len(self.classes_)))

        for sample_row, x in enumerate(X):
            for class_col in range(len(self.classes_)):
                likehood_matrix[sample_row, class_col] = self.__compute_likehood_per_sample(x, class_col)
        return likehood_matrix

    def __compute_likehood_per_sample(self, sample, class_col):
        return np.log2(self.pi_[class_col]) - 0.5 * np.log2(np.linalg.det(self.vars_[class_col])) - 0.5 * \
            np.log2(np.dot(np.dot(np.transpose(sample - self.mu_[class_col]), self.vars_inv[class_col]), (sample - self.mu_[class_col])))

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
