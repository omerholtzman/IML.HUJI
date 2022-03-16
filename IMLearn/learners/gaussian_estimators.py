from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> None:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None
        return

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.nanmean(X, axis=0)
        self.var_ = np.nanstd(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        def gaussian_pdf_function(x):
            return 1 / (np.sqrt(2 * np.pi * self.var_)) * \
                    np.exp(- 1 * (x - self.mu_) ** 2 / (2 * self.var_))

        pdf_array = np.zeros(len(X))
        for index, x in enumerate(X):
            pdf_array[index] = gaussian_pdf_function(x)
        return pdf_array

    def get_mean(self):
        return self.mu_

    def get_var(self):
        return self.var_

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        each_variable_computation = lambda x: (x - mu) ** 2
        return_factor = - 1 / (2 * sigma ** 2)
        X = list(map(each_variable_computation, X))
        multiple_factor = np.log(1 / (np.sqrt(2 * np.pi * sigma) ** len(X)))
        return multiple_factor + return_factor * sum(X)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.cov_ = np.cov(X, rowvar=False)
        self.mu_ = np.mean(X, axis=0)

        # def calculate_mean(X: np.ndarray, index):
        #     sum = 0
        #     for i in range(len(X)):
        #         sum += X[i][index]
        #     return sum / len(X)
        #
        # for i in range(len(X[0])):
        #     self.mu_[i] = calculate_mean(X, i)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        def gaussian_pdf_function(x: np.ndarray):
            return 1 / (np.sqrt((2 * np.pi) ** len(X) * det(self.cov_))) * \
                np.exp(-0.5 * np.transpose(X - self.mu_).dot(inv(self.cov_)).dot(X - self.mu_))

        pdf_array = np.zeros(len(X))
        for index, x in enumerate(X):
            pdf_array[index] = gaussian_pdf_function(x)
        return pdf_array

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        each_variable_computation = lambda x: (np.transpose(x - mu)).dot(inv(cov)).dot(x - mu)
        dim_factor = len(X) * np.log(1 / (np.sqrt((2 * np.pi) ** len(X[0])) * det(cov)))

        # Activate the vfunc on all matrix rows.
        vfunc = np.vectorize(each_variable_computation, signature='(n)->()')
        p = vfunc(X)
        # ATTENTION: debugging here - why is p 2d array??

        return dim_factor - 0.5 * sum(p)

    def get_mu(self):
        return self.mu_

    def get_cov(self):
        return self.cov_