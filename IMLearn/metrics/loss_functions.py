import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    __check_labels_sizes(y_pred, y_true)
    sum_squared_error = np.sum(np.power(y_true - y_pred, 2))
    return sum_squared_error / len(y_true)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    __check_labels_sizes(y_pred, y_true)
    misses = np.sum([True if y_true[index] != y_pred[index] else False for index in range(len(y_true))])
    return misses / len(y_true) if normalize else misses


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    __check_labels_sizes(y_pred, y_true)
    misses = np.sum([True if y_true[index] == y_pred[index] else False for index in range(len(y_true))])
    return misses / len(y_true)


def __check_labels_sizes(y_pred, y_true):
    if len(y_true) != len(y_pred):
        raise Exception(f"predicted labels and true label are different in their sizes."
                        f" func: {misclassification_error.__name__}")

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
