from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

import matplotlib.pyplot as plt


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    # print(data_matrix.Functional.to_string(index=False))  # prints a specific column in the dataFrame.
    # print(df.info)
    # for column in df.columns:
    #     if is_numeric_dtype(df[column]):
    #         print(column)
    wanted_features = ['LotArea']
    predicted_label = 'SalePrice'
    return df[wanted_features], pd.Series(df[predicted_label])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = r"C:\Users\t8851692\PycharmProjects\IML.HUJI\exercises\datafigures") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for index, column in enumerate(X.columns):
        if is_numeric_dtype(X[column]):
            x = X[column]
            plt.scatter(x, y)
            plt.savefig(output_path + f'fig{index}.png')
            # fig.write_image(output_path, r"\file.png")

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"C:\Users\t8851692\PycharmProjects\IML.HUJI\datasets\house_train.csv")
    feature_evaluation(X, y)

    # LotArea, Condition1, Condition2, OverallQual, OverallCond,
    exit()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
