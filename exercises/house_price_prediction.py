from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error

from typing import NoReturn
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

import matplotlib.pyplot as plt
from IMLearn.utils.utils import split_train_test


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
    print(df.columns)

    wanted_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view',
                       'condition', 'grade', 'sqft_above', 'sqft_living15', 'sqft_lot15']
    predicted_label = 'price'
    processed_df = df[wanted_features].fillna(0)
    return processed_df, pd.Series(df[predicted_label])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "C:/Users/t8851692/PycharmProjects/IML.HUJI/exercises/datafigures/") -> NoReturn:
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
    plot_features_to_price(X, output_path, y)
    calculate_pearson_correlation(X, y, output_path)

def calculate_pearson_correlation(X, y, output_path):
    def pearson_correlation(x, y):
        current_pearson_correlation = np.cov(x, y, rowvar=True) / (np.std(x) * np.std(y))
        # returns 2*2 matrix
        return current_pearson_correlation[0, 1]

    pearson_correlations_list = [pearson_correlation(X[X.columns[i]], y) for i in range(len(X.columns))]
    print(pearson_correlations_list)

    pearson_correlations_list = pearson_correlations_list
    # Creating histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(pearson_correlations_list, bins=len(X.columns))

    # Set title
    ax.set_title(f"pearson_correlations between features and price")

    # adding labels
    ax.set_xlabel('features')
    ax.set_ylabel('pearson correlations values')

    # Make some labels.
    rects = ax.patches
    labels = [X[X.columns[i]].name[:8] for i in range(len(rects))]
    print(labels)

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label,
                ha='center', va='bottom')

    plt.savefig(output_path + f'pearson_correlations.png')
    plt.show()


def plot_features_to_price(X, output_path, y):
    for index, column in enumerate(X.columns):
        if is_numeric_dtype(X[column]):
            x = X[column]
            plt.scatter(x, y)
            plt.savefig(output_path + f'fig_{X.columns[index]}.png')
            plt.title(f"{X.columns[index]}")
            plt.clf()


def activate_linear_regressor(X_train, Y_train, X_test, Y_test):
    iterations = 10
    mean_list, std_list = [], []
    model = LinearRegression()
    for percent in (range(10, 101)):
        data_length = int(len(X_train) * percent / 100)
        current_performance = []
        for _ in range(iterations):
            data = (pd.concat([X_train, Y_train], axis=1)).sample(data_length)
            train = data.drop(['price'], axis=1)
            test = data['price']
            model.fit(train, test)
            current_performance.append(model.loss(X_test, Y_test))
        mean_list.append(np.mean(current_performance))
        std_list.append(2 * np.std(current_performance))
    return list(range(10, 101)), mean_list, std_list


def plot_linear_regressor_performance(percentages_list, mean_list, std_list):
    plt.plot(percentages_list, mean_list, yerr=std_list)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"C:\Users\t8851692\PycharmProjects\IML.HUJI\datasets\house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)
    exit()
    # Question 3 - Split samples into training- and testing sets.
    X_train, Y_train, X_test, Y_test = split_train_test(X, y)


    # Question 4 - Fit model over increasing percentages of the overall training data
    percentage, mean_list, std_list = activate_linear_regressor(X_train, Y_train, X_test, Y_test)
    plot_linear_regressor_performance(percentage, mean_list, std_list)
