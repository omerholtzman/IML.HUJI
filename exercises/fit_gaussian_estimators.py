from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


PART_2_LINSPACE = np.linspace(-10, 10, 200)


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    model, samples = EX1()

    # Question 2 - Empirically showing sample mean is consistent
    EX2(samples)

    # Question 3 - Plotting Empirical PDF of fitted model
    EX3(model, samples)

def EX3(model, samples):
    samples = np.sort(samples)
    pdf_array = model.pdf(samples)

    plt.scatter(samples, pdf_array, label="value by pdf")
    plt.scatter(samples, np.zeros(len(samples)), alpha=0.1, label="heat by amount")
    plt.xlabel("Value of the sample")
    plt.ylabel("The pdf of the sample")
    plt.title("The pdf of sample in consideration to its value (samples value-sorted)")
    plt.legend()
    plt.show()


def EX2(samples):
    mean = np.zeros(100)
    my_gaussian = UnivariateGaussian()
    for i in range(1, 101, 1):
        data = samples[:(i*10)]
        my_gaussian.fit(data)
        mean[i - 1] = my_gaussian.get_mean()
    mean -= 10
    mean = list(mean)
    mean = [np.abs(mean[i]) for i in range(len(mean))]
    samples_size = [i * 10 for i in range(1, 101, 1)]
    plt.plot(samples_size, mean)
    plt.title("Distance between estimated mean and mean, function of sample size")
    plt.xlabel("Sample Size")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()


def EX1():
    data = np.random.normal(loc=10, scale=1, size=1000)
    my_gaussian = UnivariateGaussian()
    my_gaussian.fit(data)
    print("(" + str(my_gaussian.get_mean()) + ", " + str(my_gaussian.get_var()) + ")")
    plt.hist(data, bins=20)
    plt.title("Hist graph of normal gaussian data")
    plt.xlabel("Value")
    plt.ylabel("Amount of samples")
    plt.legend()
    plt.show()

    return my_gaussian, data


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = EX4()

    # Question 5 - Likelihood evaluation
    loglikehood = EX5(samples)

    # Question 6 - Maximum likelihood
    EX6(loglikehood)

def EX6(loglikehood):
    max_log_likehood = max([max(loglikehood[i]) for i in range(len(loglikehood))])

    for i in range(len(loglikehood)):
        for j in range(len(loglikehood[0])):
            if loglikehood[i][j] == max_log_likehood:
                f1, f3 = PART_2_LINSPACE[i], PART_2_LINSPACE[j]
    print("Max loglikehood chance is: " + str(max_log_likehood) + ", for the vector: " + str([f1, 0, f3, 0]))


def EX5(samples):
    cov_matrix = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])

    f1_values = PART_2_LINSPACE
    f3_values = PART_2_LINSPACE

    log_likehood = []

    for f1_value in f1_values:
        print(f1_value)
        log_likehood_row = []
        for f3_value in f3_values:
            tried_mu = np.array([f1_value, 0, f3_value, 0])
            log_likehood_value = MultivariateGaussian.log_likelihood(tried_mu, cov_matrix, samples)
            log_likehood_row.append(log_likehood_value)
        log_likehood.append(log_likehood_row)

    fig, ax = plt.subplots()
    im = ax.imshow(np.array(log_likehood))

    # ax.set_xticks(np.arange(len(PART_2_LINSPACE)), labels=PART_2_LINSPACE)
    # ax.set_yticks(np.arange(len(PART_2_LINSPACE)), labels=PART_2_LINSPACE)

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             # rotation_mode="anchor")
    plt.title("Heatmap of likelihood of f3 and f1 features")
    plt.xlabel("f3 parameter")
    plt.ylabel("f1 parameter")

    # plt.grid(True)
    plt.show()

    return log_likehood



def EX4():
    mu_matrix = np.array([0, 0, 4, 0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean=mu_matrix, cov=cov_matrix, size=1000)
    my_multivariate_gaussian = MultivariateGaussian()
    my_multivariate_gaussian.fit(samples)
    print(my_multivariate_gaussian.get_mu())
    print(my_multivariate_gaussian.get_cov())
    return samples


def quiz_Q_3():
    samples = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    x = UnivariateGaussian.log_likelihood(1, 1, samples)
    y = UnivariateGaussian.log_likelihood(10, 1, samples)

    print("mu = 1, sigma = 1, loglikehood = " + str(x))
    print("mu = 10, sigma = 1, loglikehood = " + str(y))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
    # Max loglikehood chance is: -5985.002059790862, for the vector: [-0.05025125628140792, 0, 3.9698492462311563, 0]
