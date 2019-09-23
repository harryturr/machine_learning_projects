import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats

# class to get the data, clean it, format it
class process_data:
    def __init__(self, url, delim):
        self.url = url
        self.delim = delim

    def get_data(self):
        data = pd.read_csv(self.url, sep=self.delim)
        self.data = data
        return data

    def shuffle_data(self):
        data_shuffle = self.data.sample(frac=1)
        self.data_shuffle = data_shuffle
        return data_shuffle

    def clean_data(self):
        clean_data = self.data[(np.abs(stats.zscore(self.data)) < 3).all(axis=1)]
        self.data = clean_data
        return clean_data

    def column_drop(self, column_name):
        X = self.data.drop([column_name], axis=1)
        self.X = X
        self.column_name = column_name
        return X

    def add_intercept(self):
        X = self.X.to_numpy()
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        self.X_int = X
        return X

    def get_Y(self):
        Y = np.asarray(self.data[self.column_name].copy())
        for i in range(0, len(Y)):
            if Y[i] <= 5:
                Y[i] = 0
            else:
                Y[i] = 1
        self.Y = Y
        return Y

    def get_corr(self):
        corr = self.X.corr()
        corr.style.background_gradient(cmap="coolwarm").set_precision(2)
        return corr


def data_worker(x):
    x.get_data()
    x.clean_data()
    x.column_drop("quality")
    x.get_Y()
    x.add_intercept()
    x.get_corr()
    print("data has been gathered and processed \n")


# logistic regression
class LogisticRegression:
    def __init__(self, X, Y, max_iterations=5000000, learning_rate=0.02, stop=0.002):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.stop = stop
        self.X = X
        self.Y = Y
        self.data = np.column_stack((self.X, self.Y))
        self.n = X.shape[0]
        self.p = X.shape[1]

    def invlogit(self, X, w):
        output = 1 / (1 + np.exp(-np.dot(X, w)))
        return output

    def loss_function(self, w):
        output = (
            -np.sum(
                self.Y * np.log(self.invlogit(self.X, w))
                + (1 - self.Y) * np.log(1 - self.invlogit(self.X, w))
            )
            / self.n
        )
        return output

    def fit(self):
        w = np.zeros(self.p)
        loss = np.zeros(self.max_iterations)

        for i in range(1, self.max_iterations):
            delta = np.zeros(self.p)
            #    for j in range(0,n-1):
            #        sum = sum + (Y[j]-invlogit(X[j,:],w))*X[j,:]/n
            log_odds = self.invlogit(self.X, w)
            delta = np.dot(self.X.T, (self.Y - log_odds)) / self.n
            w = w + delta * self.learning_rate
            if i % 100000 == 0:
                loss[i] = self.loss_function(w)
                step = -(loss[i] - loss[i - 100000])
                print(
                    i,
                    "iterations completed and the current cross-entropy loss is at ",
                    "%.7f" % loss[i],
                    "%.3f" % step,
                    "improvement",
                )
                if step <= self.stop and step > 0:
                    print("Converged in", i, "iterations")
                    return w
            if i == self.max_iterations - 1:
                print("Gradient descent failed to converged in", i, "iterations")
                return w

    def predict(self, X):
        w = self.fit()
        predicted = self.invlogit(X, w)
        for i in range(0, len(predicted)):
            if predicted[i] >= 0.5:
                predicted[i] = 1
            else:
                predicted[i] = 0
        return predicted


def evaluate(Y, Y_hat):
    correct = np.where((Y == Y_hat) == True)
    num_correct = np.array(correct).shape[1]
    n = len(Y)
    acc = num_correct / n * 100
    print(
        "The number of correct predictions is ",
        num_correct,
        "overall accuracy is",
        acc,
        "%",
    )
    return acc


# Cross Validation (for logistic regression)


def split(data, k):
    n = data.shape[0]
    p = data.shape[1]
    output = np.zeros((k - 1, math.floor(n / k), p))
    for i in range(0, k - 1):
        output[i] = data[i * math.floor(n / k) : (i + 1) * math.floor(n / k), :]
    output = output.tolist()
    rest = (data[-(n - (k - 1) * math.floor(n / k)) :, :]).tolist()
    output.append(rest)
    return output


def sp_trainvalidate(data, k):
    data_split = split(data, k)
    Train = [None] * k
    Validate = [None] * k
    for i in range(0, k):
        Validate[i] = np.asarray(data_split[i])
        Train[i] = data_split.copy()
        Train[i].pop(i)
        for j in range(0, k - 1):
            Train[i][j] = np.asanyarray(Train[i][j])
        Train[i] = np.vstack(Train[i][0 : k - 1])
    return (Train, Validate)


def cross_val_LR(X, Y, k):
    print("running %s - fold cross validation for Logistic Regression" % k)
    data = np.column_stack((X, Y))
    lists = sp_trainvalidate(data, k)
    Train = lists[0]
    Validate = lists[1]
    p = X.shape[1]
    accuracy = np.zeros(k)
    for i in range(0, k):
        model = LogisticRegression(X=Train[i][:, 0:p], Y=Train[i][:, p], stop=0.001)
        Y_hat = model.predict(Validate[i][:, 0:p])[0]
        accuracy[i] = evaluate(Validate[i][:, p], Y_hat)
    avg_acc = np.mean(accuracy)
    print("The average accuracy for the logistic regression model is", avg_acc, "%")
    return accuracy


# LDA
class LDA:  # Assumes X has first column of ones
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_nointc = X[:, 1:].copy()
        self.data = np.column_stack((self.X_nointc, self.Y))
        self.n = self.data.shape[0]
        self.p = self.data.shape[1]

    def fit(self):
        class0 = self.X_nointc[np.where(self.Y == 0), :][0]
        class1 = self.X_nointc[np.where(self.Y == 1), :][0]

        num_class0 = class0.shape[0]
        num_class1 = class1.shape[0]
        prob_0 = num_class0 / self.n
        prob_1 = num_class1 / self.n

        mean_0 = np.mean(class0, axis=0)
        mean_1 = np.mean(class1, axis=0)
        var_cov = np.cov(self.X_nointc.T)

        w_0 = (
            np.log(prob_1 / prob_0)
            - 1 / 2 * np.linalg.multi_dot([mean_1.T, np.linalg.inv(var_cov), mean_1])
            + 1 / 2 * np.linalg.multi_dot([mean_0.T, np.linalg.inv(var_cov), mean_0])
        )
        w_rest = np.matmul(np.linalg.inv(var_cov), (mean_1 - mean_0))
        return (w_0, w_rest)

    def predict(self, X_nointc):
        w_0 = self.fit()[0]
        w_rest = self.fit()[1]
        predicted_LDA = w_0 + np.matmul(X_nointc, w_rest)
        for i in range(0, predicted_LDA.shape[0]):
            if predicted_LDA[i] >= 0:
                predicted_LDA[i] = 1
            else:
                predicted_LDA[i] = 0
        return predicted_LDA


# Cross Validation LDA
def cross_val_LDA(X, Y, k):
    print("running %s - fold cross validation for LDA" % k)
    X_nointc = X[:, 1:].copy()
    data_LDA = np.column_stack((X_nointc, Y))
    n = data_LDA.shape[0]
    p = data_LDA.shape[1]

    v_w0 = np.zeros(5)
    v_wr = np.zeros((5, 11))
    accuracy_LDA = np.zeros(5)

    data_split_LDA = sp_trainvalidate(np.column_stack((X_nointc, Y)), 5)
    Train_LDA = data_split_LDA[0]
    Validate_LDA = data_split_LDA[1]

    for i in range(0, 5):
        model = LDA(X=Train_LDA[i][:, 0:p], Y=Train_LDA[i][:, p - 1])
        (v_w0, v_w1) = model.fit()
        Y_hat_LDA = model.predict(Validate_LDA[i][:, 0 : p - 1])
        accuracy_LDA[i] = evaluate(Validate_LDA[i][:, p - 1], Y_hat_LDA)

    avg_acc_LDA = np.mean(accuracy_LDA)
    print("Overall average validated accuracy for LDA is", avg_acc_LDA, "% \n")
