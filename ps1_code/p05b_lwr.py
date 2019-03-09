import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)


    # *** START CODE HERE ***
    # Fit a LWR model
    lwlr = LocallyWeightedLinearRegression(tau)
    lwlr.fit(x_train, y_train)
    y_pred = lwlr.predict(x_eval)

    # Get MSE value on the validation set
    print("MSE (tau = 0.5) = ", sum(np.square(y_eval - y_pred)) / y_eval.size)
    util.plot5(x_train, y_train, x_eval, y_eval, y_pred,'output/p05b')


    # Plot validation predictions on top of training set
    # Plot data
    # No need to use np.savetxt in this problem
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set."""
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def getw(self, x, X, tau):
        W = 0.5 * np.exp(np.true_divide(-1 * np.square(X - x), 2 * tau * tau))
        return np.diag(W[:, 1])

    def gettheta(self, x_test):
        m, n = self.x.shape
        x_test = np.array([x_test] * m)

        W = self.getw(x_test, self.x, self.tau)
        # theta = (X^TWX)^{-1}X^TWY
        xtwx = np.matmul(np.matmul(self.x.T, W), self.x)
        xtwy = np.matmul(np.matmul(self.x.T, W), self.y)
        theta = np.matmul(np.linalg.inv(xtwx), xtwy)
        return theta

    def predict(self, x):
        # *** START CODE HERE ***
        y = []
        for xi in x:
            theta = self.gettheta(xi)
            y.append(np.dot(theta, xi))
        return y
        # *** END CODE HERE ***
