import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)


    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    mse = {}
    c = 0
    for tau in tau_values:
        c += 1
        lwlr = LocallyWeightedLinearRegression(tau)
        lwlr.fit(x_train, y_train)
        y_pred = lwlr.predict(x_valid)
        mse[tau] = sum(np.square(y_valid - y_pred)) / y_valid.size
        util.plot5(x_train, y_train, x_valid, y_valid, y_pred,'output/p05c' + str(c), title='tau = ' + str(tau))
    print(mse)

    tau = 0.05
    lwlr = LocallyWeightedLinearRegression(tau)
    lwlr.fit(x_train, y_train)
    y_pred = lwlr.predict(x_test)
    mse = sum(np.square(y_test - y_pred)) / y_test.size
    print(mse)
    util.plot5(x_train, y_train, x_test, y_test, y_pred, 'output/p05c_test', title='test prediction')
    np.savetxt(pred_path, y_pred, delimiter=',')

    # Run on the test set to get the MSE value
    # Plot data
    # Run on the test set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***
