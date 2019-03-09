import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(d): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    lr = GDA()
    lr.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = np.empty_like(y_eval)

    for i in range(len(x_eval)):
        y_pred[i] = lr.predict(x_eval[i])

    # np.savetxt(pred_path, np.column_stack((x_eval, y_pred)), delimiter=',')
    np.savetxt(pred_path, y_pred, delimiter=',')

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    # *** END CODE HERE ***


class GDA(LinearModel):

    def sigmoid(self, theta, x):
        return 1 / (1 + np.exp(-1 * np.dot(theta, x)))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # Find phi, mu_0, mu_1, and sigma
        phi = 0
        mu_0_n = np.zeros(n)
        mu_0_d = 0
        mu_1_n = np.zeros(n)
        mu_1_d = 0
        sigma = np.zeros((n, n))
        for i in range(m):
            if y[i] == 1:
                phi += 1 / m
                mu_1_n += x[i]
                mu_1_d += 1
            else:
                mu_0_n += x[i]
                mu_0_d += 1
        mu_0 = mu_0_n / mu_0_d
        mu_1 = mu_1_n / mu_1_d

        for i in range(m):
            if y[i] == 1:
                sigma += ((x[i] - mu_1) * (x[i] - mu_1)[:,np.newaxis]) / m
            else:
                sigma += ((x[i] - mu_0) * (x[i] - mu_0)[:,np.newaxis]) / m

        theta_0 = np.log((1 - phi) / phi) - 0.5 * (mu_0.T @ np.linalg.inv(sigma) @ mu_0 - mu_1.T @ np.linalg.inv(sigma) @ mu_1)
        theta = np.transpose(mu_0 - mu_1) @ np.linalg.inv(sigma)
        self.theta = np.insert(theta, 0, theta_0)
        util.plot(x, y, self.theta, 'output/p01e_2')
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given linear coefficients theta.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        return self.sigmoid(self.theta, x)
        # *** END CODE HERE

main(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01e_pred_1.txt')
