import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = np.empty_like(y_eval)

    for i in range(len(x_eval)):
        y_pred[i] = lr.predict(x_eval[i])

    # np.savetxt(pred_path, np.column_stack((x_eval, y_pred)), delimiter=',')
    np.savetxt(pred_path, y_pred, delimiter=',')

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def sigmoid(self, theta, x):
        return 1 / (1 + np.exp(-1 * np.dot(theta, x)))

    def gradient(self, x, y, theta, m, n):
        grad = np.zeros(n)
        for i in range(m):
            for j in range(n):
                grad[j] += (-1 * ((y[i] - self.sigmoid(theta, x[i])) * x[i][j]) / m)
        return grad

    def hessian(self, x, y, theta, m, n):
        hess = np.zeros((n, n))
        for i in range(m):
            for j in range(n):
                for k in range(n):
                    hess[j][k] += (self.sigmoid(theta, x[i]) * (1 - self.sigmoid(theta, x[i])) * x[i][j] * x[i][k] / m)
        return hess

    def update(self, x, y, theta, m, n):
        return theta - (np.linalg.inv(self.hessian(x, y, theta, m, n)) @ self.gradient(x, y, theta, m, n))

    def loss(self, x, y, theta, m, n):
        loss = 0
        for i in range(m):
            loss += -1 * (y[i] * np.log(self.sigmoid(theta, x[i])) + (1 - y[i]) * (np.log(1 - self.sigmoid(theta, x[i])))) / m
        return loss

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Logistic regression model parameters, including intercept.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        theta = self.theta
        if theta is None:
            theta = np.zeros(n)

        while True:
            loss = self.loss(x, y, theta, m, n)
            theta_new = self.update(x, y, theta, m, n)
            if self.verbose:
                print("Loss: ", loss, " 1-norm: ", np.linalg.norm(theta_new - theta, ord=1))
            if np.linalg.norm(theta_new - theta, ord = 1) < self.eps:
                self.theta = theta_new
                break
            theta = theta_new

        self.theta = theta
        util.plot(x, y, self.theta, 'output/p01b')


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction at a new point x given logistic
        regression parameters theta. Input will not have an intercept term
        (i.e. not necessarily x[0] = 1), but theta expects an intercept term.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        # *** START CODE HERE ***
        return self.sigmoid(self.theta, x)
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(train_path='../data/ds1_train.csv',
             eval_path='../data/ds1_valid.csv',
             pred_path='output/p01b_pred_1.txt')
