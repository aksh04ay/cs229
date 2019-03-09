import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    pr = PoissonRegression(max_iter=10000)
    pr.step_size = lr
    pr.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = np.empty_like(y_eval)

    for i in range(len(x_eval)):
        y_pred[i] = pr.predict(x_eval[i])

    # np.savetxt(pred_path, np.column_stack((x_eval, y_pred)), delimiter=',')
    np.savetxt(pred_path, y_pred, delimiter=',')

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def update(self, x, y, theta, m, n):
        return theta + self.step_size * self.gradloglikelihood(x, y, theta, m, n) / m

    def gradloglikelihood(self, x, y, theta, m, n):
        grad = np.zeros(n)
        for i in range(m):
            for j in range(n):
                dotp = np.dot(theta, x[i])
                grad[j] += ((y[i] - np.exp(dotp)) * x[i][j])

        return grad

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Logistic regression model parameters, including intercept.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        print(m,n)
        theta = self.theta
        if theta == None:
            theta = np.zeros(n)

        for i in range(self.max_iter):
            theta_new = self.update(x, y, theta, m, n)
            if np.linalg.norm(theta_new - theta, ord=1) < self.eps:
                self.theta = theta_new
                break
            else:
                if i % 100 == 0:
                    print(i, np.linalg.norm(theta_new - theta, ord=1))
            # if self.verbose:
            #     print('Log likelihood: ', loglikehood)
        # else:
        #     print(i, np.linalg.norm(theta_new - theta, ord=1))
            theta = theta_new

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
        return np.exp(np.dot(self.theta, x))
        # *** END CODE HERE ***

if __name__ == "__main__":
    main(lr=1e-7,
            train_path='../data/ds4_train.csv',
            eval_path='../data/ds4_valid.csv',
            pred_path='output/p03d_pred.txt')