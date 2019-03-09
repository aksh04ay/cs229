import pickle


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run solver to fit linear model.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: Linear model parameters, including intercept.
        """
        raise NotImplementedError('Subclass of LinearModel must implement fit method.')

    def predict(self, x):
        """Make a prediction at a new point x given linear model
        parameters theta. Input will not have an intercept term
        (i.e. not necessarily x[0] = 1), but theta expects an intercept term.

        Args:
            x: New data point, NumPy array of shape (1, n).

        Returns:
            Predicted probability for input x.
        """
        raise NotImplementedError('Subclass of LinearModel must implement predict method.')

    def save(self, save_path):
        """Save model to disk.

        Args:
            save_path: Path for saving the model.
        """
        with open(save_path, 'wb') as pkl_fh:
            pickle.dump(self, pkl_fh)

    @staticmethod
    def load(model_path):
        """Load a LinearModel from a given path.

        Args:
            model_path: Path for loading the model.

        Returns:
            LinearModel object.
        """
        with open(model_path, 'rb') as pkl_fh:
            obj = pickle.load(pkl_fh)

        if not isinstance(obj, LinearModel):
            raise ValueError('Not a LinearModel instance: {}'.format(model_path))

        return obj

