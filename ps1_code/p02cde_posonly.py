import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, y_train = util.load_dataset(train_path, add_intercept=True, label_col='t')

    lr = LogisticRegression(verbose=False)
    lr.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(test_path, add_intercept=True, label_col='t')
    y_pred = np.empty_like(y_eval)

    for i in range(len(x_eval)):
        y_pred[i] = lr.predict(x_eval[i])

    # np.savetxt(pred_path_c, np.column_stack((x_eval, y_pred)), delimiter=',')
    np.savetxt(pred_path_c, y_pred, delimiter=',')
    util.plot(x_eval, y_eval, lr.theta, 'output/p02c')

    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, add_intercept=True, label_col='y')

    lr2 = LogisticRegression(verbose=False)
    lr2.fit(x_train, y_train)

    x_eval2, y_eval2 = util.load_dataset(test_path, add_intercept=True, label_col='t')
    y_pred2 = np.empty_like(y_eval2)

    for i in range(len(x_eval2)):
        y_pred2[i] = lr2.predict(x_eval2[i])

    # np.savetxt(pred_path_d, np.column_stack((x_eval2, y_pred2)), delimiter=',')
    np.savetxt(pred_path_d, y_pred2, delimiter=',')
    util.plot(x_eval2, y_eval2, lr2.theta, 'output/p02d')

    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True, label_col='t')
    y_pred = np.empty_like(y_valid)
    alpha_n = 0
    alpha_d = 0
    for i in range(len(x_valid)):
        if y_valid[i] == 1:
            alpha_d += 1
            alpha_n += lr2.predict(x_valid[i])

    alpha = alpha_n / alpha_d

    for i in range(len(x_eval)):
        y_pred[i] = lr.predict(x_eval[i]) / alpha

    # np.savetxt(pred_path_e, np.column_stack((x_eval2, y_pred2 / alpha)), delimiter=',')
    np.savetxt(pred_path_e, y_pred2 / alpha, delimiter=',')
    util.plot(x_eval2, y_eval2, lr2.theta, 'output/p02e', correction=alpha)
    # util.plot(x_eval2, y_eval2, lr2.theta, 'output/p02e_2', correction=1)

    # Plot and use np.savetxt to save outputs to pred_path
    # *** END CODER HERE

if __name__ == "__main__":
    main(train_path='../data/ds3_train.csv',
            valid_path='../data/ds3_valid.csv',
            test_path='../data/ds3_test.csv',
            pred_path='output/p02X_pred.txt')
