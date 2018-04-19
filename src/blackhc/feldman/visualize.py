import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition
from sklearn.metrics import confusion_matrix

def read_data(repaired=True):
    if not repaired:
        data = pd.read_csv('input/german_processed_age_cat.csv')
    else:
        data = pd.read_csv('input/german_repaired.csv')

    one_hot_data = pd.get_dummies(data)
    return one_hot_data


def get_Xtarget(one_hot_data, category, ignored_features=[]):
    X = one_hot_data[one_hot_data.columns.difference(ignored_features + [category])]
    target = one_hot_data[category]
    return X, target


def project2d_classifier(X, target):
    logistic = linear_model.LogisticRegression()
    logistic.fit(X, target)

    decision_coeff = np.matmul(X, logistic.coef_.T)

    # plt.plot(p1, np.zeros_like(p1), 'x')

    # Remove the part of the data that is important for the decision.
    remainder_X = X - logistic.coef_ * decision_coeff / np.linalg.norm(logistic.coef_)**2

    # Sanity check
    assert (np.matmul(remainder_X, logistic.coef_.T) < 0.1).all()

    # Now get the main principal component of the rest.
    pca = decomposition.PCA(1)
    remainder_main = pca.fit_transform(remainder_X)
    return decision_coeff + logistic.intercept_, remainder_main


def project2d_classifier_goal(X, target, target2):
    logistic_1 = linear_model.LogisticRegression()
    logistic_1.fit(X, target)
    print(logistic_1.score(X, target))
    print(confusion_matrix(target, logistic_1.predict(X)))

    #print(logistic_1.intercept_)

    logistic_2 = linear_model.LogisticRegression()
    logistic_2.fit(X, target2)
    print(logistic_2.score(X, target2))
    #print(logistic_2.intercept_)

    #q, r = np.linalg.qr(np.hstack([logistic_1.coef_.T, logistic_2.coef_.T]))
    #projected = np.matmul(X, q)
    projected = np.matmul(X, np.hstack([logistic_1.coef_.T, logistic_2.coef_.T]))
    print(np.hstack([logistic_1.coef_.T, logistic_2.coef_.T]))
    return projected[:, 0] + logistic_1.intercept_, projected[:, 1] + logistic_2.intercept_


def project2d(X):
    pca = decomposition.PCA(2)
    projected = pca.fit_transform(X)
    return projected[:, 0], projected[:, 1]


def plot(xy, target):
    plt.figure()
    for mask, color in zip([target == 0, target == 1], ['r', 'b']):
        plt.scatter(xy[0][mask], xy[1][mask], c=color, alpha=0.5, marker='x')


def plot2(xy, target, target2):
    plt.figure()
    for mask, color in zip([target == 0, target == 1], ['r', 'b']):
        for mask2, marker in zip([target2 != 1, target2 == 1], ['v', '^']):
            plt.scatter(xy[0][mask & mask2], xy[1][mask & mask2], c=color, alpha=0.5, marker=marker)

