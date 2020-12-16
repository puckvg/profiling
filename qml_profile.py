import numpy as np
import qml
#import sklearn
from qml.kernels import gaussian_kernel
from qml.math import cho_solve
#from sklearn.kernel_ridge import KernelRidge


def generate_dummy_data():
    X = np.random.rand(1000, 1000)
    y = np.random.rand(1000)

    return X, y


def compute_kernel_qml(X_i, X_j, sigma=1e3):
    return gaussian_kernel(X_i, X_j, sigma)


def train_KRR_qml(X, y, sigma=1e3, llambda=1e-8):
    K = compute_kernel_qml(X, X, sigma=sigma)
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, y)
    return alpha


def train_KRR_sklearn(X, y, sigma=1e-3, llambda=1e-8):
    clf = KernelRidge(alpha=llambda, kernel="rbf", gamma=sigma)
    clf.fit(X, y)
    return clf


def predict_KRR_sklearn(
    X_test, y_test, X_train=None, y_train=None, clf=None, sigma=1e3, llambda=1e-8
):
    if not clf:
        assert all(
            v is not None for v in [X_train, y_train]
        ), "cannot train model without training data"
        clf = train_KRR_sklearn(X_train, y_train, sigma=sigma, llambda=llambda)
    y_pred = clf.predict(X_test)
    return y_pred


def predict_KRR_qml(
    X_test, y_test, X_train=None, y_train=None, alpha=None, sigma=1e3, llambda=1e-8
):
    if not alpha:
        assert all(
            v is not None for v in [X_train, y_train]
        ), "cannot train model without training data"
        alpha = train_KRR_qml(X_train, y_train, sigma=sigma, llambda=llambda)
    Ks = compute_kernel_qml(X_test, X_train, sigma)
    y_pred = np.dot(Ks, alpha)

    return y_pred

def main(X, y, test_size=0.33):
    train_size = 1-test_size 
    train_int = int(train_size*len(X))
    
    X_train, X_test = X[:train_int], X[train_int:]
    y_train, y_test = y[:train_int], y[train_int:]
    pred_qml = predict_KRR_qml(X_test, y_test, X_train=X_train, y_train=y_train)
 #   pred_sklearn = predict_KRR_sklearn(X_test, y_test, X_train=X_train, y_train=y_train)
    return 

if __name__ == "__main__":
    X, y = generate_dummy_data()
    main(X,y)
