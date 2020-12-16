import qml 
from qml.math import cho_solve
from qml.kernels import gaussian_kernel
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.kernel_ridge import KernelRidge
import numpy as np 


def generate_dummy_data():
    X = np.random.rand(100,100)
    y = np.random.rand(100)

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


def predict_KRR_sklearn(X_test, y_test, X_train=None, y_train=None,
        clf=None, sigma=1e3, llambda=1e-8):
    if not clf: 
        assert all(v is not None for v in [X_train, y_train]), "cannot train model without training data"
        clf = train_KRR_sklearn(X_train, y_train, sigma=sigma, llambda=llambda)
    y_pred = clf.predict(X_test)
    return y_pred


def predict_KRR_qml(X_test, y_test, X_train=None, y_train=None, alpha=None, 
                    sigma=1e3, llambda=1e-8):
    if not alpha: 
        assert all(v is not None for v in [X_train, y_train]), "cannot train model without training data"
        alpha = train_KRR_qml(X_train, y_train, sigma=sigma, llambda=\
                llambda)
    Ks = compute_kernel_qml(X_test, X_train, sigma)
    y_pred = np.dot(Ks, alpha)

    return y_pred 


if __name__ == "__main__":
    X, y = generate_dummy_data()
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    pred_qml = predict_KRR_qml(X_test, y_test, X_train=X_train, y_train=y_train)
    pred_sklearn = predict_KRR_sklearn(X_test, y_test, X_train=X_train, y_train=y_train)
    





