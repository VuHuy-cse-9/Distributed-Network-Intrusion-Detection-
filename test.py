from sklearn.svm import SVC
import time
import numpy as np

N = 5
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N, 2))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N, 2))
X = np.concatenate((X0, X1), axis=0)
Y = np.concatenate((np.ones((N, )), -1*np.ones((N, ))))
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")

svc = SVC(kernel = 'linear', C = 1e5)
svc.fit(X, Y)

N = 10
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N, 2))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N, 2))
X = np.concatenate((X0, X1), axis=0)
Y = np.concatenate((np.ones((N, )), -1*np.ones((N, ))))
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")
svc.fit(X, Y)

print(svc.predict(X))