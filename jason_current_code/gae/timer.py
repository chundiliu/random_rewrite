from preprocess_graph import *
import time
import numpy as np
Q, X = load_data()

start = time.time()
a = np.matmul(X.T, Q)
b = np.argsort(-a, axis=0)
elapse = time.time() - start
print(elapse)
print(elapse / 70.0)
print(a.shape)
