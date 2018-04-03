import scipy.linalg as spla
import numpy as np
import numpy.random as npr
import pylab as pl

npr.seed(13)

N = 500
x = np.linspace(0, 5, N)
K = np.exp(-0.5*(x[:, np.newaxis]-x[np.newaxis, :])**2)

(evals, evecs) = spla.eigh(K)

z = npr.randn(N)

exponent = 0.2

newK = K**exponent
# dnewK = np.diag(newK)[:,np.newaxis]
# newK = (newK / np.sqrt(dnewK * dnewK.T)) + 1e-6*np.eye(N)

pl.subplot(2, 2, 1)
pl.imshow(K)
pl.colorbar()

pl.subplot(2, 2, 3)
pl.plot(x, np.dot(spla.cholesky(K+1e-6*np.eye(N), lower=True), z))

pl.subplot(2, 2, 2)
pl.imshow(newK)
pl.colorbar()

pl.subplot(2, 2, 4)
pl.plot(x, np.dot(spla.cholesky(newK+1e-6*np.eye(N), lower=True), z))


pl.show()
