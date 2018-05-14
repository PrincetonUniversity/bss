import numpy as np
import scipy
import scipy.linalg

_LOG_2PI = np.log(2 * np.pi)


class multivariate_normal_gen(object):
    def __call__(self, *args, **kwargs):
        return multivariate_normal_frozen(*args, **kwargs)

    def _maha(self, x, mean, chol, precision_multiplier=1):
        # Return the squared Mahalanobis distance for a given observation x with mean, chol and precision multiplier
        dev = x - mean
        solve = scipy.linalg.solve_triangular(chol/np.sqrt(precision_multiplier), dev, lower=True, trans=0)
        return np.dot(solve.T, solve)

    def _logpdf(self, x, mean, chol, log_det_cov, rank, precision_multiplier=1):
        maha = self._maha(x, mean, chol, precision_multiplier)
        return -0.5 * (rank * (_LOG_2PI - np.log(precision_multiplier)) + log_det_cov + maha)


class multivariate_normal_frozen(object):
    def __init__(self, cov, mean=None, min_eigenval=None, jitter=None, check_finite=True):
        self.d = cov.shape[0]
        if mean is None:
            mean = np.zeros(self.d)
        self._dist = multivariate_normal_gen()
        self.mean = mean
        self.check_finite = check_finite
        self.cov_info = _PD(cov, min_eigenval=min_eigenval, jitter=jitter, lower=True, check_finite=check_finite)
        self.cov = self.cov_info.M

    @property
    def chol(self):
        return self.cov_info.chol

    def rvs(self, precision_multiplier=1):
        return self.mean + np.dot(self.chol / np.sqrt(precision_multiplier), np.random.randn(self.d))

    def maha(self, x, precision_multiplier=1):
        return self._dist._maha(x, self.mean, self.cov_info.chol, precision_multiplier)

    def logpdf(self, x, precision_multiplier=1):
        return self._dist._logpdf(x, self.mean, self.cov_info.chol, self.cov_info.log_pdet, self.d, precision_multiplier)

    def solve(self, x):
        # TODO: Verify math
        return scipy.linalg.solve_triangular(self.chol, x, lower=True, trans=0, check_finite=self.check_finite)

    def dot(self, x):
        # TODO: Verify math
        return np.dot(self.chol, x)


class _PD(object):
    """
    Compute coordinated functions of a symmetric positive definite matrix.

    This class is inspired by the _PSD class in the scipy.stats._multivariate package
    Where the _PSD class stores the symmetric eigendecomposition of a symmetric semidefinite matrix, we choose to
    store the Cholesky decomposition of a symmetric positive definite matrix. A symmetric positive definite matrix fits
     our notion of a covariance matrix across genotypes better.

    If the incoming covariance matrix is not positive definite, we make it so inside the constructor for this class,
    by eliminating negative eigenvalues. To avoid conditioning issues (i.e. when the covariance matrix might be
    singular), we add a small amount of jitter, or noise, to the diagonal elements.

    """
    @classmethod
    def _fix(cls, M, min_eigenval=None, jitter=None, check_finite=True):
        d = M.shape[0]
        if min_eigenval is not None:
            # scipy.linalg.eigh returns a vector w of eigenvalues, and a matrix v of eigenvectors.
            # The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i]
            # We therefore have:
            # M = v * diag(w) * v'
            evals, evecs = scipy.linalg.eigh(M, check_finite=check_finite)

            # eliminate negative eigenvalues
            evals[evals < min_eigenval] = min_eigenval

            # recompute the covariance matrix using the standard notion of eigenvectors and eigenvalues
            M = np.dot(evecs, np.dot(np.diag(evals), evecs.T))

        if jitter is not None:
            # Add jitter and renormalize
            M = M + jitter * np.eye(d)
            diagR = np.diag(M)[:, np.newaxis]
            M = M / np.sqrt(diagR * diagR.T)

        return M

    def __init__(self, M, min_eigenval=0, jitter=1e-6, lower=True, check_finite=True):

        assert M.ndim == 2, "The input covariance matrix must be 2-dimensional"
        assert M.shape[0] == M.shape[1], "The input covariance matrix must be square"

        M = _PD._fix(M, min_eigenval, jitter, check_finite)
        chol = scipy.linalg.cholesky(M, lower=lower, check_finite=check_finite)

        # Initialize the eagerly precomputed attributes.
        self.rank = M.shape[0]  # todo: do this properly!
        self.M = M
        self.chol = chol
        self.log_pdet = 2 * np.sum(np.log(np.diag(chol)))


multivariate_normal = multivariate_normal_gen()
