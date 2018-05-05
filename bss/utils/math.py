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

    def logpdf(self, x, cov, mean=None):
        pd = _PD(cov)
        raise NotImplementedError('bah')
        # return self._logpdf(x, mean, pd.chol, pd.log_pdet, pd.rank)


class multivariate_normal_frozen(object):
    def __init__(self, cov, mean=None, min_eigenval=None, jitter=None):
        self.d = cov.shape[0]
        if mean is None:
            mean = np.zeros(self.d)
        self._dist = multivariate_normal_gen()
        self.mean, self.cov = mean, cov
        self.cov_info = _PD(self.cov, min_eigenval=min_eigenval, jitter=jitter, lower=True, check_finite=True)

    @property
    def chol(self):
        return self.cov_info.chol

    def rvs(self):
        return np.dot(self.chol, np.random.randn(self.d))

    def maha(self, x, precision_multiplier=1):
        return self._dist._maha(x, self.mean, self.cov_info.chol, precision_multiplier)

    def logpdf(self, x, precision_multiplier=1):
        return self._dist._logpdf(x, self.mean, self.cov_info.chol, self.cov_info.log_pdet, self.d, precision_multiplier)


class _PD(object):

    @classmethod
    def _fix(cls, M, min_eigenval=None, jitter=None, check_finite=True):
        d = M.shape[0]
        if min_eigenval is not None:
            evals, evecs = scipy.linalg.eigh(M, check_finite=check_finite)
            evals[evals < min_eigenval] = min_eigenval
            M = np.dot(evecs, np.dot(np.diag(evals), evecs.T))

        if jitter is not None:
            # Add jitter and renormalize.
            M = M + jitter * np.eye(d)
            diagR = np.diag(M)[:, np.newaxis]
            M = M / np.sqrt(diagR * diagR.T)

        return M

    def __init__(self, M, min_eigenval=0, jitter=1e-6, lower=True, check_finite=True):
        M = _PD._fix(M, min_eigenval, jitter, check_finite)
        chol = scipy.linalg.cholesky(M, lower=lower, check_finite=check_finite)

        # Initialize the eagerly precomputed attributes.
        self.rank = M.shape[0]  # todo: do this properly!
        self.chol = chol
        self.log_pdet = 2 * np.sum(np.log(np.diag(chol)))


multivariate_normal = multivariate_normal_gen()


if __name__ == '__main__':
    x = np.load('x.npy')
    cov = np.load('cov.npy')
    dist = multivariate_normal(cov=cov)
    print(dist.chol)
    res = dist.logpdf(x)
    print(res)
    res = dist.logpdf(x)
    print(res)
    res = dist.logpdf(x, precision_multiplier=1.1)
    print(res)

