"""
A multivariate normal module inspired by the scipy.stats._multivariate module.
"""

import numpy as np
import scipy
import scipy.linalg

_LOG_2PI = np.log(2 * np.pi)


class Mvn:
    """A multivariate normal random variable.

    Parameters
    ----------
    mean : ndarray, optional
       The dx1 mean vector of the normal distribution. Assumed 0 if not specified.
    cov : ndarray
       The dxd covariance matrix. Assumed the Identity matrix if not specified.
    min_eigenval : float, optional
       The minimum eigenvalue of the covariance matrix we're willing to accept. All values below this threshold
       are set to this value. If None (the default), no eignvalues are adjusted.
       A useful value is 0, which results in the covariance matrix being made positive definite.
    jitter: float, optional
        A small amount of noise to add to the diagonals of the covariance matrix to make the covariance matrix
        invertible. If None (the default), then no jitter is applied.
    """
    def __init__(self, mean=None, cov=None, min_eigenval=None, jitter=None, check_finite=True):
        assert mean is not None or cov is not None, "At least one of mean or cov must be specified"
        if mean is None:
            cov = np.array(cov)
            self.d = cov.shape[0]
            mean = np.zeros(self.d)
        elif cov is None:
            mean = np.array(mean)
            self.d = mean.shape[0]
            cov = np.eye(self.d)

        mean = np.array(mean)
        cov = np.array(cov)

        assert cov.ndim == 2, "Covariance matrix not 2D"
        assert cov.shape[0] == cov.shape[1], "Covariance matrix not square"
        assert mean.shape[0] == cov.shape[0], "Mean vector and covariance matrices not expected shape"

        self.mean = mean
        self.check_finite = check_finite
        self.cov_info = _PD(cov, min_eigenval=min_eigenval, jitter=jitter, lower=True, check_finite=check_finite)
        self.cov = self.cov_info.M

    @property
    def chol(self):
        """ndarray: The Cholesky decomposition of the covariance matrix of this distribution. Computed after the
        covariance matrix is made positive definite inside the constructor.
        """
        return self.cov_info.chol

    def rvs(self, precision_multiplier=1):
        """
        Generate a random variate for this normal distribution, optionally applying a multiplicative factor to the
        precision matrix (or equivalently, dividing the covariance matrix by a factor).

        Parameters
        ----------
        precision_multiplier : float, optional
            Optional multiplier for the precision term in the covariance matrix, 1 by default.

        Returns
        -------
        ndarray
            A single Nx1 random variate from this normal distribution
        """
        return self.mean + np.dot(self.chol / np.sqrt(precision_multiplier), np.random.randn(self.d))

    def maha(self, x, precision_multiplier=1):
        """
        Calculate the Mahalanobis distance between a given vector x and the mean of this distribution, optionally
        applying a multiplicative factor to the precision matrix (or equivalently, dividing the covariance matrix by a
        factor).

        Parameters
        ----------
        x : ndarray
            The vector for which we wish to calculate the distance
        precision_multiplier : float, optional
            Optional multiplier for the precision term in the covariance matrix, 1 by default.

        Returns
        -------
        float
            A scalar distance value between x and the mean of this distribution
        """
        dev = x - self.mean
        solve = scipy.linalg.solve_triangular(self.chol/np.sqrt(precision_multiplier), dev, lower=True, trans=0)
        return np.dot(solve.T, solve)

    def logpdf(self, x, precision_multiplier=1):
        """
        Calculate the Log Probability Density Function value of a given variate, optionally applying a multiplicative
        factor to the precision matrix (or equivalently, dividing the covariance matrix by a factor).

        Parameters
        ----------
        x : ndarray
            The vector for which we wish to calculate the log PDF value
        precision_multiplier : float, optional
            Optional multiplier for the precision term in the covariance matrix, 1 by default.

        Returns
        -------
        float
            The Log PDF value of the vector x
        """
        maha = self.maha(x, precision_multiplier)
        return -0.5 * (self.d * (_LOG_2PI - np.log(precision_multiplier)) + self.cov_info.log_pdet + maha)

    def whiten(self, x):
        """
        Transform the dx1 random variate x into a whitened random vector with unit diagonal covariance.

        Parameters
        ----------
        x : ndarray
            The dx1 vector for that we wish to whiten, or more generally, the dxN data matrix that wish to whiten

        Returns
        -------
        ndarray
            The dx1 whitened vector, or more generally, the dxN whitened data matrix

        Notes
        -----
        For a detailed explanation of why this may be useful, see :cite:`Murray2010b`
        """
        return scipy.linalg.solve_triangular(self.chol, (x - self.mean).T, lower=True, trans=0, check_finite=self.check_finite)

    def correlate(self, x):
        """
        Transform a random variate x into a variate correlated according to this Multivariate normal distribution,
        and centered around this distribution's mean.
        This is accomplished by affine transforming the given data vector or data matrix.

        Parameters
        ----------
        x : ndarray
            The dx1 vector for that we wish to transform, or more generally, the dxN data matrix that wish to transform

        Returns
        -------
        ndarray
            The dx1 transformed vector, or more generally, the dxN transformed data matrix

        Notes
        -----
        Uncorrelated random variables normally distributed with mean 0 and variance 1
            Z ~ N(0, I)
        can be transformed to correlated random variables X with mean A and covariance Sigma:
            X ~ N(A, Sigma)
        by selecting an affine transform:
            X = A + BZ
        where:
            B B' = Sigma
        We choose B to be the Cholesky factorization, since we have computed it for this class.

        For a detailed explanation of why this may be useful, see :cite:`Murray2010b`
        """
        return np.dot(self.chol, x.T).T + self.mean


class _PD:
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
            """
            scipy.linalg.eigh returns a vector w of eigenvalues, and a matrix v of eigenvectors.
            The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i]
            We therefore have:
            M = v * diag(w) * v'
            """
            evals, evecs = scipy.linalg.eigh(M, check_finite=check_finite)

            # eliminate eigenvalues smaller than min_eigenval
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
        M = _PD._fix(M, min_eigenval, jitter, check_finite)
        chol = scipy.linalg.cholesky(M, lower=lower, check_finite=check_finite)

        # Initialize the eagerly precomputed attributes
        self.M = M
        self.chol = chol
        self.log_pdet = 2 * np.sum(np.log(np.diag(chol)))
