import operator
from cachetools import cached
import numpy as np
import numpy.random as npr
import scipy
import scipy.linalg as spla
import scipy.stats as sps

from bss import logger
from bss.utils.mcmc import elliptical_slice, slice_sample


class ProbitSS(object):
    def __init__(self, X, Y, R,
                 target_sparsity=0.01, gamma0_v=1.0,
                 lamb_a=1e-6, lamb_b=1e-6,
                 nu_a=1e-6, nu_b=1e-6,
                 xi_a=1.0, xi_b=1.0,
                 check_finite=True,
                 sample_xi=True,
                 min_eigenval=0, jitter=1e-6):
        r"""
        The Probit model used for our modelling for sparse regression using a Gaussian field.

        .. math::

            y|X,\beta,\beta_0, \nu \propto \mathcal{N}(\beta_0 1_n + X \beta, \nu^{-1} I_n)

        :param X: The predictor matrix of real numbers, n x p in size, where n is the no. of samples
            (genotypes) and p is the no. of features (SNPs)
        :param Y: The response vector of real numbers, n x 1 in size, with each value representing the
            phenotype value for the sample
        :param R: The covariance matrix for the SNPs, p x p in size. The matrix may not be positive-definite,
            but is converted to one internally.
        :param target_sparsity: The proportion of included predictors. For example, a value of 0.01 indicates that
            around 1% of total SNPs are expected be included in our model. This value affects the probit threshold
            $\gamma_0$ of the model.
        :param gamma0_v: Variance of the probit threshold $\gamma_0$
        :param lamb_a: Shape parameter of the gamma prior placed on the model parameter
            lambda, where lambda is the inverse squared global scale parameter for the regression weights.
        :param lamb_b: Inverse-scale parameter of the gamma prior placed on the model parameter
            lambda, where lambda is the inverse squared global scale parameter for the regression weights.

        """
        self.X = X
        self.Y = Y
        self.R = R

        self.N, self.P = self.X.shape

        self.nu_a = nu_a
        self.nu_b = nu_b
        self.xi_a = xi_a
        self.xi_b = xi_b

        self.sample_xi = sample_xi
        self.check_finite = check_finite

        if self.sample_xi:
            self._xi_distribution = sps.beta(self.xi_a, self.xi_b)
            self.xi = self._xi_distribution.mean()
        else:
            self.xi = 1.0 - 1e-6

        # Initialize scalar model parameters to their prior means.
        self._gamma0_distribution = sps.norm(loc=sps.norm.ppf(1.0 - target_sparsity), scale=gamma0_v)
        self.gamma0 = self._gamma0_distribution.mean()
        self._lambda_distribution = sps.gamma(lamb_a, scale=1./lamb_b)
        self.lamb = self._lambda_distribution.mean()
        self._nu_distribution = sps.gamma(self.nu_a, scale=1./self.nu_b)
        self.nu = self._nu_distribution.mean()

        # Ensure that the base correlation matrix is positive definite.
        # This also guarantees that there are ones along the diagonal.
        self._bend_matrix(min_eigenval, jitter)

        # Initialize the sparsity function.
        self.gamma = np.dot(self.get_cholesky(), npr.randn(self.P))
        # The above is an efficient way to sample from a multivariate gaussian distribution with known
        # covariance matrix and known means (zeros here). We could also have done:
        # self.gamma = sps.multivariate_normal.rvs(cov=self.get_covariance())
        # But we don't do so to avoid messing with the unit-testing results.

    def _bend_matrix(self, min_eigenval=0, jitter=1e-6):
        """Make the correlation matrix positive definite and normalized."""

        # Ensure that no eigenvalues are too small.
        # spla.eigh returns vector D and matrix V such that R = V*diag(D)*V'.
        evals, evecs = spla.eigh(self.R, check_finite=self.check_finite)
        evals[evals < min_eigenval] = min_eigenval
        self.R = np.dot(evecs, np.dot(np.diag(evals), evecs.T))

        # Add jitter and renormalize.
        self.R = self.R + jitter * np.eye(self.R.shape[0])
        diagR = np.diag(self.R)[:, np.newaxis]
        self.R = self.R / np.sqrt(diagR * diagR.T)

    def get_covariance(self, xi=None):
        """
        Compute the convex sum between R and the identity.
        Todo: cache on xi
        """
        if xi is None:
            xi = self.xi
        return (1.0 - xi) * np.eye(self.P) + xi * self.R

    @cached(cache={}, key=operator.attrgetter('xi'))
    def get_cholesky(self):
        return spla.cholesky(self.get_covariance(self.xi), lower=True, check_finite=False)

    def masked_covariance(self, gamma, gamma0, lamb):
        masked_X = self.X[:, gamma > gamma0]
        return (np.dot(masked_X, masked_X.T) + np.ones((self.N, self.N))) / lamb + np.eye(self.N)

    def log_marg_like(self, gamma, gamma0, nu, lamb):
        """
        Compute (or return the pre-computed) log marginal likelihood.
        """
        return sps.multivariate_normal(cov=self.masked_covariance(gamma, gamma0, lamb)/nu).logpdf(self.Y)

    def log_joint(self):
        return sum([
            self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb),
            self._gamma0_distribution.logpdf(self.gamma0),
            self._nu_distribution.logpdf(self.nu),
            self._lambda_distribution.logpdf(self.lamb),
            sps.multivariate_normal(cov=self.get_covariance()).logpdf(self.gamma),
            self._xi_distribution.logpdf(self.xi) if self.sample_xi else 0.0
        ])

    def run_mcmc(self, iters=1000, burnin=100, post_trace=False):

        logpost_trace = np.zeros(iters)
        inclusion_trace = np.zeros((iters, self.P), dtype=bool)

        for i in range(-burnin, iters):
            log_post = self.log_joint()

            logger.info(
                '%05d / %05d] logprob: %f [gamma0:%f lambda:%f nu:%f xi:%f' %
                (i, iters, log_post, self.gamma0, self.lamb, self.nu, self.xi)
            )

            self._update_parameters()

            if i >= 0:
                inclusion_trace[i, :] = self.gamma > self.gamma0
                logpost_trace[i] = log_post

        if post_trace:
            return logpost_trace
        else:
            return np.mean(inclusion_trace, 0), inclusion_trace[np.argmax(logpost_trace), :]

    def _update_parameters(self):
        self._update_gamma()
        self._update_gamma0()
        self._update_lambda()
        self.update_nu()
        if self.sample_xi:
            self._update_xi()

    def _update_gamma(self):
        """
        Apply MCMC transition operator to the function gamma.
        This is performed with an iteration of elliptical slice sampling.
        """

        # Construct the log likelihood for elliptical slice sampling.
        def slice_fn(gamma):
            return self.log_marg_like(gamma, self.gamma0, self.nu, self.lamb)

        self.gamma = elliptical_slice(self.gamma, self.get_cholesky(), slice_fn)

    def update_nu(self):
        r"""
        The scalar nu determines the precision of the residual Gaussian noise of the response variables.
        With the choice of a conjugate gamma prior distribution, the conditional posterior is also gamma:

        .. math::

            p(\nu | y, X, \Gamma, \lambda) \propto
            \mathcal{N}(y|0,\nu^{-1}(\lambda^{-1}(1_n1_n^T + X \Gamma X^T) + I_n)) \, Gam(\nu|a_\nu, b_\nu)

            = Gam(\nu | a_\nu^{(n)}, b_\nu^{(n)})

            a_\nu^{(n)} = a_\nu + \frac{N}{2}

            b_\nu^{(n)} = b_\nu + \frac{1}{2} y^T (\lambda^{-1} (1_n 1_n^T + X \Gamma X^T) + I_n)^{-1} y

        This function thus updates the value of nu using this analytical approach, by updating the parameters
        of the gamma distribution and then drawing a sample from it.
        """
        cov = self.masked_covariance(self.gamma, self.gamma0, self.lamb)

        distance = scipy.spatial.distance.mahalanobis(np.zeros(self.Y.shape), self.Y, np.linalg.inv(cov))
        post_a = self.nu_a + 0.5 * self.N
        post_b = self.nu_b + 0.5 * (distance**2)

        self.nu = npr.gamma(post_a, 1 / post_b)

    def _update_lambda(self):
        """
        Apply MCMC transition operator to the global weight inverse-scale parameter.
        """

        def slice_fn(lamb):
            if lamb < 0:
                return -np.inf
            return self.log_marg_like(self.gamma, self.gamma0, self.nu, lamb) + self._lambda_distribution.logpdf(lamb)

        self.lamb = slice_sample(self.lamb, slice_fn, verbose=False)

    def _update_gamma0(self):
        """
        Apply MCMC transition operator to the sparsity threshold.
        """

        def slice_fn(gamma0):
            return self.log_marg_like(self.gamma, gamma0, self.nu, self.lamb) + self._gamma0_distribution.logpdf(gamma0)

        self.gamma0 = slice_sample(self.gamma0, slice_fn, step_out=True)

    def _update_xi(self):
        """
        Apply MCMC transition operator to the correlation exponent.
        """
        # Compute the latent whitened variables.
        whitened = spla.solve_triangular(
            self.get_cholesky(), self.gamma, lower=True, trans=0, check_finite=self.check_finite
        )

        # Construct the slice sampling function.
        def slice_fn(xi):
            if xi <= 0 or xi >= 1:
                return -np.inf

            try:
                chol_cov = spla.cholesky(self.get_covariance(xi), lower=True, check_finite=self.check_finite)
            except np.linalg.linalg.LinAlgError:
                return -np.inf

            gamma = np.dot(chol_cov, whitened)

            return self.log_marg_like(gamma, self.gamma0, self.nu, self.lamb) + self._xi_distribution.logpdf(xi)

        self.xi = slice_sample(self.xi, slice_fn)
        self.gamma = np.dot(self.get_cholesky(), whitened)
