import operator
from cachetools import cached, cachedmethod, Cache
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps

from bss import logger
from bss.utils.mcmc import elliptical_slice, slice_sample
from bss.utils.math import multivariate_normal


class Probit(object):
    def __init__(self, X, Y, R, target_sparsity=0.01, gamma0_v=1.0, lamb_a=1e-6, lamb_b=1e-6, nu_a=1e-6, nu_b=1e-6,
                 xi=0.999999, xi_prior_shape=(1, 1), check_finite=True, min_eigenval=0, jitter=1e-6):
        r"""
        The Probit model used for our modeling for sparse regression using a Gaussian field.

        .. math::

            y|X,\beta,\beta_0, \nu \propto \mathcal{N}(\beta_0 1_n + X \beta, \nu^{-1} I_n)

        :param X: The predictor matrix of real numbers, n x p in size, where n is the no. of samples
            (genotypes) and p is the no. of features (SNPs).
        :param Y: The response vector of real numbers, n x 1 in size, with each value representing the
            phenotype value for the sample.
        :param R: The covariance matrix for the SNPs, p x p in size. The matrix may not be positive-definite,
            but is converted to one internally.
        :param target_sparsity: The proportion of included predictors. For example, a value of 0.01 indicates that
            around 1% of total SNPs are expected be included in our model. This value affects the probit threshold
            gamma_0 of the model.
        :param gamma0_v: Variance of the probit threshold gamma_0
        :param lamb_a: Shape parameter of the gamma prior placed on the model parameter
            lambda, where lambda is the inverse squared global scale parameter for the regression weights.
        :param lamb_b: Inverse-scale parameter of the gamma prior placed on the model parameter
            lambda, where lambda is the inverse squared global scale parameter for the regression weights.
        :param nu_a: Shape parameter of the gamma prior placed on the model parameter
            nu, where nu is the residual precision.
        :param nu_b: Inverse-scale parameter of the gamma prior placed on the model parameter
            nu, where nu is the residual precision.
        :param xi: The shrinkage constant in the interval [0,1] to regularize the covariance matrix towards the
            identity matrix. This ensures that the covariance matrix is positive definite.
            If None, then xi is sampled from a beta distribution with shape parameters specified by the tuple
            xi_prior_shape.
        :param xi_prior_shape: Shape parameters of the beta prior placed on the model parameter xi, specified as a
            2-tuple of real values. xi is the shrinkage constant to regularize the covariance matrix towards the
            identity matrix. This ensures that the covariance matrix is positive definite.
            This argument is ignored and xi is not sampled, if it is specified explicitly using the xi parameter.
        :param check_finite: Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.
            This parameter is passed on to several linear algebra functions in scipy internally.
        :param min_eigenval: Minimum Eigenvalue we can accept in the covariance matrix. Any eigenvalues encountered
            below this threshold are set to zero, and the resulting covariance matrix normalized to give ones on the
            diagonal.
        :param jitter: A small value to add to the diagonals of the covariance matrix to avoid conditioning issues.
        """
        self.X = X
        self.Y = Y
        self.R = R

        self.N, self.P = self.X.shape

        self.nu_a, self.nu_b = nu_a, nu_b

        self.check_finite = check_finite

        if xi is None:
            self.sample_xi = True
            self._xi_distribution = sps.beta(*xi_prior_shape)
            self.xi = self._xi_distribution.mean()
        else:
            self.sample_xi = False
            self.xi = xi

        # Initialize scalar model distributions and the parameter values to their prior means.
        self._gamma0_distribution = sps.norm(loc=sps.norm.ppf(1.0 - target_sparsity), scale=gamma0_v)
        self.gamma0 = self._gamma0_distribution.mean()
        self._lambda_distribution = sps.gamma(lamb_a, scale=1./lamb_b)
        self.lamb = self._lambda_distribution.mean()
        self._nu_distribution = sps.gamma(self.nu_a, scale=1./self.nu_b)
        self.nu = self._nu_distribution.mean()

        self._bend_matrix(min_eigenval, jitter)
        self.cov = multivariate_normal(cov=R, min_eigenval=min_eigenval, jitter=jitter)

        # Initialize the sparsity function.
        self.gamma = self.get_covariance(self.xi).rvs()

        self.cache = Cache(maxsize=1)


    def _bend_matrix(self, min_eigenval=0, jitter=1e-6):
        # Make the correlation matrix positive definite and normalized.
        # This also guarantees that there are ones along the diagonal.

        # spla.eigh returns vector D and matrix V such that R = V*diag(D)*V'.
        evals, evecs = spla.eigh(self.R, check_finite=self.check_finite)
        evals[evals < min_eigenval] = min_eigenval
        self.R = np.dot(evecs, np.dot(np.diag(evals), evecs.T))

        # Add jitter and renormalize.
        self.R = self.R + jitter * np.eye(self.P)
        diagR = np.diag(self.R)[:, np.newaxis]
        self.R = self.R / np.sqrt(diagR * diagR.T)

    # @cached(cache={})
    def get_covariance(self, xi):
        x1 = multivariate_normal(cov=(xi * self.R) + (1.0 - xi) * np.eye(self.P))
        x2 = multivariate_normal(cov=(xi * self.cov.cov) + (1.0 - xi) * np.eye(self.P))
        assert(np.allclose(x1.cov, x2.cov))
        return x2

    def masked_covariance(self, gamma, gamma0, lamb):
        masked_X = self.X[:, gamma > gamma0]
        return self.masked_covariance2(masked_X, lamb)

    def mykey(self, *args):
        key = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key.append(hash(arg.data.tobytes()))
            else:
                key.append(arg)
        return tuple(key)

    @cachedmethod(cache=operator.attrgetter('cache'), key=mykey)
    def masked_covariance2(self, masked_X, lamb):
        return multivariate_normal((np.dot(masked_X, masked_X.T) + np.ones((self.N, self.N))) / lamb + np.eye(self.N))

    def log_marg_like(self, gamma, gamma0, nu, lamb):
        return self.masked_covariance(gamma, gamma0, lamb).logpdf(self.Y, precision_multiplier=nu)

    def log_joint(self):
        return sum([
            self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb),
            self._gamma0_distribution.logpdf(self.gamma0),
            self._nu_distribution.logpdf(self.nu),
            self._lambda_distribution.logpdf(self.lamb),
            self.get_covariance(self.xi).logpdf(self.gamma),
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

        self.gamma = elliptical_slice(self.gamma, self.get_covariance(self.xi).chol, slice_fn)

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

        distance_sq = cov.maha(self.Y)
        post_a = self.nu_a + 0.5 * self.N
        post_b = self.nu_b + 0.5 * distance_sq

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
            self.get_covariance(self.xi).chol, self.gamma, lower=True, trans=0, check_finite=self.check_finite
        )

        # Construct the slice sampling function.
        def slice_fn(xi):
            if xi <= 0 or xi >= 1:
                return -np.inf

            try:
                chol_cov = self.get_covariance(np.float64(xi)).chol
            except np.linalg.linalg.LinAlgError:
                return -np.inf

            gamma = np.dot(chol_cov, whitened)

            return self.log_marg_like(gamma, self.gamma0, self.nu, self.lamb) + self._xi_distribution.logpdf(xi)

        self.xi = slice_sample(self.xi, slice_fn)
        self.gamma = np.dot(self.get_covariance(self.xi).chol, whitened)
