from unittest import TestCase
import numpy as np
import itertools
from scipy.stats import multivariate_normal, norm

from bss.samplers.elliptical import EllipticalSliceSampler


class SliceTestCase(TestCase):
    def setUp(self):
        np.random.seed(12345)

    def tearDown(self):
        pass

    def test_ess_trace(self):

        log_likelihood_dist = multivariate_normal([1, 0, 2, 0.1, -0.3], 2*np.eye(5))

        # Log value of a univariate normal distribution with mean 0 and variance 1, at any point x
        def log_likehood_func(x):
            return log_likelihood_dist.logpdf(x)

        prior_dist = multivariate_normal(np.zeros(5), np.eye(5))
        sampler = EllipticalSliceSampler(prior_dist, log_likehood_func)

        x0 = prior_dist.rvs()
        trace = sampler.chain(x0, iters=5, burn_in=0)
        expected_trace = [
            [-0.53412099, 0.44249311, -0.57198773, -0.72451367, 1.6079185],
            [1.13995897, -1.35510654, 0.50610655, 0.53487132, 0.46795533],
            [-0.48761705, -2.00788121, 0.57430304, 0.67940248, -0.3076545],
            [1.23135582, -2.24824447, 0.20128533, 0.64819096, -0.11114885],
            [-0.37489531, -2.06416557, 0.37259254, -0.05097587, 1.2912787]
        ]

        self.assertTrue(np.allclose(trace, expected_trace))

    def test_ess_sampling(self):
        """
        A non-deterministic test where we test the Elliptical Slice Sampler with a normal prior distribution and a
        normal likelihood function, and check to see if the sample mean and variance come "close" (with very liberal
        tolerances) to what we'd expect from the analytical solution.
        """

        # This is a truly randomized test
        np.random.seed()

        iters = 50000  # total iterations of mcmc, including the burn-in
        burn_in = 30000

        # Our prior and likelihood are both normal with parameters (mu1, std1) and (mu2, std2) respectively.
        mu1, mu2 = 0., 2.
        std1, std2 = 1.3, 1.4
        var1, var2 = std1**2, std2**2

        # The product of the prior and likelihood function is a normal distribution with parameters:
        mu = (mu2 * var1 + mu1 * var2) / (var1 + var2)  # ~0.926
        var = 1. / (1./var1 + 1./var2)  # ~0.907

        prior_dist = norm(mu1, std1)
        log_likelihood_dist = norm(mu2, std2)

        # Log value of a univariate normal distribution with mean 0 and variance 1, at any point x
        def log_likehood_func(x):
            return log_likelihood_dist.logpdf(x)

        sampler = EllipticalSliceSampler(prior_dist, log_likehood_func)

        x0 = prior_dist.rvs()
        trace = [x0] + list(itertools.islice(sampler.start(x0), burn_in + iters))

        sample_mu = np.mean(trace[burn_in:])
        sample_var = np.var(trace[burn_in:])

        # Liberal tolerances
        self.assertAlmostEqual(mu, sample_mu, places=1)
        self.assertAlmostEqual(var, sample_var, places=1)
