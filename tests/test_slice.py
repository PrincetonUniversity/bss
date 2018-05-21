from unittest import TestCase
import numpy as np
import itertools
from scipy.stats import norm
from bss.samplers.slice import SliceSampler


class SliceTestCase(TestCase):
    def setUp(self):
        np.random.seed(12345)

    def tearDown(self):
        pass

    def test_slice_trace(self):

        # Log value of a univariate normal distribution with mean 0 and variance 1, at any point x
        def lognorm(x):
            y = norm.pdf(x)
            return np.log(y) if y > 0 else -np.inf

        sampler = SliceSampler(lognorm)

        x0 = np.random.rand()
        trace = sampler.chain(x0, iters=15, burn_in=5)
        expected_trace = [
            -0.24692804511312516, -0.2626026175464963, -0.8829065163861624, -0.20729683175068903, -0.32319889775216437,
            -1.1694350886917464, -0.4179608528104426, -0.03021588067083758, 0.19220688805937514, 0.1247319144859459
        ]
        self.assertTrue(np.allclose(expected_trace, trace))

    def test_slice_sampling(self):
        """
        A non-deterministic test where we test the Slice Sampler with a normal distribution, and check to see if the
        sample mean and variance come "close" (with very liberal tolerances) to what we'd expect.
        """
        # Log value of a univariate normal distribution with mean 0 and variance 1, at any point x
        def lognorm(x):
            y = norm.pdf(x)
            return np.log(y) if y > 0 else -np.inf

        # This is a truly randomized test
        np.random.seed()

        iters = 10000  # total iterations of mcmc, including the burn-in
        burn_in = 5000

        sampler = SliceSampler(lognorm)

        x0 = np.random.rand()
        trace = [x0] + list(itertools.islice(sampler.start(x0), burn_in + iters))

        # Check to see if we're within 5% of the expected values
        self.assertTrue(np.abs(np.mean(trace[-iters:])) < 0.05)
        self.assertTrue(np.abs(np.var(trace[-iters:]) - 1) < 0.05)
