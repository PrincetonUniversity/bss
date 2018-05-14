from unittest import TestCase
import numpy as np
import itertools
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bss.utils.samplers import slice_sample
from bss.utils.samplers2 import SliceSampler


class SliceTestCase(TestCase):
    def setUp(self):
        np.random.seed(12345)

    def tearDown(self):
        pass

    def test_slice(self):

        # Log value of a univariate normal distribution with mean 0 and variance 1, at any point x
        def lognorm(x):
            y = norm.pdf(x)
            return np.log(y) if y>0 else -np.inf

        sampler = SliceSampler(lognorm)

        x0 = np.random.rand()
        trace = [x0] + list(itertools.islice(sampler.start(x0), 9))

        expected_trace = [
            0.9296160928171479, 0.941575910361962, 0.7220915230913388, 0.9616751866936547, -0.37423932861781595,
            0.2039944359771243, -0.24692804511312516, -0.2626026175464963, -0.8829065163861624, -0.20729683175068903
        ]
        self.assertTrue(np.allclose(expected_trace, trace))

