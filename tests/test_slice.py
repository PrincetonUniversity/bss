from unittest import TestCase
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bss.utils.samplers import slice_sample


class SliceTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_slice(self):
        def lognorm(x):
            y = norm.pdf(x)
            return np.log(y) if y>0 else -np.inf
            # return np.log(beta(5,2).pdf(x))

        vals = [np.random.rand()]
        for i in range(1, 1000):
            y = slice_sample(vals[-1], lognorm)
            vals.append(y)

        # fig1 = plt.figure()
        # plt.hist(vals, bins=30)
        # plt.show()
        self.assertTrue(True)


