from unittest import TestCase
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bss.utils.samplers import slice_sample
from bss.utils.samplers2 import SliceSampler

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

        x0 = np.random.rand()
        l1 = [x0]
        l2 = [x0]
        N = 1000

        np.random.seed(12345)
        sampler = SliceSampler(lognorm)
        i = 0
        for x in sampler.start(x0):
            l1.append(x)
            i += 1
            if i==N:
                break

        np.random.seed(12345)
        for i in range(N):
            x0 = slice_sample(x0, lognorm)
            l2.append(x0)

        print(l1)
        print('---------')
        print(l2)
        self.assertTrue(np.allclose(l1, l2))



