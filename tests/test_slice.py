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

        sampler = SliceSampler(lognorm)
        x0 = np.random.rand()
        i = 0
        samples = sampler.sample(x0)
        for x in samples:
            print(x)
            i += 1
            if i==10:
                break



