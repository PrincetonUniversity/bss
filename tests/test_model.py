import os.path
import numpy as np
from unittest import TestCase

from bss.models.probit import ProbitSS
from bss.utils.data import load_data

DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')


class ModelTestCase(TestCase):
    def setUp(self):
        np.random.seed(1)

    def tearDown(self):
        pass

    def test_model1(self):
        X, y, _, R = load_data(os.path.join(DATA_DIR, 'real0_*_10000.out'))
        model = ProbitSS(
            X=X,
            Y=y,
            R=R,
            sample_xi=True
        )
        trace = model.run_mcmc(burnin=5, iters=10, post_trace=True)
        expected_trace = [
            -2712.1243232190068, -2816.0395965482576, -2827.3281565130937, -1582.3140750335701, -2805.4378869844559,
            -2837.8348894101946, -2864.5580147460505, -2484.2768900178071, -2382.4815505858278, -2796.0928580805762
        ]
        self.assertEqual(len(trace), len(expected_trace))
        for i, val in enumerate(expected_trace):
            self.assertAlmostEqual(val, trace[i], places=6)

    def test_model2(self):
        X, y, _, R = load_data(os.path.join(DATA_DIR, 'real0_*_10000.out'))
        model = ProbitSS(
            X=X,
            Y=y,
            R=R,
            sample_xi=False
        )
        trace = model.run_mcmc(burnin=5, iters=10, post_trace=True)
        expected_trace = [
            4613.3151614505305, 4700.0565201187101, 4651.464728613646, 4654.9619244943506, 4628.7974616871106,
            4629.8682792805203, 4637.0845615021908, 4658.0776869396414, 4609.6594171963898, 4672.197914151111
        ]
        self.assertEqual(len(trace), len(expected_trace))
        for i, val in enumerate(expected_trace):
            self.assertAlmostEqual(val, trace[i], places=6)
