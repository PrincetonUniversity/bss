import os.path
import numpy as np
from unittest import TestCase

from bss.models.probit import Probit
from bss.utils.data import load_data

DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')


class ModelTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, _, cls.R = load_data(os.path.join(DATA_DIR, 'real0_*_10000.out'))

    def setUp(self):
        np.random.seed(1)

    def tearDown(self):
        pass

    def test_model1(self):
        model = Probit(
            X=self.X,
            Y=self.y,
            R=self.R,
            xi=None
        )
        trace = model.run_mcmc(burnin=5, iters=10)
        expected_trace = [
            -2712.1243232190068, -2816.0395965482576, -2827.3281565130937, -1582.3140750335701, -2805.4378869844559,
            -2837.8348894101946, -2864.5580147460505, -2484.2768900178071, -2382.4815505858278, -2796.0928580805762
        ]
        self.assertEqual(len(trace), len(expected_trace))
        for x, y in zip(trace, expected_trace):
            self.assertAlmostEqual(x, y, places=6)

    def test_model2(self):
        model = Probit(
            X=self.X,
            Y=self.y,
            R=self.R
        )
        trace = model.run_mcmc(burnin=5, iters=10)
        expected_trace = [
            4613.3151614505305, 4700.0565201187101, 4651.464728613646, 4654.9619244943506, 4628.7974616871106,
            4629.8682792805203, 4637.0845615021908, 4658.0776869396414, 4609.6594171963898, 4672.197914151111
        ]
        self.assertEqual(len(trace), len(expected_trace))
        for x, y in zip(trace, expected_trace):
            self.assertAlmostEqual(x, y, places=6)

    def test_model_detailed_trace(self):
        model = Probit(
            X=self.X,
            Y=self.y,
            R=self.R,
            xi=None
        )
        detailed_trace = model.run_mcmc(burnin=5, iters=10, detailed=True)

        expected = {
            'joint': [
                -2712.1243232190068, -2816.0395965482576, -2827.3281565130937, -1582.3140750335701, -2805.4378869844559,
                -2837.8348894101946, -2864.5580147460505, -2484.2768900178071, -2382.4815505858278, -2796.0928580805762
            ],
            'likelihood': [
                -684.06472998, -682.95231508, -682.53844185, -681.25566689, -680.87534291,
                -680.64811811, -681.42748042, -680.54291735, -680.97633156, -680.45279408
            ],
            'xi': [
                0.04826542, 0.04400981, 0.85999135, 0.13222765, 0.04636762, 0.07791028,
                0.44413487, 0.55616501, 0.10905116, 0.634995
            ],
            'nu': [
                1.03414633, 1.00874212, 0.98707078, 1.03425261, 0.97088564,
                1.08444457, 1.02924025, 0.93905739, 1.00373751, 1.07782064
            ],
            'gamma0': [
                4.53206381, 4.20932857, 4.0223458, 4.63599829, 2.78279365,
                1.52285834, 2.99568083, 3.42461735, 3.36309594, 2.48723791
            ],
            'lambda': [
                4.03646936e+00, 7.44532960e+00, 1.26766841e+02, 5.45426416e+02, 6.34643878e+03,
                21230.900822630727, 13882.875127113773, 8300.9957627273707, 25404.505387005316, 98950.409313213691
            ]
        }

        for k, v in expected.items():
            for x, y in zip(detailed_trace[k], v):
                self.assertAlmostEqual(x, y, places=6)

    def _test_model_conditional(self):
        X, y, _, R = load_data(os.path.join(DATA_DIR, 'real0_*_10000.out'))
        model = Probit(
            X=X,
            Y=y,
            R=R
        )
        trace = model.run_geweke(iters=200, burnin=50)
        # expected_trace = [
        #     4613.3151614505305, 4700.0565201187101, 4651.464728613646, 4654.9619244943506, 4628.7974616871106,
        #     4629.8682792805203, 4637.0845615021908, 4658.0776869396414, 4609.6594171963898, 4672.197914151111
        # ]
        # self.assertEqual(len(trace), len(expected_trace))
        # for x, y in zip(trace, expected_trace):
        #     self.assertAlmostEqual(x, y, places=6)