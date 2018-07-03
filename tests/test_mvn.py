from unittest import TestCase
import numpy as np
from scipy.stats import multivariate_normal

from bss.mvn import Mvn


class MvnTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_constructors(self):
        # Multivariate distibutions can be constructed by specifying either the mean vector or the covariance matrix.
        try:
            Mvn(mean=[0, 0])
        except:
            self.fail("Unable to construct distribution with just a mean vector.")

        try:
            Mvn(cov=[[1, 0], [0, 1]])
        except:
            self.fail("Unable to construct distribution with a 2D covariance matrix")

        # We can use numpy arrays too
        try:
            Mvn(mean=np.random.rand(5))
        except:
            self.fail("Unable to construct distribution with a random numpy mean vector")

        try:
            Mvn(cov=np.eye(3))
        except:
            self.fail("Unable to construct distribution with a random numpy covariance matrix")

    def test_bad_constructors(self):
        # Covariance matrix not square
        self.assertRaises(Exception, Mvn, cov=np.eye(3, 2))
        # Covariance matrix not symmetric
        self.assertRaises(Exception, Mvn, cov=np.random.rand(3, 3))
        # Incompatible mean and variance
        self.assertRaises(Exception, Mvn, mean=[1, 2], cov=np.random.rand(3, 3))

    def test_logpdf(self):
        """
        Check our logpdf values against scipy multivariate_normal values for the same inputs
        """
        dims = 10
        x = np.random.rand(dims)
        logpdf = multivariate_normal([0]*dims).logpdf(x)

        dist = Mvn(mean=np.zeros(dims))
        result = dist.logpdf(x)

        self.assertAlmostEqual(logpdf, result)

    def test_correlate(self):
        """
        Generate some correlated samples using the Mvn class.
        """
        dims = 3
        samples = 10000
        cov = np.array([
            [3.412,  2.143, 2.531],
            [2.143, 5.321,  0.531],
            [2.531,  0.531,  2.316]
        ])

        uncorrelated_data = np.random.multivariate_normal(mean=[0]*dims, cov=np.eye(dims), size=samples)
        mvn = Mvn(cov=cov)
        correlated_data = mvn.correlate(uncorrelated_data)

        cov2 = np.cov(correlated_data, rowvar=False)
        self.assertTrue(np.allclose(cov, cov2, atol=0.1))

    def test_decorrelate(self):
        """
        Generate some correlated samples, decorrelate them using the whitening transform in the Mvn class, and check
        to make sure the correlation between the decorrelated variables is indeed the identity matrix.
        """
        dims = 3
        samples = 10000
        cov = np.array([
            [3.412,  2.143, 2.531],
            [2.143, 5.321,  0.531],
            [2.531,  0.531,  2.316]
        ])

        data = np.random.multivariate_normal(mean=[0]*dims, cov=cov, size=samples)

        mvn = Mvn(cov=cov)
        decorrelated_data = mvn.whiten(data).T

        cov = np.cov(decorrelated_data, rowvar=False)
        self.assertTrue(np.allclose(cov, np.eye(dims), atol=0.1))
