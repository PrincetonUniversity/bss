import logging
import itertools
import numpy as np
import numpy.random as npr


logger = logging.getLogger(__name__)


class EllipticalSliceSampler:
    """
    Elliptical Slice Sampling algorithm as outlined in :cite:`Murray2010`.

    Parameters
    ----------
    normal_dist : object
        Object representing the normal distribution that current_state is sampled from. This object needs to be able to
        supply a d-dimensional variate through an rvs() method.
    log_like_fn : callable
        Log Likelihood function that takes in a single argument, a d-dimensional vector representing
        a state, and returns a scalar log-likelihood value
    """
    def __init__(self, normal_dist, log_like_fn):
        self.normal_dist = normal_dist
        self.log_like_fn = log_like_fn

    def start(self, x0=None):
        """
        Start the sampling process at a given point defined by x0

        Parameters
        ----------
        x0 : ndarray, optional
            The starting point of the sampling process, a dx1 vector

        Returns
        -------
        iterable
            A (never-ending) iterable of dx1 samples from slice-sampling
        """
        x0 = x0 if x0 is not None else self.normal_dist.rvs()
        return _EllipticalSliceSamplerIterator(x0, self.normal_dist, self.log_like_fn)

    def chain(self, x0=None, iters=100000, burn_in=50000):
        """
        Generate a fixed number of samples, for a given burn_in and no. of iterations

        Parameters
        ----------
        x0 : ndarray, optional
            The starting point of the sampling process, a dx1 vector
        iters: int, optional
            The no. of iterations of sampling to run, including the burn-in, if any.
        burn_in: int, optional
            The no. of samples to discard as burn-in samples.

        Returns
        -------
        list
            A list of iters-burn_in samples, each a dx1 numpy vector
        """
        return list(itertools.islice(self.start(x0), burn_in, iters))

    def one(self, x0=None):
        """
        Generate a single sample, given the starting point of sampling.

        Parameters
        ----------
        x0 : ndarray, optional
            The starting point of the sampling process, a dx1 vector

        Returns
        -------
        ndarray
            A single sample, a dx1 vector
        """
        return next(self.start(x0))


class _EllipticalSliceSamplerIterator:

    def __init__(self, x0, normal_dist, log_like_fn):
        self.current_state = x0
        self.normal_dist = normal_dist
        self.log_like_fn = log_like_fn

    def __iter__(self):
        return self

    def __next__(self):
        current_state = self.current_state
        nu = self.normal_dist.rvs()
        threshold = np.log(npr.rand()) + self.log_like_fn(current_state)

        phi = npr.rand() * 2 * np.pi
        phi_max = phi
        phi_min = phi_max - 2 * np.pi

        while True:
            new_state = current_state * np.cos(phi) + nu * np.sin(phi)
            if self.log_like_fn(new_state) > threshold:
                self.current_state = new_state
                return self.current_state

            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise Exception("Shrank to zero!")

            phi = npr.rand() * (phi_max - phi_min) + phi_min
