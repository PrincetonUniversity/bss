"""
Sampling functions used to update parameters for the Probit model.
"""

import numpy as np
import numpy.random as npr


def elliptical_slice_sample(current_state, normal_dist, log_like_fn):
    """
    Elliptical Slice Sampling algorithm as outlined in :cite:`Murray2010`.

    :param current_state: A d-dimensional vector of latent variables, representing the current state,
     drawn from a normal distribution represented by normal_dist
    :param normal_dist: Object representing the normal distribution that current_state is sampled from.
     This object needs to be able to supply a d-dimensional variate through an rvs() method.
     The Mvn class provided in the bss module works for this purpose.
    :param log_like_fn: Log Likelihood function that takes in a single argument, a d-dimensional vector representing
     a state, and returns a scalar log-likelihood value
    :return: The new state (d-dimensional) after the sampling step
    """
    nu = normal_dist.rvs()
    threshold = np.log(npr.rand()) + log_like_fn(current_state)

    phi = npr.rand() * 2*np.pi
    phi_max = phi
    phi_min = phi_max - 2*np.pi

    while True:
        new_state = current_state * np.cos(phi) + nu * np.sin(phi)
        if log_like_fn(new_state) > threshold:
            return new_state

        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception("Shrank to zero!")

        phi = npr.rand() * (phi_max - phi_min) + phi_min


def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000,
                 compwise=True, doubling_step=True, verbose=False):
    """
    Exponential-Expansion slice sampling as per :cite:`Neal2003`
    TODO: Complete writeup

    :param init_x:
    :param logprob:
    :param sigma:
    :param step_out:
    :param max_steps_out:
    :param compwise:
    :param doubling_step:
    :param verbose:
    :return:
    """
    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction * z + init_x)

        def acceptable(z, llh_s, L, U):
            while (U - L) > 1.1 * sigma:
                middle = 0.5 * (L + U)
                splits = (0 < middle <= z) or (z < middle <= 0)
                if z < middle:
                    U = middle
                else:
                    L = middle
                # Probably these could be cached from the stepping out.
                if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(L):
                    return False
            return True

        upper = sigma * npr.rand()
        lower = upper - sigma
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            if doubling_step:
                while (dir_logprob(lower) > llh_s or dir_logprob(upper) > llh_s) and (
                        l_steps_out + u_steps_out) < max_steps_out:
                    if npr.rand() < 0.5:
                        l_steps_out += 1
                        lower -= (upper - lower)
                    else:
                        u_steps_out += 1
                        upper += (upper - lower)
            else:
                while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                    l_steps_out += 1
                    lower -= sigma
                while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                    u_steps_out += 1
                    upper += sigma

        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * npr.rand() + lower
            new_llh = dir_logprob(new_z)
            if np.isnan(new_llh):
                print(new_z, direction * new_z + init_x, new_llh, llh_s, init_x, logprob(init_x))
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s and acceptable(new_z, llh_s, start_lower, start_upper):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print("Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in)

        return new_z * direction + init_x

    if type(init_x) == float or isinstance(init_x, np.number):
        init_x = np.array([init_x])
        scalar = True
    else:
        scalar = False

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        new_x = init_x.copy()
        for d in ordering:
            direction = np.zeros(dims)
            direction[d] = 1.0
            new_x = direction_slice(direction, new_x)

    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction ** 2))
        new_x = direction_slice(direction, init_x)

    if scalar:
        return float(new_x[0])
    else:
        return new_x
