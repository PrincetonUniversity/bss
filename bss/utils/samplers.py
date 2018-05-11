"""
Sampling functions used to update parameters for the Probit model.
"""
import numbers
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


def slice_sample(x0, logprob, w=1.0, expand=True, max_steps_out=1000, compwise=True, doubling_step=True):
    """
    Exponential-Expansion slice sampling as per :cite:`Neal2003`

    :param x0: The current sample, scalar or d-dimensional
    :param logprob: A function taking a single sample of the same shape as x0,
        and returning the log-probability of that sample
    :param w: estimate of the typical size of a slice
    :param expand: Whether to expand the slice size, either by linear increments or a doubling process
    :param max_steps_out: The max. no. of expansion steps to take, either in a single direction (when stepping out in
        fixed-width increments), or the total max steps in both directions (when stepping out using doubling steps)
    :param compwise:
    :param doubling_step: Whether to use the doubling procedure described to expand the slice size; Useful to expand
        intervals faster than stepping out in fixed-width increments
    :return:
    """
    def direction_slice(x0, direction):

        def dir_logprob(z):
            # Log-probability value from a sample that is |z| distance away from the d-dimensional point x0, in the
            # direction specified by the axis-aligned 'direction' (a vector with exactly one 1 and d-1 0s)
            return logprob(x0 + direction * z)

        def acceptable(x0, x1, y, L, R):
            # Test for whether a new point x1 is an acceptable next state, when the interval L-R was found by the
            # doubling procedure. This additional check is needed only if doubling steps were performed during the
            # slice expansion phase.

            # Note that x1 is acceptable only if it could have produced the interval L-R. This entails additional
            # checks in the case when the current point x0 and the new point x1 are on different 'sides' of
            # the L-R interval. This is tracked through the variable D below. If x0 and x1 are on the same side of
            # the middle, then we halve the interval and check again, and repeat the process till we reach our original
            # slice width (w)

            while (R - L) > 1.1 * w:  # A factor of 1.1 guards against possible round-off error
                middle = (L + R)/2.

                # Whether x0 and x1 are on different halves of the L-R interval
                D = (x0 < middle <= x1) or (x1 < middle <= x0)

                # Reduce our interval by half, retaining the half that has x1
                if x1 < middle:
                    R = middle
                else:
                    L = middle

                # If x0 and x1 were on different sides of the middle, then the original L-R interval could have
                # been produced by doubling from an interval containing x1 ONLY IF both ends were not off-slice.
                if D and y >= dir_logprob(L) and y >= dir_logprob(R):
                    return False

            # x1 was not rejected in the loop above. It's acceptable.
            return True

        # Note that in our implementation, the "current value" (x0 in Neal2003) is implicitly 0,
        # since we find out the TODO: ??

        # ----------------------------------
        # 1. Finding an appropriate interval
        # ----------------------------------

        # Randomly position the initial interval of width w around x0 (0)
        R = w * npr.rand()
        L = R - w

        # Log-Likelihood value at TODO: ??
        y = np.log(npr.rand()) + dir_logprob(0.0)

        # The no. of "step-out" steps we've taken in the lower and upper directions
        L_steps_out = R_steps_out = 0

        if expand:
            if doubling_step:
                # Produce a sequence of intervals, each twice the size of the previous one, until and interval is
                # found with both ends outside the slice, or a predetermined limit on the no. of step-outs has been
                # reached.
                while (L_steps_out + R_steps_out) < max_steps_out and (dir_logprob(L) > y or dir_logprob(R) > y):
                    # Note that the two sides are not expanded equally. Instead just one side is expanded, chosen at
                    # random (irrespective of whether that side is already outside the slice). This is essential to
                    # the correctness of the method, since it produces a final interval that could have been obtained
                    # from points other than the current one. :cite:`Neal2003`
                    if npr.rand() < 0.5:
                        L_steps_out += 1
                        L -= (R - L)
                    else:
                        R_steps_out += 1
                        R += (R - L)

            # Simple linear expansion of slice
            else:
                # As long as we remain under the plot and haven't exceeded the max. no. of steps
                # in either direction, keep expanding the slice by increments of w
                while dir_logprob(L) > y and L_steps_out < max_steps_out:
                    L_steps_out += 1
                    L -= w
                while dir_logprob(R) > y and R_steps_out < max_steps_out:
                    R_steps_out += 1
                    R += w

        # ----------------------------------------------------------
        # 2. Sampling from the part of the slice within the interval
        # ----------------------------------------------------------

        # Repeatedly sample uniformly from an interval that is initially equal to R-L, and which shrinks each time
        # a point is drawn that is not acceptable.
        steps_in = 0
        while True:
            steps_in += 1
            x1 = L + npr.rand()*(R - L)
            new_y = dir_logprob(x1)

            # TODO: Should the additional 'acceptable' check be performed only if doubling_step = True ?
            if new_y > y and acceptable(0, x1, y, L, R):
                break
            elif x1 < 0:
                L = x1
            elif x1 > 0:
                R = x1
            elif np.isnan(new_y):
                raise RuntimeError("Slice sampler got a NaN")
            else:
                raise RuntimeError("Slice sampler shrank to zero!")

        # print("Steps Out:", L_steps_out, R_steps_out, " Steps In:", steps_in)
        return x0 + direction * x1

    scalar = isinstance(x0, (numbers.Number, np.number))
    if scalar:
        x0 = np.array([x0])

    dims = x0.shape[0]
    if compwise:
        ordering = list(range(dims))
        npr.shuffle(ordering)
        new_x = x0.copy()
        for d in ordering:
            direction = np.zeros(dims)
            direction[d] = 1.0
            new_x = direction_slice(new_x, direction)

    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction ** 2))
        new_x = direction_slice(x0, direction)

    if scalar:
        return float(new_x[0])
    else:
        return new_x
