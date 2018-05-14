import numbers
import numpy as np
import numpy.random as npr


class SliceSampler(object):
    """
    Exponential-Expansion slice sampling as per :cite:`Neal2003`

    :param x0: The current sample, scalar or d-dimensional
    :param logprob: A function taking a single sample of the same shape as x0,
        and returning the log-probability of that sample
    :param w: estimate of the typical size of a slice
    :param expand: Whether to expand the slice size, either by linear increments or a doubling process
    :param max_steps_out: The max. no. of expansion steps to take, either in a single direction (when stepping out in
        fixed-width increments), or the total max steps in both directions (when stepping out using doubling steps)
    :param compwise: TODO
    :param doubling_step: Whether to use the doubling procedure described to expand the slice size; Useful to expand
        intervals faster than stepping out in fixed-width increments
    :return:
    """

    def __init__(self, logprob, w=1.0, expand=True, max_steps_out=1000, compwise=True, doubling_step=True):
        self.logprob = logprob
        self.w = w
        self.expand = expand
        self.max_steps_out = max_steps_out
        self.compwise = compwise
        self.doubling_step = doubling_step

    def start(self, x0):
        return SliceSamplerIterator(x0, self.logprob, self.w, self.expand, self.max_steps_out, self.compwise, self.doubling_step)


class SliceSamplerIterator(object):

    def __init__(self, x0, logprob, w=1.0, expand=True, max_steps_out=1000, compwise=True, doubling_step=True):

        self.logprob = logprob
        self.w = w
        self.expand = expand
        self.max_steps_out = max_steps_out
        self.compwise = compwise
        self.doubling_step = doubling_step

        self.scalar = isinstance(x0, (numbers.Number, np.number))
        if self.scalar:
            self.x0 = np.array([x0])
        else:
            self.x0 = x0

        self.dims = self.x0.shape[0]

    def __iter__(self):
        return self

    def __next__(self):

        if self.compwise:
            new_x = self.x0.copy()
            directions = np.eye(self.dims)
            npr.shuffle(directions)
            for direction in directions:
                new_x = self.direction_slice(new_x, direction)
        else:
            direction = npr.randn(self.dims)
            direction = direction / np.sqrt(np.sum(direction ** 2))
            new_x = self.direction_slice(self.x0, direction)

        # Update current position for the next iteration, and return the new position we're at
        self.x0 = new_x
        if self.scalar:
            return float(new_x[0])
        else:
            return new_x

    def dir_logprob(self, z, direction):
        """
        Log-probability value from a sample that is |z| distance away from the d-dimensional point x0, in the
        direction specified by the axis-aligned 'direction' (a vector with exactly one 1 and d-1 0s)
        """
        return self.logprob(self.x0 + direction * z)

    def acceptable(self, x0, x1, y, L, R, direction):
        """
        Test for whether a new point x1 is an acceptable next state, when the interval L-R was found by the
        doubling procedure. This additional check is needed only if doubling steps were performed during the
        slice expansion phase.

        Note that x1 is acceptable only if it could also have produced the interval L-R. This entails additional
        checks in the case when the current point x0 and the new point x1 are on different 'sides' of
        the L-R interval. This is tracked through the variable D below. If x0 and x1 are on the same side of
        the middle, then we only consider that half the interval and check again, and repeat the process till we
        reach our original slice width (w)
        """
        while (R - L) > 1.1 * self.w:  # A factor of 1.1 guards against possible round-off error
            middle = (L + R) / 2.

            # Whether x0 and x1 are on different halves of the L-R interval
            D = (x0 < middle <= x1) or (x1 < middle <= x0)

            # Reduce our interval by half, retaining the half that has x1
            if x1 < middle:
                R = middle
            else:
                L = middle

            # If x0 and x1 were on different sides of the middle, then the original L-R interval could have
            # been produced by doubling from an interval containing x1 ONLY IF both ends were not off-slice.
            if D and y >= self.dir_logprob(L, direction) and y >= self.dir_logprob(R, direction):
                return False

        # x1 was not rejected in the loop above. It's acceptable.
        return True

    def direction_slice(self, x0, direction):
        # Note that in our implementation, the "current value" (x0 in Neal2003) is implicitly 0.
        # since we simply add x0 to x1 at the very end, in the direction specified by the vector 'direction'

        # ----------------------------------
        # 1. Finding an appropriate interval
        # ----------------------------------

        # Randomly position the initial interval of width w around 0
        R = self.w * npr.rand()
        L = R - self.w

        # Log-Likelihood value at x0. TODO: Why the additional value??
        y = self.logprob(x0) + np.log(npr.rand())

        # The no. of "step-out" steps we've taken in the lower and upper directions
        L_steps_out = R_steps_out = 0

        if self.expand:
            if self.doubling_step:
                # Produce a sequence of intervals, each twice the size of the previous one, until an interval is found
                # with both ends outside the slice, or a predetermined limit on the no. of step-outs has been reached.
                while (L_steps_out + R_steps_out) < self.max_steps_out and (self.dir_logprob(L, direction) > y or self.dir_logprob(R, direction) > y):
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
                while self.dir_logprob(L, direction) > y and L_steps_out < self.max_steps_out:
                    L_steps_out += 1
                    L -= self.w
                while self.dir_logprob(R, direction) > y and R_steps_out < self.max_steps_out:
                    R_steps_out += 1
                    R += self.w

        # ----------------------------------------------------------
        # 2. Sampling from the part of the slice within the interval
        # ----------------------------------------------------------

        # Repeatedly sample uniformly from an interval that is initially equal to R-L, and which shrinks each time
        # a point is drawn that is not acceptable.
        steps_in = 0
        while True:
            steps_in += 1
            x1 = L + npr.rand() * (R - L)
            new_y = self.dir_logprob(x1, direction)

            # TODO: Should the additional 'acceptable' check be performed only if doubling_step = True ?
            if new_y > y and self.acceptable(0, x1, y, L, R, direction):
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

