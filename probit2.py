import sys
import time
import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats  as sps
import pylab        as pl

from data import *
from mcmc import *
from util import *

class ProbitSS:

    def __init__(self, X, Y, R, 
                 target_sparsity=0.01, gamma0_v=1.0,
                 lamb_a=1e-6, lamb_b=1e-6,
                 nu_a=1e-6, nu_b=1e-6,
                 xi_a=1.0, xi_b=1.0,                 
                 check_finite=True,
                 sample_xi=True,
                 min_eval=0, jitter=1e-6):
        """
        Constructor for ProbitSS class.

        Parameters
        ----------
        X : explanatory variables (NxP matrix)
        Y : dependent variables (N-vector)
        R : correlations (PxP symmetric matrix)

        target_sparsity : prior mean of sparsity (used to compute gamma0_m, mean of probit threshold prior)
        gamma0_v : variance of probit threshold prior

        lamb_a : global weight inverse-scale prior, gamma shape parameter
        lamb_b : global weight inverse-scale prior, gamma inverse-scale parameter

        nu_a : observation precision prior, gamma shape parameter
        nu_b : observation precision prior, gamma inverse-scale parameter

        xi_a : prior on correlation weakening, beta parameter
        xi_b : prior on correlation weakening, beta parameter

        check_finite : perform sanity checks before linear algebra (bool)
        min_eval : minimum allowed eigenvalue in correlation matrix (float)
        jitter : added to diagonal of covariance to ensure positive-definiteness
        
        """

        self.X = X
        self.Y = Y
        self.R = R # Absolute value?

        self.N = self.X.shape[0]
        self.P = self.X.shape[1]

        self.gamma0_m = sps.norm.ppf(1.0 - target_sparsity)
        self.gamma0_v = gamma0_v
        self.lamb_a   = lamb_a
        self.lamb_b   = lamb_b
        self.nu_a     = nu_a
        self.nu_b     = nu_b
        self.xi_a     = xi_a
        self.xi_b     = xi_b

        self.sample_xi    = sample_xi
        self.check_finite = check_finite

        # Initialize scalar model parameters to their prior means.
        self.gamma0 = self.gamma0_m
        self.lamb   = self.lamb_a/self.lamb_b
        self.nu     = self.nu_a/self.nu_b

        if self.sample_xi:
            self.xi = self.xi_a/(self.xi_a + self.xi_b)
        else:
            self.xi = 1.0-1e-6

        # Ensure that the base correlation matrix is positive definite.
        # This also guarantees that there are ones along the diagonal.
        self.bend_matrix(min_eval, jitter)

        # Initialize the sparsity function.
        self.gamma = np.dot(self.get_cholesky(), npr.randn(self.P))

    def bend_matrix(self, min_eval, jitter):
        """Make the correlation matrix positive definite and normalized."""

        # Ensure that no eigenvalues are too small.
        # spla.eigh returns vector D and matrix V such that R = V*diag(D)*V'.
        (evals, evecs) = spla.eigh(self.R, check_finite=self.check_finite)
        evals[evals < min_eval] = min_eval
        self.R = np.dot(evecs, np.dot(np.diag(evals), evecs.T))

        # Add jitter and renormalize.
        self.R = self.R + jitter*np.eye(self.R.shape[0])
        diagR  = np.diag(self.R)[:,np.newaxis]
        self.R = self.R / np.sqrt(diagR * diagR.T)

    def xi_adjusted(self, xi):
        """
        Compute the convex sum between R and the identity.
        """
        return (1.0-xi)*np.eye(self.P) + xi*self.R

    def get_cholesky(self):
        """
        Compute (or return the pre-computed) lower-triangular Cholesky of the 
        elementwise-exponentiated correlation matrix.
        """
        try:
            if self._last_xi != self.xi:
                self._last_xi = self.xi
                self._chol = spla.cholesky(self.xi_adjusted(self.xi), lower=True, check_finite=self.check_finite)
            return self._chol
        except AttributeError:
            # If _last_xi doesn't exist, make it exist and try again.
            # This is just an obscure way to make the caching logic local.
            self._last_xi = -1
            return self.get_cholesky()

    def log_marg_like(self, gamma, gamma0, nu, lamb):
        """
        Compute (or return the pre-computed) log marginal likelihood.
        """

        try:
            mask = gamma > gamma0

            # If everything is the same, just return immediately.
            if np.array_equal(mask, self._last_mask) and lamb == self._last_lamb and nu == self.last_nu:
                return self._log_marg_like

            # If the mask is the same, we may be able to save some time.
            chol_cov = None
            if np.array_equal(mask, self._last_mask):

                if lamb == self._last_lamb:
                    # If lambda is also the same, we can save the cholesky.
                    chol_cov = self._last_chol_cov
                else:
                    # Otherwise, we can still save the masking of the input.
                    masked_X = self._last_masked_X
            else:
                masked_X = self.X[:,mask]
                self._last_masked_X = masked_X

            if chol_cov is None:
                covar    = (np.dot(masked_X, masked_X.T) + np.ones((self.N, self.N)))/lamb + np.eye(self.N)
                chol_cov = spla.cholesky(covar, lower=True, check_finite=self.check_finite)
                self._last_chol_cov = chol_cov

            solve = spla.solve_triangular(chol_cov/np.sqrt(nu), self.Y, lower=True, trans=0, check_finite=self.check_finite)
                
            self._log_marg_like = -0.5*self.N*np.log(2*np.pi/nu) - np.sum(np.log(np.diag(chol_cov))) - 0.5*np.dot(solve.T, solve)
            self._last_chol_cov = chol_cov
            self._last_mask     = mask
            self._last_nu       = nu
            self._last_lamb     = lamb

            return self._log_marg_like

        except AttributeError:
            self._last_mask = np.zeros(gamma.shape) + np.nan
            self._last_nu   = -1
            self._last_lamb = -1
            return self.log_marg_like(gamma, gamma0, nu, lamb)

    def log_joint(self):

        log_marg_like  = self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb)
        log_gam0_prior = self.log_gamma0_prior(self.gamma0)
        log_nu_prior   = self.log_nu_prior(self.nu)
        log_lamb_prior = self.log_lambda_prior(self.lamb)
        log_gam_prior  = self.log_gamma_prior(self.gamma)

        if self.sample_xi:
            log_xi_prior   = self.log_xi_prior(self.xi)
        else:
            log_xi_prior = 0.0

        #print log_marg_like, log_gam0_prior, log_nu_prior, log_lamb_prior, log_gam_prior, log_xi_prior

        return log_marg_like + log_gam0_prior + log_nu_prior + log_lamb_prior + log_gam_prior + log_xi_prior

    def run_mcmc(self, iters=1000, burnin=100, post_trace=False):
        
        logpost_trace   = np.zeros(iters)
        inclusion_trace = np.zeros((iters, self.P), dtype=bool)

        for iter in xrange(-burnin, iters):

            log_post = self.log_joint()

            sys.stderr.write('%05d / %05d] logprob: %f [gamma0:%f lambda:%f nu:%f xi:%f\n' % (iter, iters, log_post, self.gamma0, self.lamb, self.nu, self.xi))

            self.update_gamma()
            self.update_gamma0()
            self.update_lambda()
            self.update_nu()
            if self.sample_xi:
                self.update_xi()

            if iter >= 0:
                # Record things.
                inclusion_trace[iter,:] = self.gamma > self.gamma0
                logpost_trace[iter] = log_post

        if post_trace:
            return logpost_trace
        else:
            return np.mean(inclusion_trace, 0), inclusion_trace[np.argmax(logpost_trace),:]

    def run_geweke(self, iters=1000, burnin=100):
        gamma0_trace  = np.zeros(iters)
        lambda_trace  = np.zeros(iters)
        nu_trace      = np.zeros(iters)
        xi_trace      = np.zeros(iters)
        logpost_trace = np.zeros(iters)
        loglike_trace = np.zeros(iters)

        for iter in xrange(-burnin, iters):
            log_post = self.log_joint()
            log_like = self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb)

            sys.stderr.write('%05d / %05d] logpost: %f  loglike: %f\n' % (iter, iters, log_post, log_like))

            self.update_gamma()
            self.update_gamma0()
            self.update_lambda()
            self.update_nu()
            self.update_xi()
            self.update_data()

            if iter >= 0:
                gamma0_trace[iter]  = self.gamma0
                lambda_trace[iter]  = self.lamb
                nu_trace[iter]      = self.nu
                xi_trace[iter]      = self.xi
                logpost_trace[iter] = log_post
                loglike_trace[iter] = log_like

        import pylab as pl
        pl.figure(1)
        pl.subplot(2,2,1)
        pl.hist(gamma0_trace, 25, normed=1)
        gx = np.linspace(-5, 5, 1000)
        pl.plot(gx, np.exp(self.log_gamma0_prior(gx)))
        pl.title('gamma0')

        pl.subplot(2,2,2)
        pl.hist(lambda_trace, 25, normed=1)
        gx = np.linspace(0, 10, 1000)
        pl.plot(gx, np.exp(self.log_lambda_prior(gx)))
        pl.title('lambda')

        pl.subplot(2,2,3)
        pl.hist(nu_trace, 25, normed=1)
        gx = np.linspace(0, 10, 1000)
        pl.plot(gx, np.exp(self.log_nu_prior(gx)))
        pl.title('nu')

        pl.subplot(2,2,4)
        pl.hist(xi_trace, 25, normed=1)
        gx = np.linspace(0, 1, 1000)
        pl.plot(gx, np.exp(self.log_xi_prior(gx)))
        pl.title('xi')

        pl.figure(2)
        pl.subplot(2,1,1)
        pl.plot(logpost_trace)
        pl.title('log posterior')

        pl.subplot(2,1,2)
        pl.plot(loglike_trace)
        pl.title('log marginal likelihood')

        pl.show()

    def update_gamma(self):
        """
        Apply MCMC transition operator to the function gamma.
        This is performed with an iteration of elliptical slice sampling.
        """
        
        # Construct the log likelihood for elliptical slice sampling.
        slice_fn = lambda gamma: self.log_marg_like(gamma, self.gamma0, self.nu, self.lamb)

        self.gamma = elliptical_slice(self.gamma, self.get_cholesky(), slice_fn)

    def update_nu(self):
        """
        Apply MCMC transition operator to the precision parameter.
        """

        # Take advantage of the marginal likelihood caching.
        self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb)

        solve = spla.solve_triangular(self._last_chol_cov, self.Y, lower=True, trans=0, check_finite=self.check_finite)
        inner = np.dot(solve.T, solve)

        post_a = self.nu_a + 0.5*self.N
        post_b = self.nu_b + 0.5*inner

        self.nu = npr.gamma(post_a, 1/post_b)

    def update_lambda(self):
        """
        Apply MCMC transition operator to the global weight inverse-scale parameter.
        """
        
        def slice_fn(lamb):
            if lamb < 0:
                return -np.inf

            return (self.log_marg_like(self.gamma, self.gamma0, self.nu, lamb)
                    + self.log_lambda_prior(lamb))

        self.lamb = slice_sample(self.lamb, slice_fn, verbose=False)

    def update_gamma0(self):
        """
        Apply MCMC transition operator to the sparsity threshold.
        """

        slice_fn = lambda gamma0: (self.log_marg_like(self.gamma, gamma0, self.nu, self.lamb)
                                   + self.log_gamma0_prior(gamma0))

        self.gamma0 = slice_sample(self.gamma0, slice_fn, step_out=True)

    def update_xi(self):
        """
        Apply MCMC transition operator to the correlation exponent.
        """

        if True:
            # Compute the latent whitened variables.
            whitened = spla.solve_triangular(self.get_cholesky(), self.gamma, lower=True, trans=0, check_finite=self.check_finite)

            # Construct the slice sampling function.
            def slice_fn(xi):
                if xi <= 0 or xi >= 1:
                    return -np.inf

                try:
                    chol_cov = spla.cholesky(self.xi_adjusted(xi), lower=True, check_finite=self.check_finite)
                except np.linalg.linalg.LinAlgError as err:
                    return -np.inf

                gamma = np.dot(chol_cov, whitened)

                return (self.log_marg_like(gamma, self.gamma0, self.nu, self.lamb) + self.log_xi_prior(xi))            

            self.xi    = slice_sample(self.xi, slice_fn)
            self.gamma = np.dot(self.get_cholesky(), whitened)

        else:
        # This is the slow way.  Murray and Adams, NIPS 2010 is better.
            def slice_fn(xi):
                if xi <= 0 or xi >= 1:
                    return -np.inf
                try:
                    covmat = self.xi_adjusted(xi)
                    chol_cov = spla.cholesky(covmat, lower=True, check_finite=self.check_finite)
                    solve = spla.solve_triangular(chol_cov, self.gamma, lower=True, trans=0, check_finite=self.check_finite)
                    return (-0.5*self.P*np.log(2*np.pi) - np.sum(np.log(np.diag(chol_cov))) - 0.5*np.dot(solve.T, solve)
                             + self.log_xi_prior(xi))
                except np.linalg.linalg.LinAlgError as err:
                    return -np.inf

            self.xi = uni_slice_sample(self.xi, slice_fn, 0, 1)

    def update_data(self):
        """
        Apply MCMC transition operator to the dependent variables.
        This is only useful for Geweke-style validation.
        """
        # We can use intermediate results from the marginal likelihood.
        self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb)
        chol_cov = self._last_chol_cov/np.sqrt(self.nu)
        self.Y   = np.dot(chol_cov, npr.randn(self.N))

    def log_lambda_prior(self, lamb):
        """
        Log density of gamma prior on lambda.
        """
        return gammapdfln(lamb, self.lamb_a, self.lamb_b)

    def log_nu_prior(self, nu):
        """
        Log density of gamma prior on nu.
        """
        return gammapdfln(nu, self.nu_a, self.nu_b)

    def log_gamma0_prior(self, gamma0):
        """
        Log density of Gaussian prior on sparsity threshold.
        """
        return -0.5*( np.log(2*np.pi) + np.log(self.gamma0_v) + ((gamma0-self.gamma0_m)**2)/self.gamma0_v)

    def log_gamma_prior(self, gamma):
        """
        Log density of Gaussian process prior for gamma.
        """
        solve = spla.solve_triangular(self.get_cholesky(), gamma, lower=True, trans=0, check_finite=self.check_finite)
        return -0.5*self.P*np.log(2*np.pi) - np.sum(np.log(np.diag(self.get_cholesky()))) - 0.5*np.dot(solve.T, solve)

    def log_xi_prior(self, xi):
        """
        Log density of beta prior on xi.
        """
        return betapdfln(xi, self.xi_a, self.xi_b)

if __name__ == '__main__':
    npr.seed(1)

    data_dir = '../../data/small_eQTL'
    prefix   = 'sim96'
    sys.stderr.write("Performing test with file %s.\n" % (prefix))

    (genos, phenos, eqtls, corr) = load_data(data_dir, prefix)

    model = ProbitSS(genos, phenos, corr, check_finite=False,
                     sample_xi=False,
                     target_sparsity=0.01, gamma0_v=1.0,
                     lamb_a=1e-6, lamb_b=1e-6,
                     nu_a=1e-6, nu_b=1e-6,
                     xi_a=1.0, xi_b=1.0)

    t_start = time.time()
    logpost_trace = model.run_mcmc(burnin=500, iters=5000, post_trace=True)
    t_end = time.time()
    print t_end-t_start

    pl.subplot(2,1,1)
    pl.plot(acf(logpost_trace, 200))
    pl.grid()

    pl.subplot(2,1,2)
    pl.plot(logpost_trace)
    pl.show()

    #model.run_geweke(burnin=500, iters=10000)

    #inclusion_probs = model.run_mcmc(burnin=0, iters=100)
    #print np.vstack([inclusion_probs, eqtls]).T
