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

# RPA TODO:
#  1) Speed up the code, perhaps with more memoization.
#  2) Include eigenvalue scaling for the covariance matrix.
#  3) Perform Geweke validation of the code.
#  4) Evaluate mixing using CODA and friends.

class ProbitSS:

    def __init__(self, X, Y, R):
        self.X  = X
        self.Y  = Y
        self.R  = bend_corr(R)

        # Probit threshold prior -- FIXME
        target_sparsity = 0.01
        self.gamma0_m   = sps.norm.ppf(1.0 - target_sparsity)
        self.gamma0_v   = 1.0

        # Weight/bias scale prior -- FIXME
        self.lamb_a = 1e-6
        self.lamb_b = 1e-6

        # Observation noise prior -- FIXME
        self.nu_a = 1e-6
        self.nu_b = 1e-6

        # Store off some dimensions.
        self.num_data = self.Y.shape[0]
        self.num_snps = self.X.shape[1]

        # Initialize the probit threshold.
        self.gamma0 = self.gamma0_m
        
        # Initialize the observation precision.
        self.nu = self.nu_a / self.nu_b

        # Initialize the global inverse-scale parameter.
        self.lamb = self.lamb_a / self.lamb_b

        # Precompute the expensive part.
        # Produce lower triangular cR such that R = np.dot(cR, cR.T).
        self.cR = spla.cholesky(self.R, lower=True, check_finite=False)

        # Initialize the sparsity function.
        self.gamma = np.dot(self.cR, npr.randn(self.num_snps))

    def run_mcmc(self, iters=1000, burnin=100):

        logpost_trace   = np.zeros(iters)
        inclusion_trace = np.zeros((iters, self.num_snps), dtype=bool)
        for iter in range(-burnin, iters):
            log_post = self.log_joint()
            
            sys.stderr.write('%05d / %05d] logprob: %f\n' % (iter, iters, log_post))

            self.update_gamma()
            self.update_gamma0()
            self.update_lambda()
            self.update_nu()

            if iter >= 0:
                # Record things.
                inclusion_trace[iter,:] = self.gamma > self.gamma0
                logpost_trace[iter] = log_post

        return np.mean(inclusion_trace, 0), inclusion_trace[np.argmax(logpost_trace),:]

    def update_gamma(self):
        
        slice_fn = lambda gamma: self.log_marg_like(gamma,
                                                    self.gamma0,
                                                    self.nu,
                                                    self.lamb)

        self.gamma = elliptical_slice(self.gamma, self.cR, slice_fn)

    def update_nu(self):
        mask     = self.gamma > self.gamma0
        mX       = self.X[:,mask]
        psi      = (np.dot(mX, mX.T) + np.ones((self.num_data, self.num_data)))/self.lamb + np.eye(self.num_data)
        chol_psi = spla.cholesky(psi, lower=True, check_finite=False)
        solve    = spla.solve_triangular(chol_psi, self.Y, lower=True, trans=0, check_finite=False)
        inner    = np.dot(solve.T, solve)

        post_a = self.nu_a + 0.5*self.num_data
        post_b = self.nu_b + 0.5*inner

        self.nu = npr.gamma(post_a, 1/post_b)

    def update_lambda(self):

        def slice_fn(lamb):
            if lamb < 0:
                return -np.inf

            return (self.log_marg_like(self.gamma,
                                       self.gamma0,
                                       self.nu,
                                       lamb)
                    + self.log_lambda_prior(lamb))

        self.lamb = slice_sample(self.lamb, slice_fn)

    def update_gamma0(self):
        slice_fn = lambda gamma0: (self.log_marg_like(self.gamma,
                                                      gamma0,
                                                      self.nu,
                                                      self.lamb)
                                   + self.log_gamma0_prior(gamma0))

        self.gamma0 = slice_sample(self.gamma0, slice_fn, verbose=False)

    def log_marg_like(self, gamma, gamma0, nu, lamb):
        mask     = gamma > gamma0
        mX       = self.X[:,mask]
        covar    = ((np.dot(mX, mX.T) + np.ones((self.num_data, self.num_data)))/lamb + np.eye(self.num_data))/nu
        chol_cov = spla.cholesky(covar, lower=True, check_finite=False)
        solve    = spla.solve_triangular(chol_cov, self.Y, lower=True, trans=0, check_finite=False)
        lml      = -( 0.5*self.num_data*np.log(2*np.pi)
                      + np.sum(np.log(np.diag(chol_cov)))
                      + 0.5*np.dot(solve.T, solve))
        return lml

    def log_gamma0_prior(self, gamma0):
        return -0.5*(np.log(2*np.pi) 
                     + np.log(self.gamma0_v) 
                     + ((gamma0-self.gamma0_m)**2)/self.gamma0_v)

    def log_lambda_prior(self, lamb):
        return gammapdfln(lamb, self.lamb_a, self.lamb_b)

    def log_nu_prior(self, nu):
        return gammapdfln(nu, self.nu_a, self.nu_b)

    def log_gamma_prior(self, gamma):
        solve = spla.solve_triangular(self.cR, gamma, lower=True, trans=0, check_finite=False)
        return -( 0.5*self.num_snps*np.log(2*np.pi)
                  + np.sum(np.log(np.diag(self.cR)))
                  + 0.5*np.dot(solve.T, solve))

    def log_joint(self):

        log_marg_like  = self.log_marg_like(self.gamma, self.gamma0, self.nu, self.lamb)
        log_gam0_prior = self.log_gamma0_prior(self.gamma0)
        log_nu_prior   = self.log_nu_prior(self.nu)
        log_lamb_prior = self.log_lambda_prior(self.lamb)
        log_gam_prior  = self.log_gamma_prior(self.gamma)

        return log_marg_like + log_gam0_prior + log_nu_prior + log_lamb_prior + log_gam_prior
        
def probit_inclusion_probs( xy_file=None, cor_file=None, iters=1000, burnin=500):
    if xy_file is None:
        sys.stderr.write("Must specify xy file.\n")
        sys.exit(-1)

    if cor_file is None:
        sys.stderr.write("Must specify cor file.\n")
        sys.exit(-1)

    (genos, phenos, eqtls) = load_xy_file(xy_file)
    corr = load_cor_file(cor_file)

    model = ProbitSS(genos, phenos, bend_corr(corr))

    return model.run_mcmc(burnin, iters)

if __name__ == '__main__':
    npr.seed(1)

    data_dir = 'data/small_eQTL'
    prefix   = 'sim96'
    sys.stderr.write("Performing test with file %s.\n" % (prefix))
    
    (genos, phenos, eqtls, corr) = load_data(data_dir, prefix)

    model = ProbitSS(genos, phenos, bend_corr(corr))

    t_start = time.time()
    inclusion_probs = model.run_mcmc(burnin=0, iters=100)
    t_end = time.time()
    print(t_end-t_start)

    # print(np.vstack([inclusion_probs, eqtls]).T)

    #num_iters = 500
    #log_joint_trace = np.zeros(num_iters)
    #for ii in range(num_iters):
    #    log_joint_trace[ii] = model.log_joint()
    #    print(ii, log_joint_trace[ii], model.gamma0, model.lamb, 1/model.nu, np.sum(model.gamma > model.gamma0), np.sum(eqtls == (model.gamma > model.gamma0))
    #    model.update_gamma()
    #    model.update_gamma0()
    #    model.update_lambda()
    #    model.update_nu()

    #pl.plot(log_joint_trace)
    #pl.show()
