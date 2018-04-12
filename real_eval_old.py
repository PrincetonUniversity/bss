import mkl
import sys
import numpy as np
import scipy as sp
import math
import os
from scipy import stats
import probit
import data

from sklearn.linear_model import ARDRegression, LinearRegression, LassoLars, lasso_path, enet_path

#path = "/Users/engelhardt/work/data/ryan/bayesian-group-sparsity/data/sim/"


def llik(X,y,beta):
    ll = 0
    N = X.shape[0]
    for i in range(N):
        ll = ll + math.log(sp.stats.norm.pdf(y[i] - np.inner(X[i,:],beta)))
    #print(ll)
    return ll

def AIC(X,y,beta):
    count = 0
    for b in beta:
        if not abs(b) < 0.0000001:
            count = count + 1
    return ((-2*llik(X,y,beta)) + (2 * count))

def BIC(X,y,beta):
    count = 0
    N = y.shape[0]
    for b in beta:
        if not abs(b) < 0.0000001:
            count = count + 1
    return ((-2*llik(X,y,beta)) + (count * math.log(N)))


def LassoRegressionSelection(X,y):
    clf = lasso_path(X,y, normalize=False, copy_X=True)
    alphas_lasso = np.array([model.alpha for model in clf])
    coefs_lasso = np.array([model.coef_ for model in clf])
    min_bic = 0
    min_coef = 0
    for coef in coefs_lasso:
        this_bic = BIC(X,y,coef)
        if min_bic == 0 or this_bic < min_bic:
            min_bic = this_bic
            min_coef = coef
    return min_coef

def ForwardStepwiseRegression(X,y):
    b = 0
    not_done = True
    N = y.shape[0]
    P = X.shape[1]
    include = []
    clf = LinearRegression(copy_X=True)
    remaining = range(P)
    while not_done:
        max_b = b
        max_i = 0
        for i in remaining:
            include.append(i)
            clf.fit(X[:,include],y)
            bthis = BIC(X[:,include],y,clf.coef_)
            if max_b == 0 or bthis < max_b:
                max_b = bthis
                max_i = i
            include.pop()
        if max_b == b:
            not_done = False
        else:
            include.append(max_i)
            #print "including "+str(max_i)+" with bic "+str(max_b)
            b = max_b
            remaining.remove(max_i)
    clf.fit(X[:,include],y)
    coefs = [0]*P
    for i in range(len(include)):
        coefs[include[i]] = clf.coef_[i]
    return coefs

def ForwardStepwiseBackwardRegression(X,y, coefs = 0):
    N = y.shape[0]
    P = X.shape[1]
    include = [0]*P
    for i in range(len(coefs)):
        c = coefs[i]
        if abs(c) > 0:
            include[i] = 1
    clf = LinearRegression(copy_X=True)
    clf.fit(X[:,include],y)
    b = BIC(X[:,include],y,clf.coef_)
    not_done = True
    while not_done:
        max_b = b
        max_i = 0
        i = 0
        length = len(include)
        while i < length:
            ind = include[i]
            include.remove(ind)
            clf.fit(X[:,include],y)
            bthis = BIC(X[:,include],y,clf.coef_)
            if bthis < max_b:
                max_b = bthis
                max_i = ind
            include.insert(i,ind)
            i = i+1
        if max_b == b:
            not_done = False
        else:
            include.remove(max_i)
            b = max_b
    clf.fit(X[:,include],y)
    b = BIC(X[:,include],y,clf.coef_)
    return clf.coef_
                    
def write_predictions(P, prefix, gt, las_predict, fsr_predict, ard_predict, 
                      bgs0_predict, bgs1_predict, bgs2_predict,
                      map0, map1, map2):
    # FIXME: yeah, yeah... hard-coded paths... blargh
    filename = 'data/sim/preds/' + prefix+"_predictions"+str(P)+".out"

    results = np.vstack([gt, las_predict, fsr_predict, ard_predict, bgs0_predict, bgs1_predict, bgs2_predict, map0, map1, map2]).T
    print "Writing predictions to %s." % (filename)

    np.savetxt(filename, results)

def write_predictions_short(P, prefix, bgs1_predict, map1):
    # FIXME: yeah, yeah... hard-coded paths... blargh
    filename = 'data/sim/preds/' + prefix+"_predictions_short"+str(P)+".out"

    results = np.vstack([bgs1_predict, map1]).T
    print "Writing predictions to %s." % (filename)

    np.savetxt(filename, results)

def test_files(path, prefix, P, burnin, iters):
    onlyfiles = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
    print onlyfiles
    for fn in onlyfiles:
        if prefix in fn and 'yx_' in fn and '_'+str(P)+'.' in fn:
            print "opening "+fn
            print fn[0:fn.index('_')]
            (X,y,gt,corr1, corr2) = data.load_data2(path,fn[0:fn.index('_')], str(P))

            model1 = probit.ProbitSS(X, y, data.bend_corr(corr1))
            inclusion_probs1, map1 = model1.run_mcmc(burnin=burnin, iters=iters)  

            write_predictions_short(P, prefix, inclusion_probs1, map1)

def run_test(path, prefix, dims, burnin, iters):
    (X, y, gt, corr, cov) = data.load_data2(path, prefix, dims)
    corr = data.bend_matrix(corr, normalize=True)
    cov  = data.bend_matrix(cov, normalize=False)

    model1 = probit.ProbitSS(X, y, corr)
    inclusion_probs1, map1 = model1.run_mcmc(burnin=burnin, iters=iters)  
    print "Correlation S-S complete"

    write_predictions_short(dims, prefix, inclusion_probs1, map1)

if __name__ == '__main__':
    mkl.set_num_threads(1)
    np.random.seed(1)

    data_dir = 'data/sim/'

    # Load in the list of prefixes.
    prefix_list_file = os.path.join(data_dir, 'real_files.txt')
    #print prefix_list_file
    prefixes = [line.rstrip('\n') for line in open(prefix_list_file)]
    #print prefixes

    if len(sys.argv) == 1:
        # No argument? print total number
        print len(prefixes)
        sys.exit(-1)
    else:        
        prefix_idx = int(sys.argv[1])-1

    print "Running evaluations for %s" % (prefixes[prefix_idx])

    dims = 10000
    run_test(data_dir, prefixes[prefix_idx], dims, 500, 1000)

