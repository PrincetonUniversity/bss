import time
import math

import numpy as np
import scipy as sp
import mutual_information as it
import quantile_normalization as qn

from sklearn.linear_model import ARDRegression, LinearRegression, LassoLars, lasso_path, enet_path


# check to see if a matrix is positive definite
def isPosDef(M):
    # is symmetric?
    if (sp.transpose(M) == M).all():
        e_values, e_vectors = np.linalg.eig(M)
        for j in e_values:
            if j <= 0:
                # print("negative evalue")
                return False
        return True
    # print(M)
    return False


def matrix_cor(a):
    N = a.shape[0]
    P = a.shape[1]
    mcor = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            c = sp.stats.pearsonr(a[i, :], a[j, :])[0]
            mcor[i, j] = c
            mcor[j, i] = c
    return mcor


# this is really slow still
def matrix_mi(a):
    N = a.shape[0]
    # mix = np.zeros((N, N))
    mixd = np.zeros((N, N))
    it_tool = it.InformationTheoryTool(a)
    aint = np.zeros(a.shape)
    for i in range(N):
        aint[i, :] = [round(elem) for elem in a[i, ]]
    print(aint)
    for i in range(N):
        # mix[i, i] = it_tool.mutual_information(i,i,2)# it.mutinfo2(a[i,],a[i,])
        mixd[i, i] = it_tool.mutual_information(i, i)
    # print(mix)
    for i in range(N-1):
        for j in range(i+1, N):
            # mi = math.exp(-((mix[i,i]+mix[j,j])/2) + it.mutinfo2(a[i,],a[j,]))
            # mix[i,j] <- mi
            # mix[j,i] <- mi
            mi = math.exp(-((mixd[i, i]+mixd[j, j])/2) + it_tool.mutual_information(i, j))
            mixd[i, j] < -mi
            mixd[j, i] < -mi
        # return mix, mixd
    return mixd


def llik(X, y, beta):
    ll = 0
    N = X.shape[0]
    for i in range(N):
        ll = ll + math.log(sp.stats.norm.pdf(y[i] - np.inner(X[i, :], beta)))
    # print(ll)
    return ll


def AIC(X, y, beta):
    count = 0
    for b in beta:
        if not abs(b) < 0.0000001:
            count = count + 1
    return (-2*llik(X, y, beta)) + (2 * count)


def BIC(X, y, beta):
    count = 0
    N = y.shape[0]
    for b in beta:
        if not abs(b) < 0.0000001:
            count = count + 1
    return (-2*llik(X, y, beta)) + (count * math.log(N))


def LassoRegressionSelection(X, y):
    clf = lasso_path(X, y, normalize=False, copy_X=True)
    alphas_lasso = np.array([model.alpha for model in clf])
    coefs_lasso = np.array([model.coef_ for model in clf])
    min_bic = 0
    min_coef = 0
    for coef in coefs_lasso:
        this_bic = BIC(X, y, coef)
        if min_bic == 0 or this_bic < min_bic:
            min_bic = this_bic
            min_coef = coef
    return min_coef


def ForwardStepwiseRegression(X, y):
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
            clf.fit(X[:, ], y)
            bthis = BIC(X[:, include], y, clf.coef_)
            if max_b == 0 or bthis < max_b:
                max_b = bthis
                max_i = i
            include.pop()
        if max_b == b:
            not_done = False
        else:
            include.append(max_i)
            # print("including "+str(max_i)+" with bic "+str(max_b))
            b = max_b
            remaining.remove(max_i)
    clf.fit(X[:, include], y)
    coefs = [0]*P
    for i in range(len(include)):
        coefs[include[i]] = clf.coef_[i]
    return coefs


def ForwardStepwiseBackwardRegression(X, y, coefs=0):
    N = y.shape[0]
    P = X.shape[1]
    include = [0]*P
    for i in range(len(coefs)):
        c = coefs[i]
        if abs(c) > 0:
            include[i] = 1
    clf = LinearRegression(copy_X=True)
    clf.fit(X[:, include], y)
    b = BIC(X[:, include], y, clf.coef_)
    not_done = True
    while not_done:
        max_b = b
        max_i = 0
        i = 0
        length = len(include)
        while i < length:
            ind = include[i]
            include.remove(ind)
            clf.fit(X[:, include], y)
            bthis = BIC(X[:, include], y, clf.coef_)
            if bthis < max_b:
                max_b = bthis
                max_i = ind
            include.insert(i, ind)
            i = i+1
        if max_b == b:
            not_done = False
        else:
            include.remove(max_i)
            b = max_b
    clf.fit(X[:, include], y)
    b = BIC(X[:, include], y, clf.coef_)
    return clf.coef_


class Benchmark(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("%s : %0.3f seconds" % (self.name, end-self.start))
        return False
