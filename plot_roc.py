import sys
import numpy as np
import pylab as pl


def compute_roc(truth, preds):
    s_preds = np.sort(preds)
    inc_pred = s_preds[:, np.newaxis] >= preds
    b_truth = truth.astype(np.bool)
    tp = np.sum(inc_pred & b_truth, axis=1).astype(np.float)/np.sum(b_truth)
    fp = np.sum(inc_pred & ~b_truth, axis=1).astype(np.float)/np.sum(~b_truth)
    return np.hstack([0, fp]), np.hstack([0, tp])


if __name__ == '__main__':
    filename = sys.argv[1]
    inclusion = np.abs(np.loadtxt(filename))

    fp = np.zeros((inclusion.shape[0]-1, inclusion.shape[1]+1))
    tp = np.zeros((inclusion.shape[0]-1, inclusion.shape[1]+1))
    for ii in range(inclusion.shape[0]-1):
        fp[ii, :], tp[ii, :] = compute_roc(inclusion[0, :], inclusion[ii+1, :])

    pl.plot(fp.T, tp.T)
    pl.show()

