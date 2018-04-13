import numpy
import scipy.special
from scipy.stats import gamma, beta


def gammapdfln(x, a, b):
    """
    Log of the gamma distribution PDF.
    """
    return gamma(a, scale=1./b).logpdf(x)



def betapdfln(x, a, b):
    """
    Log of the beta distribution PDF.
    """
    return beta(a, b).logpdf(x)
