import numpy
import scipy.special


def gammaln(x):
    """
    Log of the gamma function.
    """
    # small  = numpy.nonzero(x < numpy.finfo(numpy.float64).eps)
    result = scipy.special.gammaln(x)
    # result[small] = -numpy.log(x[small])
    return result


def gammapdfln(x, a, b):
    """
    Log of the gamma distribution PDF.
    """
    return -gammaln(a) + a*numpy.log(b) + (a-1.0)*numpy.log(x) - b*x


def invgammapdfln(x, a, b):
    return -gammaln(a) + a*numpy.log(b) - (a+1.0)*numpy.log(x) - b/x


def betapdfln(x, a, b):
    """
    Log of the beta distribution PDF.
    """
    return gammaln(a + b) - gammaln(a) - gammaln(b) + (a-1)*numpy.log(x) + (b-1)*numpy.log(1-x)
