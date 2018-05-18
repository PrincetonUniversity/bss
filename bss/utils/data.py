"""
Functions to load the feature matrix X, the response vector y, and covariance matrix sigma from data files.
"""

import os.path
import glob
import csv
import numpy as np


def load_xy_file(path, delimiter=','):
    """
    Load the feature matrix X and response vector y from a given file specified by path.

    Parameters
    ----------
    path : str
        The full path to the 'xy' file containing SNP expression data.
    delimiter: str, optional
        The delimiter for the csv file

    Returns
    -------
    tuple
        A 2 tuple of values - (a NxD feature matrix, a Nx1 response vector)
    """
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)

        header = next(reader)
        phenos = list(map(float, header[2:]))

        genos = []
        for row in reader:

            genos.append(np.array(list(map(float, row[2:]))))

        x = np.array(genos).T, np.array(phenos)
        return x


def load_cor_file(path, delimiter=','):
    """
    Load the correlation file from a given file specified by path.

    Parameters
    ----------
    path : str
        The full path to the 'cor' file containing correlation values of pairs of SNPs.
    delimiter: str, optional
        The delimiter for the csv file

    Returns
    -------
    ndarray
        A DxD symmetric matrix of correlation values between pairs of SNPs
    """
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)

        _ = next(reader)  # consume header
        corr = []
        for row in reader:
            corr.append(np.array(list(map(float, row))))

        return np.array(corr)


def load_data(pattern):
    """
    A function to load the 'main' data and correlation data for a given file pattern

    Parameters
    ----------
    pattern : str
        A file pattern (with directory path), consumable by the glob module for *xy* files to process
        For example: '/some/path/data/real0_yx_*.*'

    Returns
    -------
    tuple
        A 3-tuple of numpy arrays:

        X: A numpy feature matrix (mxn) of genotype values, for m phenotypes and n SNPs

        Y: A numpy vector (mx1) of phenotype values - the response variable

        cor: A numpy matrix (nxn) of correlation values for each pair of n SNPs
    """
    for filename in glob.glob(pattern):
        dirname = os.path.dirname(filename)
        root, ext = os.path.splitext(os.path.basename(filename))
        prefix, _, suffix = root.split('_')
        xy_filename = os.path.join(dirname, '{}_yx_{}{}'.format(prefix, suffix, ext))
        correlation_filename = os.path.join(dirname, '{}_cor1_{}{}'.format(prefix, suffix, ext))

        return load_xy_file(xy_filename) + (load_cor_file(correlation_filename),)
