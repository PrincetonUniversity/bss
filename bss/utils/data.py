import os.path
import glob
import csv
import numpy as np


def load_xy_file(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        header = next(reader)
        phenos = np.array(list(map(float, header[2:])))

        eqtls = []
        genos = []
        for row in reader:
            eqtls.append(bool(int(row[1])))
            genos.append(np.array(list(map(float, row[2:]))))

        eqtls = np.array(eqtls)
        genos = np.array(genos)

        return genos.T, phenos, eqtls


def load_cor_file(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        _ = next(reader)  # consume header
        corr = []
        for row in reader:
            corr.append(np.array(list(map(float, row))))
        corr = np.array(corr)

        return corr


def load_data(pattern):
    """
    A generator function to load the 'main' data and correlation data for a given file pattern
    :param pattern: A file pattern (with directory path), consumable by the glob module for *xy* files to process
        For example: '/some/path/data/real0_yx_*.*'
    :return: A 4-tuple of numpy vectors:
        X: A numpy feature matrix (mxn) of genotype values, for m phenotypes and n SNPs
        Y: A numpy vector (mx1) of phenotype values - the response variable
        eqtls: A numpy vector (nx1) of eQTL values - one for each SNP
        cor: A numpy matrix (nxn) of correlation values for each pair of n SNPs
    """
    for filename in glob.glob(pattern):
        dirname = os.path.dirname(filename)
        root, ext = os.path.splitext(os.path.basename(filename))
        prefix, _, suffix = root.split('_')
        xy_filename = os.path.join(dirname, '{}_yx_{}{}'.format(prefix, suffix, ext))
        correlation_filename = os.path.join(dirname, '{}_cor1_{}{}'.format(prefix, suffix, ext))

        return load_xy_file(xy_filename) + (load_cor_file(correlation_filename),)
