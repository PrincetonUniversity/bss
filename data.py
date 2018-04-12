import os
import sys
import csv
import numpy        as np
import scipy.linalg as spla

def load_xy_file(filename):
    fh = open(filename, 'r')
    reader = csv.reader(fh, delimiter=',')

    header = next(reader)
    phenos = np.array(list(map(float, header[2:])))

    eqtls = []
    genos = []
    for row in reader:
        rsid = row[0]
        eqtls.append(float(row[1]))
        genos.append(np.array(list(map(float, row[2:]))))

    eqtls = np.array(eqtls)
    genos = np.array(genos)
    
    fh.close()

    return (genos.T, phenos, eqtls)

def pull_gene_info(path, gene_mapping, gene_idx):
    fh = open(path+gene_mapping, 'r')
    count = 0
    d = []
    for line in fh:
        if count == gene_idx:
            d = line.strip().split()
            break
        count = count + 1
    fh.close()
    return(d[0], d[1], d[2], d[3])

def load_cor_file(filename):
    
    fh = open(filename, 'r')
    reader = csv.reader(fh, delimiter=',')

    header = next(reader)
    corr = []
    for row in reader:
        corr.append(np.array(list(map(float, row))))
    corr = np.array(corr)
    
    fh.close()

    return corr

def load_data(data_dir, prefix):
    xy_file  = os.path.join(data_dir, prefix + '_yx_200.out')
    cor1_file = os.path.join(data_dir, prefix + '_cor_200.out')
    (genos, phenos, eqtls) = load_xy_file(xy_file)
    corr = load_cor_file(cor1_file)
    return (genos, phenos, eqtls, corr)

def load_data2(data_dir, prefix, P):
    xy_file  = os.path.join(data_dir, prefix + '_yx_'+str(P)+'.out')
    cor1_file = os.path.join(data_dir, prefix + '_cor1_'+str(P)+'.out')
    cor2_file = os.path.join(data_dir, prefix + '_cor2_'+str(P)+'.out')
    (genos, phenos, eqtls) = load_xy_file(xy_file)
    corr1 = load_cor_file(cor1_file)
    corr2 = load_cor_file(cor2_file)
    return (genos, phenos, eqtls, corr1, corr2)

def bend_corr(R):
    min_eval = 1e-6
    (evals, evecs) = spla.eigh(R)
    num_bad = np.sum(evals < min_eval)
    if num_bad > 0:
        sys.stderr.write("There are %d negative eigenvalues to bend.\n" % (num_bad))

        # Threshold the eigenvalues at an arbitrary minimum.
        evals[np.nonzero(evals < min_eval)] = min_eval

        # Reassemble the matrix.
        R = np.dot(np.dot(evecs, np.diag(evals)), evecs.T)

        # Ensure that it has ones along the diagonal.
        stds = np.sqrt(np.diag(R))
        R = (R / stds) / stds[:,np.newaxis]

        return R

    else:
        return R

def bend_matrix(R, normalize=True, min_eval=1e-6):
    (evals, evecs) = spla.eigh(R)
    num_bad = np.sum(evals < min_eval)
    if num_bad > 0:
        sys.stderr.write("There are %d negative eigenvalues to bend.\n" % (num_bad))

        # Threshold the eigenvalues at an arbitrary minimum.
        evals[np.nonzero(evals < min_eval)] = min_eval

        # Reassemble the matrix.
        R = np.dot(np.dot(evecs, np.diag(evals)), evecs.T)

        # Ensure that it has ones along the diagonal.
        if normalize:
            stds = np.sqrt(np.diag(R))
            R = (R / stds) / stds[:,np.newaxis]

        return R

    else:
        return R
    
