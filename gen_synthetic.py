import numpy        as np
import numpy.random as npr
import pylab        as pl

rng_seed        = 1  
num_snps        = 500   # The number of SNPs.
len_scale       = 50    # The length-scale of the structured sparsity, in SNPs.
global_sparsity = 0.05  # The global non-zero probability.
local_sparsity  = 0.1   # The within-group non-zero probability.
num_data        = 200   # Number of individuals.

# Compute the level of between-group sparsity.
if local_sparsity < global_sparsity:
    raise Exception("Local sparsity must be greater than global sparsity.")
else:
    group_sparsity = global_sparsity / local_sparsity
    print "Between-Group Sparsity: %f" % (group_sparsity)
    print "Within-Group Sparsity:  %f" % (local_sparsity)
    print "Effective Sparsity:     %f" % (global_sparsity)

# Index the SNPs.
X = np.arange(num_snps) + 1

# Generate the correlation matrix.
R = np.exp( -0.5 * ((X[:,np.newaxis] - X)/len_scale)**2 ) + 1e-6*np.eye(num_snps)

# Sample the group sparsity pattern.
