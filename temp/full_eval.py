#import mkl
import sys
import numpy as np
import os
#from scipy import stats
from temp import probit, utils
import data

CIS = 1000000

#path = "/Users/engelhardt/work/data/ryan/bayesian-group-sparsity/data/sim/"
path_results = "/Users/engelhardt/work/data/ryan/bayesian-group-sparsity/data/full/results/"

# read in set of files to test
def read_yx_files(filename, y, X, gt):
    header = True
    inf = open(filename, 'r')
    snpscount = 0
    for line in inf:
        d = line.strip().split(',')
        if header == True:
            header = False
            for i in range(2,len(d)):
                y.append((float)(d[i]))
        else:
            gt.append(d[1])
            for i in range(2,len(d)):
                X.append((float)(d[i]))
            snpscount = snpscount + 1

def write_predictions(P, prefix, gt, las_predict, fsr_predict, ard_predict, 
                      bgs0_predict, bgs1_predict, bgs2_predict,
                      map0, map1, map2):
    # FIXME: yeah, yeah... hard-coded paths... blargh
    filename = 'data/sim/preds/' + prefix+"_predictions"+str(P)+".out"

    results = np.vstack([gt, las_predict, fsr_predict, ard_predict, bgs0_predict, bgs1_predict, bgs2_predict, map0, map1, map2]).T
    print("Writing predictions to %s." % (filename))

    np.savetxt(filename, results)

def write_predictions_short(P, prefix, bgs1_predict, map1):
    # FIXME: yeah, yeah... hard-coded paths... blargh
    filename = 'data/sim/preds/' + prefix+"_predictions_short"+str(P)+".out"

    results = np.vstack([bgs1_predict, map1]).T
    print("Writing predictions to %s." % (filename))

    np.savetxt(filename, results)

def write_predictions_full(path, prefix, gene_name, rsids, bgs1_predict, map1):
    filename = path+prefix+"_"+gene_name+"_predictions.out"

    rsidsx = np.array(rsids, dtype='|S20')
    results = np.vstack([rsidsx, bgs1_predict, map1]).T
    print("Writing predictions to %s." % (filename))

    np.savetxt(filename, results, fmt='%s')

def test_files(path, prefix, P, burnin, iters):
    onlyfiles = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
    print(onlyfiles)
    for fn in onlyfiles:
        if prefix in fn and 'yx_' in fn and '_'+str(P)+'.' in fn:
            print("opening "+fn)
            print(fn[0:fn.index('_')])
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
    print("Correlation S-S complete")

    write_predictions_short(dims, prefix, inclusion_probs1, map1)

def run_eval(path, gene_name, gene_idx, gene_chr, gene_tss, gene_tes, 
             genotype_prefix, burnin, iters):
    # pull SNPs X
    rsids = []
    positions = []
    pfile = open(path+genotype_prefix+"_chr"+str(gene_chr)+".pos", 'r')
    for line in pfile:
        d = line.strip().split()
        if (int)(d[1]) >= gene_tss-CIS and (int)(d[1]) <= gene_tes+CIS:
            rsids.append(d[0])
            positions.append((int)(d[1]))

    genos = []
    usedrsids = []
    xfile = open(path+genotype_prefix+"_chr"+str(gene_chr)+".wmg", 'r')
    for line in xfile:
        d = line.strip().split()
        if d[0] in rsids:
            genos.append(np.array(list(map(float, d[3:]))))
            usedrsids.append(d[0])
    X = np.array(genos)

    # compute covariance matrix (correlation)
    covmat = utils.matrix_cor(X)

    # pull gene exp y: file is genes x individuals
    gfile = open(path+gene_exp, 'r')
    count = 0
    for line in gfile:
        #print(count)
        #print(gene_idx)
        #print('one iteration')
        if count == gene_idx:
            y = np.array(list(map(float, line.strip().split())))
            break
        count = count + 1
    
    corr = data.bend_matrix(covmat, normalize=True)
    print(corr[:,1])
    print(X[:,1])
    print(y)
    model1 = probit.ProbitSS(X.T, y, corr)
    inclusion_probs1, map1 = model1.run_mcmc(burnin=burnin, iters=iters)  
    print("Correlation S-S complete")

    write_predictions_full(path_results, genotype_prefix, gene_name+'_'+str(gene_idx), 
                           usedrsids, inclusion_probs1, map1)


# args: genotype_prefix gene_exp gene_mapping gene_number
if __name__ == '__main__':
    #mkl.set_num_threads(1)
    np.random.seed(1)

    data_dir = 'data/full/harvard/'

    if len(sys.argv) < 5:
        # Bad argument?
        print("Expecting: genotype_prefix gene_exp gene_mapping gene_number")
        sys.exit(-1)
    else:        
        genotype_prefix = sys.argv[1]
        gene_exp = sys.argv[2]
        gene_mapping = sys.argv[3]
        gene_idx = int(sys.argv[4])-1

    [gene_name, gene_chr, gene_tss, gene_tes] = data.pull_gene_info(data_dir, gene_mapping, gene_idx)

    # don't pull mt/x/y genes! no genotypes.
    if gene_chr in ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
                    '10', '11', '12', '13', '14', '15', '16', 
                    '17', '18', '19', '20', '21', '22']:

        print("Running evaluations for %s" % (gene_name+'_'+str(gene_idx)))
        run_eval(data_dir, gene_name, gene_idx, gene_chr, (int)(gene_tss), (int)(gene_tes), 
                 genotype_prefix, 50, 100)

