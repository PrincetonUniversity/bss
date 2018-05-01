import numpy as np
from scipy import stats
import os
import random
from temp import utils, quantile_normalization as qn
import data

# where to pull actual gene/SNP data
path1 = '/Users/engelhardt/work/data/casey/'
# where to put simulated data
path = "/Users/engelhardt/work/data/ryan/bayesian-group-sparsity/data/sim/" 

# return the number of rows in a file.
def file_len(filename):
    with open(filename) as f:
        i = -1
        for i, l in enumerate(f):
            pass
    return i + 1

# read a general file/table
def read_table(filename, genes, header = False, rownames = False, rnames=[], startat = 0, genenames=[]):
    header = True
    nrows = file_len(filename) - 1
    ncols = 0
    inf = open(filename)
    lc = 0
    for line in inf:
        if header == True:
            d = line.strip().split()
            genenames.extend(d)
            ncols = len(d)- startat
            genes = np.zeros((nrows,ncols))
            header = False
        else:
            d = line.strip().split()
            if ncols == 0:
                ncols = len(d) - startat
                genes = np.zeros((nrows,ncols))
            if rownames == True:
                rnames.append(d[0])
            for i in range(startat,ncols):
                genes[lc,i] = (float)(d[i])
            lc = lc+1            
    return genes

# include genes, gene names, and the number of SNPs per gene: P
def simulate_tag_data(genes, genenames, P):
    N = genes.shape[0]
    for i in range(len(genenames)):
        gn = genenames[i]
        gene = genes[:,i]
        prev = []
        fn = path1+"cisgenes/CAP_LCL_"+gn+"_cis_snps.out2"
        fn2 = path1+"cisgenes2/CAP_LCL_"+gn+"_cis_snps.out2"
        print(fn)
        if os.path.exists(fn):
            ns = file_len(fn)
            print(N)
            print(ns)
            snps = np.zeros((N,ns))
            rsids = []
            snps = read_table(fn, snps, rownames=True, rnames = rsids, startat=3)
        elif os.path.exists(fn2):
            ns = file_len(fn2)
            snps = np.zeros((N,file_len(fn2)))
            rsids = []
            snps = read_table(fn2, snps, rownames=True, rnames = rsids, startat=3)
        else:
            continue

        # find P indices to include
        tag_eqtls = random.sample(range(1,5),1)[0]
        total_snps = min(P+tag_eqtls, ns)
        indices = random.sample(range(len(rsids)), total_snps)
        print(indices)
        total_eqtls = tag_eqtls + random.sample(range(1,5), 1)[0]
        eqtls = random.sample(range(total_snps-tag_eqtls), total_eqtls)
        indices_sub = indices[0:(total_snps - tag_eqtls)] # remove the last tag_eqtls indices, as those are gone.

        rsids = [ rsids[j] for j in indices_sub ]
        snps_sub = snps[indices,:]
        covmat = utils.matrix_cor(snps[indices_sub, :])
        covmat2 = np.cov(snps[indices_sub,:])
        #covmat3 = matrix_mi(snps[indices_sub,:])
        print(utils.isPosDef(covmat))
        print(utils.isPosDef(covmat2))
        # print(utils.isPosDef(covmat3))
        
        effects = []
        eqtlindices = [0] * (total_snps - tag_eqtls)
        for j in eqtls:
            effect = (random.betavariate(0.1, 0.1)*2.0)-1.0
            if j in eqtls[0:(total_eqtls-tag_eqtls)]:
                eqtlindices[j] = effects
            effects.append(effect)
        print(eqtls)
        print("Number of eqtls (non-tag): "+str(sum(eqtlindices)))
        print("Number of eqtls (tag): "+str(tag_eqtls))
        genesim = []
        acov = random.uniform(0.5, 2)
        for k in range(N):
            sm = 0
            count = 0
            for j in eqtls: # include all of the eQTLs here, even the tag SNPs.
                sm = sm + (effects[count]*snps_sub[j,k])
                count = count + 1
            genesim.append(random.gauss(sm,acov))

        genesim = qn.qqnorm(genesim)
        write_matrix_csv(indices_sub, covmat, path+"/simt"+str(i)+"_cor1_"+str(P)+".out")
        write_matrix_csv(indices_sub, covmat2, path+"/simt"+str(i)+"_cor2_"+str(P)+".out")
        #write_matrix_csv(indices_sub, covmat3, path+"/simt"+str(i)+"_cor3_"+str(P)+".out")
        #write_matrix_csv(indices_sub, covmat4, path+"/simt"+str(i)+"_cor4_"+str(P)+".out")
        write_gene_csv(gn, genesim, rsids, eqtlindices, snps[indices_sub,:], path+"/simt"+str(i)+"_yx_"+str(P)+".out")


# include genes, gene names, and the number of SNPs per gene: P
def simulate_ryan_data(genes, genenames, P):
    N = genes.shape[0]
    for i in range(len(genenames)):
        gn = genenames[i]
        gene = genes[:,i]
        prev = []
        fn = path1+"cisgenes/CAP_LCL_"+gn+"_cis_snps.out2"
        fn2 = path1+"cisgenes2/CAP_LCL_"+gn+"_cis_snps.out2"
        print(fn)
        if os.path.exists(fn):
            ns = file_len(fn)
            print(N)
            print(ns)
            snps = np.zeros((N,ns))
            rsids = []
            snps = read_table(fn, snps, rownames=True, rnames = rsids, startat=3)
        elif os.path.exists(fn2):
            ns = file_len(fn2)
            snps = np.zeros((N,file_len(fn2)))
            rsids = []
            snps = read_table(fn2, snps, rownames=True, rnames = rsids, startat=3)
        else:
            continue

        # find P indices to include
        indices = random.sample(range(len(rsids)), P)
        print(indices)

        rsids = [ rsids[j] for j in indices ]
        snps = snps[indices,:]
        covmat = utils.matrix_cor(snps)
        print(snps.shape)
        covmat2 = np.cov(snps)
        #covmat3 = matrix_mi(snps)
        print(utils.isPosDef(covmat))
        print(utils.isPosDef(covmat2))
        # print(utils.isPosDef(covmat3))
        
        eqtlindices = [0] * P
        total_eqtls = random.sample(range(2,7),1)[0]
        eqtls = random.sample(range(P), total_eqtls)
        for j in eqtls:
            eqtlindices[j] = (random.betavariate(0.1, 0.1)*2.0)-1.0
        print(eqtls)
        genesim = []
        acov = random.uniform(0.5, 2)
        for k in range(N):
            sm = 0.0
            for l in range(len(eqtls)):
                j = eqtls[l]
                sm = sm + eqtlindices[j]*snps[j,k]
            genesim.append(random.gauss(sm,acov))

        genesim = qn.qqnorm(genesim)
        write_matrix_csv(indices, covmat, path+"/sim"+str(i)+"_cor1_"+str(P)+".out")
        write_matrix_csv(indices, covmat2, path+"/sim"+str(i)+"_cor2_"+str(P)+".out")
        #write_matrix_csv(indices, covmat3, path+"/sim"+str(i)+"_cor3_"+str(P)+".out")
        #write_matrix_csv(indices, covmat4, path+"/sim"+str(i)+"_cor4_"+str(P)+".out")
        write_gene_csv(gn, genesim, rsids, eqtlindices, snps, path+"/sim"+str(i)+"_yx_"+str(P)+".out")


# include genes, gene names, and the number of SNPs per gene: P
# genenames = []

def write_actual_data(genes, genenames, P):
    N = genes.shape[0]
    for i in range(len(genenames)):
        gn = genenames[i]
        gene = genes[:,i]
        prev = []
        fn = path1+"cissnps/CAP_LCL_"+gn+"_cis_snps_AH.out2"
        #fn = path1+"cisgenes/CAP_LCL_"+gn+"_cis_snps.out2"
        fn2 = path1+"cisgenes2/CAP_LCL_"+gn+"_cis_snps.out2"
        print(fn)
        if os.path.exists(fn):
            ns = file_len(fn)
            print(N)
            print(ns)
            snps = np.zeros((N,ns))
            rsids = []
            snps = read_table(fn, snps, rownames=True, rnames = rsids, startat=3)
        elif os.path.exists(fn2):
            ns = file_len(fn2)
            snps = np.zeros((N,file_len(fn2)))
            rsids = []
            snps = read_table(fn2, snps, rownames=True, rnames = rsids, startat=3)
        else:
            continue

        # find P indices to include
        indices = random.sample(range(len(rsids)), P)
        print(indices)

        covmat = utils.matrix_cor(snps)
        # print(snps.shape)
        #covmat2 = np.cov(snps)
        #covmat3 = matrix_mi(snps)
        print(utils.isPosDef(covmat))
        # print(utils.isPosDef(covmat2))
        # print(utils.isPosDef(covmat3))
        
        eqtlindices = [0] * covmat.shape[0]

        write_matrix_csv(rsids, covmat, path+"/real"+str(i)+"_cor1_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat2, path+"/real"+str(i)+"_cor2_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat3, path+"/real"+str(i)+"_cor3_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat4, path+"/real"+str(i)+"_cor4_"+str(P)+".out")
        write_gene_csv(gn, gene, rsids, eqtlindices, snps, path+"/real"+str(i)+"_yx_"+str(P)+".out")

def write_actual_subsampled_data(genes, genenames, P):
    N = genes.shape[0]
    print(N)
    for i in range(9):
        gn = genenames[i]
        gene = genes[:,i]
        prev = []
        fn = path+"real"+str(i)+"_yx_10000.out"
        print(fn)
        if os.path.exists(fn):
            ns = file_len(fn)
            print(N)
            print(ns)
            #snps = np.zeros((N,ns))
            #gene2 = np.zeros(N)
            #rsids = []
            (snps, gene2, rsids) = data.load_xy_file(fn)
        else:
            continue

        # find P indices to include
        print(snps.shape)
        snps = snps.T
        bestpvs = [1.0, 1.0, 1.0]
        bestinds = [-1, -1, -1]
        for j in range(len(snps)):
            #gradient, intercept, r_value, p_value, std_err = stats.linregress(snps[j,:],gene)
            gradient, intercept, r_value, p_value, std_err = stats.linregress(gene2,snps[j,:])
            # print(p_value)
            for k in range(len(bestpvs)):
                if p_value < bestpvs[k]:
                    bestpvs[k] = p_value
                    bestinds[k] = j
                    break
        print(bestpvs)
        allindices = range(len(rsids))
        for k in range(len(bestinds)):
            allindices.remove(bestinds[k])
        indices = random.sample(allindices, P-3)
        indices.extend(bestinds)
        print(indices)

        rsids = [ rsids[j] for j in indices ]
        snps = snps[indices,:]
        covmat = utils.matrix_cor(snps)
        # print(snps.shape)
        #covmat2 = np.cov(snps)
        #covmat3 = matrix_mi(snps)
        print(utils.isPosDef(covmat))
        # print(utils.isPosDef(covmat2))
        # print(utils.isPosDef(covmat3))
        
        eqtlindices = [0] * covmat.shape[0]
        for k in range(len(bestinds)):
            eqtlindices[P-1-k] = 1

        write_matrix_csv(rsids, covmat, path+"/real"+str(i)+"_cor1_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat2, path+"/real"+str(i)+"_cor2_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat3, path+"/real"+str(i)+"_cor3_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat4, path+"/real"+str(i)+"_cor4_"+str(P)+".out")
        write_gene_csv(gn, gene2, rsids, eqtlindices, snps, path+"/real"+str(i)+"_yx_"+str(P)+".out")

def write_gene_csv(genename, genesim, rsids, eqtlindices, snps, fn):
    outf = open(fn, 'w')
    outf.write(genename+',eqtl')
    for gi in genesim:
        outf.write(','+str(gi))
    outf.write('\n')
    for i in range(len(rsids)):
        outf.write(str(rsids[i])+','+str(eqtlindices[i]))
        for j in range(len(snps[i,])):
            outf.write(','+str(snps[i,j]))
        outf.write('\n')
    outf.close()

def write_matrix_csv(header, mat, fn):
    outf = open(fn, 'w')
    outf.write(str(header[0]))
    for i in range(1,len(header)):
        outf.write(','+str(header[i]))
    outf.write('\n')
    for i in range(mat.shape[0]):
        outf.write(str(mat[i,0]))
        for j in range(1,len(mat[i,])):
            outf.write(','+str(mat[i,j]))
        outf.write('\n')
    outf.close()


def main():
    genes = np.zeros((2, 1))
    genenames = []

    #genes = read_table(path1+'/CAP_LCL-eqtl-genes.out', genes, header = True, genenames=genenames)
    genes = read_table(path1+'/CAP_LCL-eqtl-AH-genes.out', genes, header = True, genenames=genenames)
    print(genes[1:10,1:10])
    #simulate_tag_data(genes, genenames, 200)
    #simulate_ryan_data(genes, genenames[1:500], 1000)
    write_actual_subsampled_data(genes, genenames, 100)
    #write_actual_data(genes, genenames, 10000)

if __name__ == "__main__":
    main()

