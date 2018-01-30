import numpy as np
import scipy as sp
import mutual_information as it
import quantile_normalization as qn
import os
import random
import math

# where to pull actual gene/SNP data
path1 = '/Users/engelhardt/work/data/casey/'
# where to put simulated data
path = "/Users/engelhardt/work/data/ryan/bayesian-group-sparsity/data/sim/" 

# check to see if a matrix is positive definite
def isPosDef(M):
    # is symmetric?
    if (sp.transpose(M) == M).all():
        e_values, e_vectors = np.linalg.eig(M)
        for j in e_values:
            if j <= 0:
                #print "negative evalue"
                return False
        return True
    #print M
    return False

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

def matrix_cor(a):
    N = a.shape[0]
    P = a.shape[1]
    mcor = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            c = sp.stats.pearsonr(a[i,:], a[j,:])[0]
            mcor[i,j] = c
            mcor[j,i] = c
    return mcor

# this is really slow still
def matrix_mi(a):
   N = a.shape[0]
   #mix = np.zeros((N,N))
   mixd = np.zeros((N,N))
   it_tool = it.InformationTheoryTool(a)
   aint = np.zeros(a.shape)
   for i in range(N):
       aint[i,:] = [ round(elem) for elem in a[i,] ]
   print aint
   for i in range(N):
       #mix[i,i] = it_tool.mutual_information(i,i,2)# it.mutinfo2(a[i,],a[i,])
       mixd[i,i] = it_tool.mutual_information(i,i)
   #print mix
   for i in range(N-1):
       for j in range(i+1,N):
           #mi = math.exp(-((mix[i,i]+mix[j,j])/2) + it.mutinfo2(a[i,],a[j,]))
           #mix[i,j] <- mi
           #mix[j,i] <- mi
           mi = math.exp(-((mixd[i,i]+mixd[j,j])/2) + it_tool.mutual_information(i,j))
           mixd[i,j] <- mi
           mixd[j,i] <- mi
   #return mix, mixd
   return mixd

# include genes, gene names, and the number of SNPs per gene: P
def simulate_tag_data(genes, genenames, P):
    N = genes.shape[0]
    for i in range(len(genenames)):
        gn = genenames[i]
        gene = genes[:,i]
        prev = []
        fn = path1+"cisgenes/CAP_LCL_"+gn+"_cis_snps.out2"
        fn2 = path1+"cisgenes2/CAP_LCL_"+gn+"_cis_snps.out2"
        print fn
        if os.path.exists(fn):
            ns = file_len(fn)
            print N
            print ns
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
        covmat = matrix_cor(snps[indices_sub,:])
        covmat2 = np.cov(snps[indices_sub,:])
        #covmat3 = matrix_mi(snps[indices_sub,:])
        print isPosDef(covmat)
        print isPosDef(covmat2)
        #print isPosDef(covmat3)
        
        eqtlindices = [0] * (total_snps - tag_eqtls)
        for j in eqtls[0:(total_eqtls-tag_eqtls)]:
            eqtlindices[j] = 1
        print eqtls
        print "Number of eqtls (non-tag): "+str(sum(eqtlindices))
        print "Number of eqtls (tag): "+str(tag_eqtls)
        genesim = []
        acov = random.uniform(0.5, 2)
        for k in range(N):
            sm = 0
            for j in eqtls: # include all of the eQTLs here, even the tag SNPs.
                sm = sm + snps_sub[j,k]
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
        print fn
        if os.path.exists(fn):
            ns = file_len(fn)
            print N
            print ns
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
        covmat = matrix_cor(snps)
        print snps.shape
        covmat2 = np.cov(snps)
        #covmat3 = matrix_mi(snps)
        print isPosDef(covmat)
        print isPosDef(covmat2)
        #print isPosDef(covmat3)
        
        eqtlindices = [0] * P
        for j in eqtls:
            eqtlindices[j] = 1
        print eqtls
        genesim = []
        acov = random.uniform(0.5, 2)
        for k in range(N):
            sm = 0
            for j in eqtls:
                sm = sm + snps[j,k]
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
        fn = path1+"cisgenes/CAP_LCL_"+gn+"_cis_snps.out2"
        fn2 = path1+"cisgenes2/CAP_LCL_"+gn+"_cis_snps.out2"
        print fn
        if os.path.exists(fn):
            ns = file_len(fn)
            print N
            print ns
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
        covmat = matrix_cor(snps)
        print snps.shape
        covmat2 = np.cov(snps)
        #covmat3 = matrix_mi(snps)
        print isPosDef(covmat)
        print isPosDef(covmat2)
        #print isPosDef(covmat3)
        
        eqtlindices = [0] * covmat2.shape[0]

        write_matrix_csv(rsids, covmat, path+"/real"+str(i)+"_cor1_"+str(P)+".out")
        write_matrix_csv(rsids, covmat2, path+"/real"+str(i)+"_cor2_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat3, path+"/real"+str(i)+"_cor3_"+str(P)+".out")
        #write_matrix_csv(rsids, covmat4, path+"/real"+str(i)+"_cor4_"+str(P)+".out")
        write_gene_csv(gn, gene, rsids, eqtlindices, snps, path+"/real"+str(i)+"_yx_"+str(P)+".out")

def write_gene_csv(genename, genesim, rsids, eqtlindices, snps, fn):
    outf = open(fn, 'w')
    outf.write(genename+',eqtl')
    for gi in genesim:
        outf.write(','+str(gi))
    outf.write('\n')
    for i in range(len(rsids)):
        outf.write(rsids[i]+','+str(eqtlindices[i]))
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

    genes = read_table(path1+'/CAP_LCL-eqtl-genes.out', genes, header = True, genenames=genenames)
    print genes[1:10,1:10]
    simulate_tag_data(genes, genenames, 200)
    #simulate_ryan_data(genes, genenames, 200)
    #write_actual_data(genes, genenames[1:200], 10000)

if __name__ == "__main__":
    main()

