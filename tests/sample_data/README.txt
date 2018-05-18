---------------
README
---------------

This folder has sample data that feeds into the Probit Model, and which serves as data for unit-testing.
There are two patterns of filenames in this folder:

real<p>_yx_<n>.out
real<p>_cor1_<n>.out

where <p> and <n> are integers.

Each matching pair of files provides us with all the data we need to run the model. Theoretically, multiple instances
of the model can be run in sequentially or in parallel for each matching pair of files.
Each file in a pair is described next.

----------------------------------------
SNP Expression File = real<p>_yx_<n>.out
----------------------------------------

This is a comma-separated CSV file with a header, in the general format as follows:

+-------------+------+---------------+----------------+---------------+---
| <source>   | eqtl | 1.1059457526  | 0.638855919079 | 1.11562348902  | ..
+------------+------+---------------+----------------+----------------+---
| rs8104414  |    0 |          0.41 |           0.95 |           0.03 | ..
| rs11672955 |    0 |          0.26 |           1.88 |           0.25 | ..
| ..         |   .. |          1.85 |           1.00 |           0.07 | ..
+----------+------+--------------+----------------+---------------+---|---

- <source> corresponds to (but may not match exactly) the name of the gene where the SNP data comes from. Each value
  in this column is the RSID of the SNP for that row. The SNPs obtained from these rows must match the SNPs in the
  correlation file, described later.

- eqtl is an integer field with values 0/1, indicating the expression level of the SNP.
  This field is not read and does not play a role in the model.

- Fieldnames for the 3rd column (and thereafter) have names of the form <f1>, <f2>, .. <fN>, all floating-point numbers,
  +ve or -ve. These numbers indicate the N cumulative "phenotype" values, which serve as the Y values in our sparse
  regression model.

 Each value in column I is a floating-point number in the range [0,2], and corresponds to the genotype value of
 a particular SNP towards the Ith phenotype. In the example above, rs11672955 has a contribution of 1.88 towards the
 phenotype whose cumulative expression level is 0.6388

 These values thus make up the predictor matrix X of the model. Specifically, if we think of each SNP as a "feature"
 that results in the cumulative Y value indicated by the phenotype (the column header), then our feature matrix X and
 response vector Y are

     | 0.41 0.26 1.85 .. |        | 1.1059457 |
 X = | 0.95 1.88 1.00 .. |    Y = | 0.6388559 |
     | 0.03 0.25 0.07 .. |        | 1.1156234 |
     | ..   ..   ..   .. |        | ..        |

-------------------------------------------
SNP Correlation File = real<p>_cor1_<n>.out
-------------------------------------------

This is a comma-separated CSV file with a header, in the general format as follows:

+-----------+------------+-----------+---
| rs8104414 | rs11672955 | rs3746129 | ..
+-----------+------------+-----------+---
| 1.0       | 0.164999   |  -0.86125 | ..
| 0.164999  | 1.0        |   0.34216 | ..
| -0.86125  | 0.34216    |       1.0 | ..
| ..        | ..         |        .. | ..
+-----------+------------+-----------+---

- Header names are the RSIDs, and the values are floating-point numbers in the range [-1, 1].
  These values correspond to SNP correlation values between any two pair of SNPs. It is assumed that the SNP order in
  the rows is the same as the SNP order in columns. These values thus directly give us the covariance matrix of SNPs.
  The covariance matrix sigma thus obtained must be symmetric with 1s on the diagonal, but may not be positive definite.

          |  1.0    0.165  -0.861 ..  |
  Sigma = |  0.165  1.0     0.342 ..  |
          | -0.861  0.342   1.0   ..  |
          | ..      ..      ..    1.0 |
