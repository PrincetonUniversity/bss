[![Build Status](https://travis-ci.com/PrincetonUniversity/bssr.svg?token=gFm1C2iKiRpokJuZp7Ab&branch=master)](https://travis-ci.com/PrincetonUniversity/bssr)

# Bayesian Structured Sparsity from Gaussian Fields

Companion code for the paper

[Bayesian Structured Sparsity from Gaussian Fields](https://arxiv.org/abs/1407.2235)

Barbara E. Engelhardt, Ryan P. Adams

Abstract:

Substantial research on structured sparsity has contributed to analysis of many different applications.
However, there have been few Bayesian procedures among this work. Here, we develop a Bayesian model for structured sparsity
that uses a Gaussian process (GP) to share parameters of the sparsity-inducing prior in proportion to feature similarity as defined by an arbitrary positive definite kernel. For linear regression, this sparsity-inducing prior on regression coefficients is a relaxation of the canonical spike-and-slab prior that flattens the mixture model into a scale mixture of normals. This prior retains the explicit posterior probability on inclusion parameters---now with GP probit prior distributions---but enables tractable computation via elliptical slice sampling for the latent Gaussian field. We motivate development of this prior using the genomic application of association mapping, or identifying genetic variants associated with a continuous trait. Our Bayesian structured sparsity model produced sparse results with substantially improved sensitivity and precision relative to comparable methods. Through simulations, we show that three properties are key to this improvement:

i) modeling structure in the covariates,

ii) significance testing using the posterior probabilities of inclusion, and

iii) model averaging.

We present results from applying this model to a large genomic dataset to demonstrate computational tractability.
