Note: This project has been moved to [CHIPPR](https://github.com/aimalz/chippr)!

# prob-z

The goal of this work is to demonstrate how to use probability distributions on phtometric redshifts of galaxies, rather than point estimates of redshifts, to make inferences in cosmology.  At this stage, I'm working with synthetic likelihoods and posteriors, as opposed to calculating either from photometry.  However, future work may focus on improving methods for turning photometry into redshift probability distributions.

## nz

The redshift distribution function N(z) is a relatively simple statistic that has previously been estimated from published posteriors.  I'm working on improving upon the method developed by [Sheldon, et al. (2011)](http://arxiv.org/pdf/1109.5192.pdf) for using those posteriors to do inference.  To run my code, simply download all setup.py, plots.py, and mpmcmc.py and run mpmcmc.py.

## ideas

I'm keeping a running tab of all my crazy ideas for improving and applying redshift posteriors.  The document in this directory also contains the comprehensive background I'm putting together to prepare for my qualifying exam.
