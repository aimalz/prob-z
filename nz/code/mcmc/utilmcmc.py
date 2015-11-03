"""
util-mcmc module defines handy tools for MCMC
"""

import sys
import numpy as np
import statistics
import cPickle as cpkl

def lrange(l):
    """
    lrange(l) makes a range based on the length of a list or array l
    """
    return xrange(len(l))

class mvn(object):
    """
    mvn object is multivariate normal distribution, to be used in data generation and prior to emcee
    """
    def __init__(self, mean, cov):
        self.dims = len(mean)
        self.mean = mean
        self.cov = cov
        self.icov = np.linalg.pinv(self.cov, rcond=sys.float_info.epsilon)
        (self.logdetsign, self.logdet) = np.linalg.slogdet(self.cov)

    # log probabilities
    def logpdf(self, x):
        delta = x - self.mean
        c = np.dot(delta, np.dot(self.icov, delta))
        prob = -0.5 * c
        return prob

    # W samples directly from distribution
    def sample_ps(self, W):
        outsamp = np.random.multivariate_normal(self.mean, self.cov, W)
        return (outsamp, self.mean)

    # W samples around mean of distribution
    def sample_gm(self,W):
        outsamp = [self.mean+np.random.randn(self.dims) for w in range(0,W)]
        return (outsamp,self.mean)

    # W samples from a single sample from distribution
    def sample_gs(self, W):
        rando = np.random.multivariate_normal(self.mean, self.cov)
        outsamp = [rando + np.random.randn(self.dims) for w in range(0,W)]
        return (outsamp, rando)

class post(object):
    """
    post object is posterior distribution we wish to sample
    """
    def __init__(self,idist,xvals,yprobs,interim):#data are logged posteriors (ngals*nbins), idist is mvn object
        self.prior = idist
        #self.priormean = idist.mean
        self.interim = interim
        self.xgrid = np.array(xvals)
        self.difs = self.xgrid[1:]-self.xgrid[:-1]#np.array([self.xgrid[k+1]-self.xgrid[k] for k in self.dims])
        self.lndifs = np.log(self.difs)#np.array([m.log(max(self.difs[k],sys.float_info.epsilon)) for k in self.dims])
        self.postprobs = yprobs
        self.constterm = self.lndifs-self.interim#self.priormean

    # this is proportional to log probability
    def priorprob(self,theta):
        return self.prior.logpdf(theta)

    # calculate log probability
    # speed this up some more with matrix magic?
    def lnprob(self,theta):
        constterms = theta+self.constterm
        sumterm = self.priorprob(theta)-np.dot(np.exp(theta),self.difs)#this should sufficiently penalize poor samples but somehow fails on large datasets
        for j in lrange(self.postprobs):
            #logterm = sp.misc.logsumexp(self.postprobs[j]+constterms)#shockingly slower!
            #logterm = np.logaddexp(self.postprobs[j]+constterms)#only works for two terms
            logterm = np.log(np.sum(np.exp(self.postprobs[j]+constterms)))
            sumterm += logterm
        #have been getting positive lnprob values (i.e. probabilities>1), get reasonable samples if capped at 0 but still investigating
        #in run from which plots were generated, the following was uncommented!
        #if sumterm <= 0.:
        #    return sumterm
        #else:
        #    return 0.
        return sumterm

class path(object):
    """
    path object takes templates of path style and variables for it and makes os.path objects from them
    """
    def __init__(self, path_template, filled = None):
        self.path_template = path_template
        if filled is None:
            self.filled = {}
        else:
            self.filled = filled

    # actually constructs the final path, as a string.  Optionally takes in any missing parameters
    def construct(self, **args):
        nfilled = self.filled.copy()
        nfilled.update(args)
        return self.path_template.format(**nfilled)

    # fills any number of missing parameters, returns new object
    def fill(self, **args):
        dct = self.filled.copy()
        dct.update(args)
        return path(self.path_template, dct)
