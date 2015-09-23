# handy tools defined here

import sys
import numpy as np

# constructs a range for every element in a list
def lrange(l):
    return xrange(len(l))

# path object build paths given a template and a number of variables
class path(object):

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

# multivariate normal distribution object, used in data generation and emcee prior
class mvn(object):

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

# posterior distribution we will want to sample as class
class post(object):

    def __init__(self,idist,xvals,yprobs):#data are logged posteriors (ngals*nbins), idist is mvn object
        self.prior = idist
        self.priormean = idist.mean
        self.xgrid = np.array(xvals)
        self.difs = self.xgrid[1:]-self.xgrid[:-1]#np.array([self.xgrid[k+1]-self.xgrid[k] for k in self.dims])
        self.lndifs = np.log(self.difs)#np.array([m.log(max(self.difs[k],sys.float_info.epsilon)) for k in self.dims])
        self.postprobs = yprobs
        self.constterm = self.lndifs-self.priormean

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

# tools for sampling an arbitrary distribution, used in data generation
# turn weights into proper CDF
def cdf(weights):
    tot = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
      cumsum += w
      result.append(cumsum/tot)
    return result

# sample population given weights
def choice(pop, weights):
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]

# these are good ideas but I never used them
# def dadd(l, r):
#     return l.copy().update(r)

# def key_lookup(db, mask):
#     return lambda key: db[key.filter(mask)]

# def concatenate(ll):
#     return [x for x in l for l in ll]

# def curry(f, a):
#     return lambda(b): f(a,b)
