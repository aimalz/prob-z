"""
util-sim module defines handy tools used in data generation
"""

import sys
import numpy as np
import scipy as sp
#import random
import bisect

np.random.seed(seed=0)

def lrange(x):
    """
    lrange makes a range based on the length of a list or array l
    """
    return xrange(len(x))

def safelog(xarr):
    """
    safelog takes log of array with zeroes
    """
    shape = np.shape(xarr)
    flat = xarr.flatten()
    logged = np.log(np.array([max(x,sys.float_info.epsilon) for x in flat]))
    return logged.reshape(shape)

def extend(arr,front,back):
    """
    extend appends zeroes to ends of array
    """
    return np.concatenate((np.array([sys.float_info.epsilon]*len(front)),arr,np.array([sys.float_info.epsilon]*len(back))),axis=0)

# tools for sampling an arbitrary discrete distribution, used in data generation
def cdf(weights):
    """
    cdf takes weights and makes them a normalized CDF
    """
    tot = sum(weights)
    result = []
    cumsum = 0.
    for w in weights:
      cumsum += w
      result.append(cumsum/tot)
    return result

def choice(pop, weights):
    """
    choice takes a population and assigns each element a value from 0 to len(weights) based on CDF of weights
    """
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = np.random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]

def normed(x,scale):
    """
    normed takes a numpy array and returns a normalized version of it that integrates to 1
    """
    x = np.array(x)
    scale = np.array(scale)
    norm = x/np.dot(x,scale)
    return norm

class tnorm(object):
    def __init__(self,mu,sig,ends):
        self.mu = mu
        self.sig = sig
        (self.min,self.max) = ends
        self.lo = self.loc(self.min)
        self.hi = self.loc(self.max)

    def loc(self,z):
        return (z-self.mu)/self.sig

    def phi(self,z):
        x = z/np.sqrt(2)
        term = sp.special.erf(x)
        return (1.+term)/2.

    def norm(self):
        return self.phi(self.hi)-self.phi(self.lo)

    def pdf(self,z):
        x = self.loc(z)
        pdf = sp.stats.norm.pdf(x)
        return pdf/(self.sig*self.norm())

    def cdf(self,z):
        x = self.loc(z)
        cdf = self.phi(x)-self.phi(self.lo)
        return cdf/self.norm()

    def rvs(self,J):
        func = sp.stats.truncnorm(self.lo,self.hi,loc=self.mu,scale=self.sig)
        return func.rvs(size=J)

class gmix(object):
    """
    gmix object takes a numpy array of Gaussian parameters and enables computation of PDF
    """
    def __init__(self,inarr,bounds):

        self.minZ,self.maxZ = bounds
        self.comps = inarr
        self.ncomps = len(self.comps)

        self.weights = np.transpose(self.comps)[2]
#         mincomps = [(self.minZ-comp[0])/comp[1] for comp in self.comps]
#         maxcomps = [(self.maxZ-comp[0])/comp[1] for comp in self.comps]
        self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]#[sp.stats.truncnorm(mincomps[c],maxcomps[c],loc=self.comps[c][0],scale=self.comps[c][1]) for c in lrange(self.comps)]
#         self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]
#         self.weights = np.array([self.calccdf(c,self.minZ,self.maxZ) for c in lrange(self.comps)])

    def pdfs(self,zs):
        out = np.array([self.weights[c]*np.array([self.comps[c].pdf(z) for z in zs]) for c in xrange(self.ncomps)])
        return out

    def pdf(self,zs):
        return np.sum(self.pdfs(zs),axis=0)

    def cdfs(self,zs):
        out = np.array([self.weights[c]*np.array([self.comps[c].cdf(z) for z in zs]) for c in xrange(self.ncomps)])
        return out

    def cdf(self,zs):
        return np.sum(self.cdfs(zs),axis=0)

    def binned(self,zs):
        thing = self.cdf(zs)
        return thing[1:]-thing[:-1]

    def sample(self,N):
        choices = [0]*self.ncomps
        for j in xrange(N):
            choices[choice(xrange(self.ncomps), self.weights)] += 1
        samps = np.array([])
        for c in xrange(self.ncomps):
            j = choices[c]
            Zs = self.comps[c].rvs(j)
            samps = np.concatenate((samps,Zs))
        return np.array(samps)
