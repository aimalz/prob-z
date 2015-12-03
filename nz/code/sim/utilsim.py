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

# tools for sampling an arbitrary distribution, used in data generation
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
    norm = x/sum(x/scale)
    return (norm,safelog(norm))

class gmix(object):
    """
    gmix object takes a numpy array of Gaussian parameters and enables computation of PDF
    """
    def __init__(self,inarr):
        self.comps = inarr
        self.ncomps = len(self.comps)
        self.weights = np.transpose(self.comps)[2]

    def pdf(comp,z):
        return z*comp[2]*(2*np.pi*comp[1])**-0.5*np.exp(-1.*(z-comp[0])**2/(2.*comp[1]))

    def pdfs(self,z):
        output = []
        for comp in self.comps:
            output.append(self.pdf(comp,z))
        return np.array(output)

    def sumpdf(self,z):
        return sum(self.pdfs(z))

    def fullpdf(self,comp,zs):
        output = [sys.float_info.epsilon]*len(zs)
        for z in zs:
            output.append(pdf(comp,z))
        return np.array(output)

    def sumfullpdf(self,zs):
        output = []
        for comp in comps:
            output.append(self.fullpdf(comp,zs))
        output = np.transpose(np.array(output))
        return np.array([sum(z) for z in output])

    def sample(self,N):
        output = []
        for n in xrange(N):
            comp = choice(self.comps,self.weights)
            P = np.random.uniform(1)
            z = comp[0]+comp[1]*np.sqrt(2.)*sp.special.erfinv(2.*P-1.)
            output.append(z)
        return np.array(output)
