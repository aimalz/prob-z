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
        return max(sys.float_info.epsilon,self.phi(self.hi)-self.phi(self.lo))

    def pdf(self,z):
        x = self.loc(z)
        pdf = sp.stats.norm.pdf(x)
        return pdf/(self.sig*self.norm())

    def cdf(self,z):
        x = self.loc(z)
        cdf = self.phi(x)-self.phi(self.lo)
        result = cdf/self.norm()
        #print('a cdf: {}/{}'.format(result,z))
        return result

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
        self.weights = self.weights/sum(self.weights)
#         mincomps = [(self.minZ-comp[0])/comp[1] for comp in self.comps]
#         maxcomps = [(self.maxZ-comp[0])/comp[1] for comp in self.comps]
        self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]#[sp.stats.truncnorm(mincomps[c],maxcomps[c],loc=self.comps[c][0],scale=self.comps[c][1]) for c in lrange(self.comps)]
#         self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]
#         self.weights = np.array([self.calccdf(c,self.minZ,self.maxZ) for c in lrange(self.comps)])

    def pdfs(self,zs):
        print('zs.shape={}'.format(np.shape(zs)))
        out = np.array([self.weights[c]*self.comps[c].pdf(zs) for c in xrange(self.ncomps)])
        print('pdfs.out.shape={}'.format(np.shape(out)))
        return out

    def pdf(self,zs):
        out = np.sum(self.pdfs(zs),axis=0)
        print('pdf.out.shape={}'.format(np.shape(out)))
        return out

    def cdfs(self,zs):
        out = np.array([self.weights[c]*self.comps[c].cdf(zs) for c in xrange(self.ncomps)])
        return out

    def cdf(self,zs):
        out = np.sum(self.cdfs(zs),axis=0)
        return out

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

class cont(object):
    """
    cont object takes a numpy array of normalized discrete distribution and its range and enables computation of PDF
    """
    def __init__(self,inarr,bounds):

        self.ndim = len(inarr)
        self.Zs = bounds
        self.difs = self.Zs[1:]-self.Zs[:-1]
        self.weights = inarr/np.dot(inarr,self.difs)
#         mincomps = [(self.minZ-comp[0])/comp[1] for comp in self.comps]
#         maxcomps = [(self.maxZ-comp[0])/comp[1] for comp in self.comps]
        self.dims = [uniform(loc=self.Zs[k],scale=self.difs[k]) for k in xrange(self.ndim)]#[sp.stats.truncnorm(mincomps[c],maxcomps[c],loc=self.comps[c][0],scale=self.comps[c][1]) for c in lrange(self.comps)]
#         self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]
#         self.weights = np.array([self.calccdf(c,self.minZ,self.maxZ) for c in lrange(self.comps)])

    def pdf(self,zs):
        out = np.array([self.weights[k]*np.array([self.dims[k].pdf(z) for z in zs]) for k in xrange(self.ndim)])
        return out

    def cdf(self,zs):
        out = np.array([self.weights[k]*np.array([self.dims[k].cdf(z) for z in zs]) for k in xrange(self.ndim)])
        return out

    def sample(self,N):
        choices = [0]*self.ndim
        for j in xrange(N):
            choices[choice(xrange(self.ndim), self.weights)] += 1
        samps = np.array([])
        for k in xrange(self.ndim):
            j = choices[k]
            Zs = self.dims[k].rvs(j)
            samps = np.concatenate((samps,Zs))
        return np.array(samps)

def makelf(truZ,zfactor,elements,outlier=None):#,dgen=None):

    if outlier is None:
        outlier = []

    mixmod = [[truZ+elem.shift,elem.stddev*zfactor,elem.weight] for elem in elements]
    mixmod.extend([[elem.obsZ,elem.stddev,elem.weight] for elem in outlier])

    lf = mixmod

    return(lf)#,dgen)

def makepdf(grid,truZ,gal,intp=None,dgen=None,outlier=None):

    elements = gal.elements

    zfactor = gal.makezfactor(truZ)

    difs = grid[1:]-grid[:-1]
    dif = difs[np.argmin(grid-truZ)]
    allsummed = np.zeros(len(grid)-1)

    lf = makelf(truZ,zfactor,elements,outlier=outlier)#,dgen)

    pdf = gmix(lf,(min(grid),max(grid)))

    if dgen != None:
        dgdist = gmix(dgen,(min(grid),max(grid)))
        const = dgdist.pdf(truZ)*dif
    else:
        const = 0.

    cdf = pdf.cdf(grid)
    spread = cdf[1:]-cdf[:-1]
    allsummed += spread
    allsummed += const
    if intp != None:
        pf = intp*allsummed
    else:
        pf = allsummed
    pf = np.array(pf)
    #pf = pf/max(np.dot(pf,difs),sys.float_info.epsilon)

    return(pf)
