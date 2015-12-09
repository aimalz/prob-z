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
    def __init__(self,inarr,bounds):

        self.minZ,self.maxZ = bounds
        self.comps = inarr
        self.ncomps = len(self.comps)

        self.weights = np.transpose(self.comps)[2]
        self.comps = [sp.stats.norm(loc=comp[0],scale=comp[1]) for comp in self.comps]
        sums = np.array([self.calccdf(c,self.minZ,self.maxZ) for c in lrange(self.comps)])
        self.weights = self.weights/sum(self.weights/sums)
        # self.weights = self.weights/sum(self.weights)

    def calcpdf(self,c,z):
        if z <= self.maxZ and z >= self.minZ:
            return self.weights[c]*self.comps[c].pdf(z)
        else:
            return sys.float_info.epsilon

    def calccdf(self,c,z1,z2):
        if z1>z2:
            z1,z2 = z2,z1
        if z2 > self.maxZ:
            z2 = self.maxZ
        if z1 < self.minZ:
            z1 = self.minZ
        return self.weights[c]*(self.comps[c].cdf(z2)-self.comps[c].cdf(z1))

    def pdfs(self,z):
        output = []
        for c in xrange(self.ncomps):
            output.append(self.calcpdf(c,z))
        return np.array(output)

    def cdfs(self,z1,z2):
        output = []
        for c in xrange(self.ncomps):
            output.append(self.calccdf(c,z1,z2))
        return np.array(output)

    def sumpdf(self,z):
        return sum(self.pdfs(z))

    def sumcdf(self,z1,z2):
        return sum(self.cdfs(z1,z2))

    def fullpdf(self,c,zs):
        output = [sys.float_info.epsilon]*len(zs)
        for z in lrange(zs):
            output[z] += self.calcpdf(c,zs[z])
        return np.array(output/sum(self.weights))

    def sumfullpdf(self,zs):
        output = np.array([sys.float_info.epsilon]*len(zs))
        for c in xrange(self.ncomps):
            output += self.weights[c]*self.fullpdf(c,zs)
        #output = np.transpose(np.array(output))
        return output#/np.dot(output,zdifs) #np.array([sum(z) for z in output])

    def sample(self,N):
        choices = [0]*self.ncomps
        for j in xrange(N):
            choices[choice(xrange(self.ncomps), self.weights)] += 1
        samps = np.array([])
        for c in xrange(self.ncomps):
            j = choices[c]
            Zs = self.comps[c].rvs(size=j)
            samps = np.concatenate((samps,Zs))
        return np.array(samps)

    def binned(self,zends):
        zdifs = zends[1:]-zends[:-1]
        output = [sys.float_info.epsilon]*(len(zdifs))
        for z in lrange(zdifs):
            output[z] += self.sumcdf(zends[z],zends[z+1])/zdifs[z]
        return np.array(output/np.dot(output,zdifs))
