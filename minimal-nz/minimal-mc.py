# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#set up for probability distributions
import pyfits

#all p(z) share these bins
zbins = pyfits.open("http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits")

#reformat bins
zlos = [zbins[1].data[i][0] for i in range(0,len(zbins[1].data))]
zhis = [zbins[1].data[i][1] for i in range(0,len(zbins[1].data))]
nbins = len(zbins[1].data)

#use centers of bins for plotting
zmids = [(zbins[1].data[i][0]+zbins[1].data[i][1])/2. for i in range(0,nbins)]

#use differences for plotting
zdifs = [zbins[1].data[i][1]-zbins[1].data[i][0] for i in range(0,nbins)]

# <codecell>

import math as m
import random

#set true value of N(z)=theta
seedprior = random.uniform(0.,1.)
preprior = [seedprior]
while len(preprior)<nbins:
    nextprior = random.uniform(preprior[len(preprior)-1]-1,preprior[len(preprior)-1]+1)
    if nextprior >= 0.:
        preprior.append(nextprior)

altpriorNz = [m.exp(pre) for pre in preprior]
sumprior = sum(altpriorNz)
normpriorNz = [altpriorNz[k]/sumprior for k in range(0,nbins)]
priorNz = normpriorNz

# <codecell>

import sys
import numpy as np
import scipy as sp
from scipy import stats

#sample theta=p(z|N(z)) given prior value

def gensamp(mu,cov):
    attempt = sp.stats.multivariate_normal.rvs(mean=mu,cov=cov)#alternatively, list(np.random.multivariate_normal(mu,cov))
    prenorm = [x if x>0. else sys.float_info.epsilon for x in attempt]
    summed = sum(prenorm)
    normed = [y/summed for y in prenorm]
    return normed

sigN = 0.1/nbins
covN = [sigN**2.]*np.identity(nbins)
#covariance is arbitrary, but it really goes to hell if larger or smaller. . .

# <codecell>

import matplotlib.pyplot as plt

#plot some samples from the prior assuming a covariance
plt.figure(1)
plt.title('Prior Samples')
plt.plot(zmids,priorNz,linewidth = 3,color='k')
for i in range(0,7):
    plt.plot(zmids,gensamp(priorNz,covN))
plt.ylabel('probability')
plt.xlabel('redshift')
plt.savefig('rando-prior-samps.png')

# <codecell>

#random selection of galaxies per bin
#code taken wholesale from unutbu on StackOverflow

import bisect
import collections

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

ngals = 10000
weights=priorNz
population = range(0,nbins)
counts = collections.defaultdict(int)
for n in range(ngals):
    counts[choice(population, weights)] += 1

draws = [counts[k] for k in counts]

# <codecell>

#assign actual redshifts uniformly within each bin
truezs = []
for k in range(0,nbins):
    ndraws = draws[k]
    zdraws = [random.uniform(zlos[k],zhis[k]) for j in range(0,ndraws)]
    truezs = truezs+zdraws

# <codecell>

%%timeit from scipy.stats import norm
#jitter center of z distribution
normeddif = [zdif/nbins for zdif in zdifs]
sigZ = sum(zdifs)/nbins
shiftZ = [truez+random.gauss(0.,sigZ) for truez in truezs]
#shiftZ = truezs #to turn off shifting
#generate gaussian likelihood per galaxy
sigP = [sigZ*truez for truez in truezs]
spreadZ = [[max(sys.float_info.epsilon,norm.pdf(zmids[k],loc=shiftZ[n],scale=sigP[n])) for k in range(0,nbins)] for n in range(0,ngals)]
spreadsum = [sum(spread) for spread in spreadZ]
spreadZs = [[spreadZ[n][k]/spreadsum[n] for k in range(0,nbins)] for n in range(0,ngals)]
#noisify gaussians
sigE = [sigZ*shift**2 for shift in shiftZ]
noiseZ = [[max(sys.float_info.epsilon,spreadZs[n][k]+random.gauss(0,sigE[n])) for k in range(0,nbins)] for n in range(0,ngals)]
noisedsum = [sum(noise) for noise in noiseZ]
noiseZs = [[noiseZ[n][k]/noisedsum[n] for k in range(0,nbins)] for n in range(0,ngals)]
#to turn on/off noisification of gaussian distributions
pobs = noiseZs#spreadZs

# <codecell>


# <codecell>

#visualize some of the p(d|z) distributions

plt.figure(2)
plt.title('Galaxy Redshift Likelihood Functions')
for i in random.sample(range(0,10000),7):
    plt.plot(zmids,pobs[i],label=str(i))
plt.ylabel('probability')
plt.xlabel('redshift')
plt.legend()
plt.savefig('rando-likelihoods.png')

# <codecell>

#log likelihood for data set given histogram heights theta

def loglik(theta):
    outprod = 0.
    for n in range(0,ngals):
        outsum = 0.
        for k in range(0,nbins):
            term = pobs[n][k]*theta[k]
            outsum +=term
        outprod += m.log(outsum)
    return outprod

# <codecell>

#log prior probabilities

def logprior(theta):
    priordist = sp.stats.multivariate_normal(mean=priorNz,cov=covN)
    outprod = priordist.pdf(theta)
    if outprod > 0.:
        return m.log(outprod)
    else:
        return m.log(sys.float_info.epsilon)

# <codecell>

#helper functions for the MH algorithm

def product(inlist):
    outlist = loglik(inlist)+logprior(inlist)
    return (inlist,outlist)

def compare(proposed,previous):
    num = proposed[1]
    den = previous[1]
    return num-den

# <codecell>

#initialize
first = gensamp(priorNz,covN)
init = [1./nbins]*nbins

howmany = 1000#because it's slow, would ideally set another threshold

old = init
new = first
previous = product(old)
dist = []
ratios = []
#metropolis-hastings, here we go!

# <codecell>

while len(dist) < howmany:
    proposed = product(new)
    r = compare(proposed,previous)
    ratios.append(r)
    if r >= 0.:
        previous = proposed
        print 'sample '+str(len(dist))+' accepted with r='+str(r)
    else: 
        rando = random.uniform(0.,1.)
        if rando < m.exp(r):
            previous = proposed
            print 'sample '+str(len(dist))+' accepted with r='+str(r)
    dist.append(tuple(previous[0]))
    new = gensamp(previous[0],covN)

# <codecell>

#plot accepted proposals

unique = [u for u in np.where(np.array(ratios)>0.)[0]]
nplots = int(m.log10(howmany))

plt.figure(3,figsize=(nplots**2, 2*nplots))
plt.suptitle('Posterior Samples')
for p in range(0,nplots):
    plt.subplot(nplots,1,p+1)
    plt.ylim(min(priorNz)-0.01,max(priorNz)+0.01)
    plt.plot(zmids,priorNz,linewidth = 2,color='k',label='prior')
    for i in unique[int(m.floor(p*len(unique)/nplots)):int(m.ceil((p+1)*len(unique)/nplots))]:
        #if i < int(m.ceil((p+1)*len(unique)/nplots)) and i >= int(m.floor(p*len(unique)/nplots)):
        plt.plot(zmids,dist[i],label=str(i))
    plt.ylabel('N(z)')
    plt.xlabel('redshift')
    plt.legend(fontsize='x-small')
plt.savefig('mc-results.png')

