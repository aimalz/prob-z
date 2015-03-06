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

#useful for plotting
zdifs = [zbins[1].data[i][1]-zbins[1].data[i][0] for i in range(0,nbins)]

# <codecell>

import math as m
import random

#set prior on N(z)

avg_prob = 1./nbins
#flat mean on prior
priorNz = np.array([avg_prob]*nbins)
logpriorNz = [m.log(avg_prob)]*nbins
logvars = [0.]+[avg_prob/n for n in range(1,nbins)]
#pseudo covariance matrix here
covN = np.array([[logvars[abs(i-j)] for j in range(0,nbins)] for i in range(0,nbins)])

# <codecell>

#want to calculate pdfs with improper covariance matrix

import scipy.sparse
import scipy.sparse.linalg as la

def lognormpdf(vec,mu,cov):
    ndim = len(cov)
    norm_coeff = ndim*m.log(2*m.pi)+np.linalg.slogdet(cov)[1]
    err = vec-mu
    if (sp.sparse.issparse(cov)):
        numerator = sp.sparse.la.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)
    return -0.5*(norm_coeff+numerator)

# <codecell>

import sys
import numpy as np
import scipy as sp
from scipy import stats

#sample p(z|N(z)) given prior

def gensamp(mu,cov):#input logprior
    #attempt = #np.random.multivariate_normal(mu,cov)#can't use this with pseudo covariance matrix
    attempt = sp.stats.multivariate_normal.rvs(mean=mu,cov=cov)#alternatively, list(np.random.multivariate_normal(mu,cov))
    prenorm = [m.exp(x) for x in attempt]# if m.exp(x)>0. else sys.float_info.epsilon for x in attempt]
    summed = sum(prenorm)
    normed = [x/summed for x in prenorm]
    return normed#output non-logged sample

# <codecell>

#plot some samples of the prior, i.e. possible N(z)

sample6 = [gensamp(logpriorNz,covN) for i in range(0,6)]
samptups6 = [[(sample6[i][j],sample6[i][j]) for j in range(0,nbins)] for i in range(0,6)]

import matplotlib.pyplot as plt
colors = "bgrcmy"

plt.figure(1)
plt.title('Prior Samples')
plt.rc('text',usetex=True)
for i in range(0,6):
    for j in range(0,nbins):
        plt.step(zmids,sample6[i],color=colors[i])
plt.step(zmids,priorNz,linewidth=2.,color='k')
plt.ylabel(r'$p(z|\vec{\mathcal{N}})$')
plt.xlabel(r'$z$')
plt.savefig('log-prior-samps.png')
plt.close()

# <codecell>

#random selection of galaxies per bin

import bisect
import collections

def cdf(weights):
    tot = sum(weights)
    ans = []
    allsum = 0
    for w in weights:
        allsum += w
        ans.append(allsum/tot)
    return ans

def choice(pop, weights):
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    randval = random.random()
    item = bisect.bisect(cdf_vals, randval)
    return population[item]

ngals = 10000#arbitrary

trueNz = gensamp(logpriorNz,covN)
inprobs=trueNz
redshifts = range(0,nbins)
bins = collections.defaultdict(int)
for n in range(0,ngals):
    bins[choice(redshifts, inprobs)] += 1

draws = [bins[k] for k in bins]

# <codecell>

#assign actual redshifts uniformly within each bin
truezs = []
for k in range(0,nbins):
    ndraws = draws[k]
    zdraws = [random.uniform(zlos[k],zhis[k]) for j in range(0,ndraws)]
    truezs = truezs+zdraws

# <codecell>

from scipy.stats import norm
#jitter center of z distribution
normeddif = [zdif/nbins for zdif in zdifs]
sigZ = sum(zdifs)/nbins
shiftZ = [truez+random.gauss(0.,sigZ) for truez in truezs]# = truezs#to turn off shifting
#generate gaussian likelihood per galaxy
sigP = [sigZ*m.sqrt(truez) for truez in truezs]
spreadZ = [[max(sys.float_info.epsilon,norm.pdf(zmids[k],loc=shiftZ[n],scale=sigP[n])) for k in range(0,nbins)] for n in range(0,ngals)]
spreadsum = [sum(spread) for spread in spreadZ]
spreadZs = [[spreadZ[n][k]/spreadsum[n] for k in range(0,nbins)] for n in range(0,ngals)]
#noisify gaussians
sigE = [sigZ*truez**2 for truez in truezs]
noiseZ = [[max(sys.float_info.epsilon,spreadZs[n][k]+random.gauss(0,sigE[n])) for k in range(0,nbins)] for n in range(0,ngals)]
noisedsum = [sum(noise) for noise in noiseZ]
noiseZs = [[noiseZ[n][k]/noisedsum[n] for k in range(0,nbins)] for n in range(0,ngals)]
pobs = noiseZs#spreadZs#to turn off noisification of gaussian distributions
logpobs = [[m.log(val) for val in noiseZ] for noiseZ in noiseZs]

# <codecell>

#visualize some of the p(d|z) distributions
chosen  = random.sample(range(0,ngals),7)

plt.figure(2)
plt.rc('text', usetex=True)
plt.title('Galaxy Redshift Likelihood Functions')
for i in chosen:
    plt.step(zmids,pobs[i],label='galaxy '+str(i)+' with z='+str(truezs[i]))
plt.ylabel(r'$p(\vec{d}_{n}|z)$')
plt.xlabel(r'$z$')
plt.legend(fontsize='x-small')
plt.savefig('rando-lik-samps.png')
plt.close()

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
    #priordist = #sp.stats.multivariate_normal(mean=priorNz,cov=covN)#can't use this with improper covariance matrix
    outprod = lognormpdf(theta,priorNz,covN)#priordist.pdf(theta)
    return outprod

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
first = gensamp(logpriorNz,covN)
init = priorNz

howmany = ngals#because it's slow, would ideally set a threshold on precision or number of accepted samples

old = init
new = first
previous = product(old)
dist = []
ratios = []

# <codecell>

#metropolis-hastings, here we go!
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
pool = len(unique)
nplots = int(m.log10(howmany))

plt.figure(3,figsize=(2*nplots,2*nplots))
plt.rc('text', usetex=True)
plt.suptitle('Posterior Samples')
for p in range(0,nplots):
    plt.subplot(nplots,1,p+1)
    plt.ylim(min(trueNz)-0.01,max(trueNz)+0.01)
    plt.step(zmids,trueNz,linewidth = 2,color='k',label=r'true $\vec{\mathcal{N}}$')
    for i in unique[int(m.floor(p*pool/nplots)):int(m.ceil((p+1)*pool/nplots))]:
        plt.step(zmids,dist[i],label='proposal '+str(i)+' with r='+str(ratios[i]))
    plt.ylabel(r'$p(z|\vec{\mathcal{N}})$')
    plt.xlabel(r'$z$')
    plt.legend(fontsize='xx-small')
plt.savefig('mcmc-results.png')
plt.close()

# <codecell>

#take average of accepted distributions
flipped = np.transpose([dist[i] for i in unique])
avgnz = [sum(flipped[k])/pool for k in range(0,nbins)]

# <codecell>

#calculate posteriors for each galaxy
posts = [[pobs[n][k]*avgnz[k] for k in range(0,nbins)] for n in range(0,ngals)]

# <codecell>

#implement sheldon method on actual posteriors
sheldon_prep = np.transpose(posts)
sheldon_prenorm = [sum(line) for line in sheldon_prep]
sheldon_summed = sum(sheldon_prenorm)
sheldon_normed = [s/sheldon_summed for s in sheldon_prenorm]

# <codecell>

#calculate sum of squares
sheldonx2 = sum([(sheldon_normed[k]-trueNz[k])**2 for k in range(0,nbins)])
statx2 = sum([(avgnz[k]-trueNz[k])**2 for k in range(0,nbins)])

# <codecell>

plt.figure(4)
plt.rc('text', usetex=True)
plt.title('Comparison of Methods')
plt.step(zmids,trueNz,c='b',label=r'True $\mathcal{N}(z)$')
plt.step(zmids,sheldon_normed,c='r',label=r'Sheldon Approach with SSQ='+str(sheldonx2))
plt.step(zmids,avgnz,c='g',label=r'Average Posterior with SSQ='+str(statx2))
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathcal{N}(z)$')
plt.legend(fontsize='x-small')
plt.savefig('compare-sheldon.png')
plt.close()

