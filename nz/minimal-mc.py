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
zdif = sum(zdifs)/nbins

# <codecell>

import math as m
import random
import numpy as np

#set true value of N(z)=theta

avg_prob = 1./nbins
realistic = [zmid**max(zlos)*m.exp(-1*zmid*min(zhis)) for zmid in zmids]
priorNz = np.array([r/sum(realistic) for r in realistic])#np.array([avg_prob]*nbins)
logpriorNz = np.array([m.log(prior) for prior in priorNz])
logvars = [1./m.sqrt(nbins)]+[1./nbins]+[0.]*(nbins-2)
covN = np.array([[logvars[abs(i-j)] for j in range(0,nbins)] for i in range(0,nbins)])

# <codecell>

#sample theta=p(z|N(z)) given prior value
import sys
import scipy as sp
from scipy import stats

def gensamp(mu,cov):#input logprior
    attempt = sp.stats.multivariate_normal.rvs(mean=mu,cov=cov)#alternatively, list(np.random.multivariate_normal(mu,cov))
    prenorm = [m.exp(x) for x in attempt]
    summed = sum(prenorm)
    normed = [x/summed for x in prenorm]
    logged = [m.log(x) for x in normed]
    return logged

def genprob(mu,cov):#input logprior
    logged = gensamp(mu,cov)
    probs = [m.exp(log) for log in logged]
    return probs

# <codecell>

#plot samples of prior
sample6 = [genprob(logpriorNz,covN) for i in range(0,6)]
samptups6 = [[(sample6[i][j],sample6[i][j]) for j in range(0,nbins)] for i in range(0,6)]

import matplotlib.pyplot as plt
colors = "bgrcmy"

plt.figure(1)
plt.title('Prior Samples')
plt.rc('text', usetex=True)
plt.semilogy()
for i in range(0,6):
    for j in range(0,nbins):
        plt.step(zmids,sample6[i],color=colors[i],linewidth=0.75)
plt.step(zmids,priorNz,linewidth=2.,color='k')
plt.ylabel(r'$\ln[p(z|\vec{\mathcal{N}})]$')
plt.xlabel(r'$z$')
plt.show()
#plt.savefig('real-prior-samps.png')
plt.close()

# <codecell>

#random selection of galaxies per bin
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

ngals = 1000#arbitrary
truelogNz = gensamp(logpriorNz,covN)
trueNz = genprob(logpriorNz,covN)
weights=trueNz
population = range(0,nbins)
counts = [1]*nbins
for n in range(ngals-nbins):
    counts[choice(population, weights)] += 1

# <codecell>

#assign actual redshifts uniformly within each bin
trueZs = []
for k in range(0,nbins):
    ndraws = counts[k]
    zdraws = [random.uniform(zlos[k],zhis[k]) for j in range(0,ndraws)]
    trueZs = trueZs+zdraws

# <codecell>

#generate redshift likelihoods
from scipy.stats import norm

#jitter zs to simulate inaccuracy
normeddif = [zdif/nbins for zdif in zdifs]
sigZ = sum(zdifs)/nbins
shiftZ = [true+random.gauss(0.,sigZ) for true in trueZs]
#generate gaussian likelihood per galaxy to simulate imprecision
sigP = [m.sqrt(sigZ)*true**2 for true in trueZs]
spreadZ = [[max(sys.float_info.epsilon,norm.pdf(zmids[k],loc=shiftZ[n],scale=max(sigP[n],sigZ))) for k in range(0,nbins)] for n in range(0,ngals)]
spreadsum = [sum(spread) for spread in spreadZ]
spreadZs = [[spreadZ[n][k]/spreadsum[n] for k in range(0,nbins)] for n in range(0,ngals)]

pobs = spreadZs
logpobs = [[m.log(val) for val in pob] for pob in pobs]

# <codecell>

#plot sample
zdif = sum(zdifs)/nbins
binfront = [min(zlos)+x*zdif for x in range(int(m.floor((min(shiftZ)-min(zlos))/zdif)),0)]
binback = [max(zhis)+x*zdif for x in range(0,int(m.ceil((max(shiftZ)-max(zhis))/zdif)))]
binends = binfront+sorted(set(zlos+zhis))+binback
histnz = [true*ngals for true in trueNz]

plt.figure(2)
plt.rc('text', usetex=True)
plt.title('Simulated Redshift Distribution')
plt.yscale('log', nonposy='clip')
plt.hist(trueZs,sorted(set(zlos+zhis)),alpha = 0.35,label='true redshifts')
plt.hist(shiftZ,binends,alpha = 0.35,label='observed redshifts')
plt.step(zmids,histnz,linewidth = 2,label=r'true $\mathcal{N}(z)$')
plt.xlabel('redshift')
plt.ylabel('frequency')
plt.legend(loc='upper left')
#plt.savefig('inputs.png')
plt.show()
plt.close()

# <codecell>

#visualize some of the p(d|z) distributions
chosen = [int((x/7.)*ngals) for x in range(0,7)]#random.sample(range(0,ngals),7)

plt.figure(3)
plt.rc('text', usetex=True)
plt.title('Galaxy Redshift Likelihood Functions')
for i in chosen:
    plt.step(zmids,pobs[i],label='galaxy '+str(i)+' with z='+str(trueZs[i]))
plt.ylabel(r'$p(\vec{d}_{n}|z)$')
plt.xlabel(r'$z$')
plt.legend(fontsize='x-small')
#plt.savefig('lik-samps.png')
plt.show()
plt.close()

# <codecell>

#log likelihood for data set given loghistogram heights theta
def loglik(theta):
    probtheta = np.array([m.exp(t) for t in theta])
    outprod = 0.
    for n in range(0,ngals):
        outsum = 0.
        for k in range(0,nbins):
            term = pobs[n][k]*probtheta[k]
            outsum +=term
        outprod += m.log(outsum)
    return outprod

# <codecell>

#log prior probabilities given log histogram heights theta
def logprior(theta):
    priordist = sp.stats.multivariate_normal(mean=logpriorNz,cov=covN)
    outprod = m.log(priordist.pdf(theta))
    return outprod

# <codecell>

#helper functions for the MH algorithm

def product(inlist):
    outlist = loglik(inlist)+logprior(inlist)
    return (inlist,outlist)

def compare(proposed,previous):
    return proposed[1]-previous[1]

# <codecell>

#initialize
first = gensamp(logpriorNz,covN)
init = [m.log(avg_prob)]*nbins#logpriorNz

howmany = ngals*10#because it's slow, would ideally set another threshold

old = init
new = first
previous = product(old)
dist = []
ratios = []
#metropolis-hastings, here we go!

# <codecell>

import time

acced = 0
pool = 0
while len(dist) < howmany:
    proposed = product(new)
    r = compare(proposed,previous)
    ratios.append(r)
    if r >= 0.:
        previous = proposed
        pool +=1
        if pool%10==0:
            print 'at sample '+str(len(dist))+' accepted '+str(pool)+' by merit at '+str(time.time()) 
        #print 'sample '+str(len(dist))+' accepted with r='+str(r)
    else: 
        rando = random.uniform(0.,1.)
        if rando < m.exp(r):
            previous = proposed
            acced += 1
            if acced%10==0:
                print 'at sample '+str(len(dist))+' accepted '+str(acced)+' by chance at '+str(time.time()) 
            #print 'sample '+str(len(dist))+' accepted with r='+str(r)
    dist.append(tuple(previous[0]))
    new = gensamp(previous[0],covN)

# <codecell>

#plot accepted proposals

unique = [u for u in np.where(np.array(ratios)>0.)[0]]
probdists = [[m.exp(d) for d in dist[u]] for u in unique]
#pool = len(unique)
nplots = int(m.log(howmany,7))
truehistprep, obshistprep = [0.]*(len(binends)-1),[0.]*(len(binends)-1)
for i in range(0,ngals):
    for k in range(0,len(binends)-1):
        if shiftZ[i]>binends[k] and shiftZ[i]<binends[k+1]:
            obshistprep[k] += 1./ngals
        if trueZs[i]>binends[k] and trueZs[i]<binends[k+1]:
            truehistprep[k] += 1./ngals
binmids = [(binends[k]+binends[k+1])/2. for k in range(0,len(binends)-1)]

plt.figure(4,figsize=(3*nplots,2*nplots))
plt.rc('text', usetex=True)
plt.suptitle('Posterior Samples')
for p in range(0,nplots):
    plt.subplot(nplots,1,p+1)
    plt.semilogy()
    plt.xlim(min(binends),1.2)
    plt.step(zmids,trueNz,color='k', linestyle=':',label=r'true $\vec{\mathcal{N}}$')
    plt.step(binmids,truehistprep,color='k',linestyle='--',label='true redshift distribution')
    plt.step(binmids,obshistprep,color='k',linewidth=2,label='observed redshift distribution')
    for u in range(int(m.floor(p*pool/nplots)),int(m.ceil((p+1)*pool/nplots))):
        plt.step(zmids,probdists[u],label='proposal '+str(unique[u])+' with r='+str(round(ratios[unique[u]],2)))
    plt.ylabel(r'$p(z|\vec{\mathcal{N}})$')
    plt.xlabel(r'$z$')
    plt.legend(fontsize='xx-small',bbox_to_anchor=(1.12, 1.0))
#plt.savefig('mcmc-results.png')
plt.show()
plt.close()

# <codecell>

#implement Sheldon method and calculate pseudo-chi squared
flipped = np.transpose([dist[unique[i]] for i in range(0,pool)])
avgnz = [sum(flipped[k])/pool for k in range(0,nbins)]

posts = [[[pobs[n][k]*probdists[j][k] for k in range(0,nbins)] for n in range(0,ngals)] for j in range(0,pool)]

sheldon_prep = [np.transpose(post) for post in posts]
sheldon_prenorm = [[sum(line) for line in thing] for thing in sheldon_prep]
sheldon_summed = [sum(thing) for thing in sheldon_prenorm]
sheldon_normed = [[sheldon_prenorm[j][k]/sheldon_summed[j] for k in range(0,nbins)] for j in range(0,pool)]
sheldon_logged = [[m.log(s) for s in thing] for thing in sheldon_normed]

logobshistprep = [m.log(max(x,sys.float_info.epsilon)) for x in obshistprep]
allsheldonx2 = [sum([(thing[k]-logobshistprep[k])**2 for k in range(0,nbins)]) for thing in sheldon_logged]
allstatx2 = [sum([(dist[u][k]-logobshistprep[k])**2 for k in range(0,nbins)]) for u in unique]

meanx2 = sum(allstatx2)/pool
meansheldonx2 = sum(allsheldonx2)/pool

# <codecell>

plt.figure(5)
plt.rc('text', usetex=True)
plt.title('Comparison of Methods')
plt.semilogy()
for u in range(0,pool):
        plt.step(zmids,probdists[u],c='b')
        plt.step(zmids,sheldon_normed[u],c='r')
plt.plot(zmids,[0.]*nbins,c='b',label=r'accepted posterior samples with $\bar{\chi^{2}}=$'+str(meanx2))
plt.plot(zmids,[0.]*nbins,c='r',label=r'Sheldon posterior estimates with $\bar{\chi^{2}}=$'+str(meansheldonx2))
plt.step(zmids,trueNz,c='k',linestyle = ':',linewidth=2,label=r'True $\mathcal{N}(z)$')
plt.step(binmids,obshistprep,color='k',linewidth=2,label='observable redshift distribution')
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathcal{N}(z)$')
plt.legend(fontsize='small',loc='lower right')
plt.savefig('compare-sheldon.png')
#plt.show()
plt.close()

# <codecell>


