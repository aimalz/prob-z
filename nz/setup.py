import timeit
import numpy as np
import os
import cPickle
import time
import pyfits

#generate number of galaxies to draw
#for consistency, must have more than one survey size
seed_ngals = [10,20,40]#2*np.arange(1,6)#[1,10]#can generate for different survey sizes
nsurvs = len(seed_ngals)
survnos = range(0,nsurvs)
nsamps = 1#instantiations of the survey, more than 1 breaks some things...
sampnos = range(0,nsamps)
#ngals = [nsamps*[seed_ngals[s]] for s in survnos]
#for poisson sampling instead of set survey size -- small number tests fail when sample size is 0!
ngals = [[np.random.poisson(seed_ngals[s]) for n in sampnos] for s in survnos]

if os.path.isfile('topdir.p'):
  topdir = cPickle.load(open('topdir.p','r'))
else:
  topdir = 'test'+str(round(time.time()))
  cPickle.dump(topdir,open('topdir.p','w'))
  os.makedirs(topdir)

#set up data structure
topdirs = [[topdir+'/'+str(seed_ngals[s])+'/'+str(n+1)+'-'+str(ngals[s][n])+'/' for n in sampnos] for s in survnos]
for tds in topdirs:
    for td in tds:
        if not os.path.exists(td):
            os.makedirs(td)

#set up for probability distributions
#all p(z) share these bins from Sheldon, et al.
#zbins = pyfits.open(os.path.join('../big-data/','zbins-12.fits'))
zbins = pyfits.open("http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits")

#reformat bins
nbins = len(zbins[1].data)
binnos = range(0,nbins)
zlos = [zbins[1].data[i][0] for i in binnos]
zhis = [zbins[1].data[i][1] for i in binnos]

#use centers of bins for plotting
zmids = [(zbins[1].data[i][0]+zbins[1].data[i][1])/2. for i in binnos]
zavg = sum(zmids)/nbins

#useful for plotting
zdifs = [zbins[1].data[i][1]-zbins[1].data[i][0] for i in binnos]
zdif = sum(zdifs)/nbins

import math as m
import random
import bisect
import sys

colors='rgbymc'
ncolors = len(colors)
colornos = range(0,ncolors)

#set true value of N(z)=theta
#tuples of form z_center, spread,magnitude
realistic_prep = [(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in zmids] for tup in realistic_prep])
realistic = [sum(realistic_comps[k]) for k in binnos]
realsum = sum(realistic)
realistic_pdf = np.array([realistic[k]/realsum/zdifs[k] for k in binnos])

plotrealistic = [sum(real) for real in realistic_comps]
plotrealisticsum = sum(plotrealistic)
plotrealistic_comps = np.transpose([[r/plotrealisticsum for r in real] for real in realistic_comps])
plotrealistic_pdf = np.array([plotrealistic[k]/plotrealisticsum for k in binnos])

trueNz = [seed_ngals[s]*realistic_pdf for s in survnos]
truePz = [realistic_pdf for s in survnos]
logtrueNz = [[m.log(max(trueNz[s][k],sys.float_info.epsilon)) for k in binnos] for s in survnos]
logtruePz = [[m.log(max(truePz[s][k],sys.float_info.epsilon)) for k in binnos] for s in survnos]

##set up for random selection of galaxies per bin
#def cdf(weights):
#    tot = sum(weights)
#    result = []
#    cumsum = 0
#    for w in weights:
#        cumsum += w
#        result.append(cumsum / tot)
#    return result
#def choice(pop, weights):
#    assert len(pop) == len(weights)
#    cdf_vals = cdf(weights)
#    x = random.random()
#    index = bisect.bisect(cdf_vals, x)
#    return pop[index]

##randomly select bin counts from true N(z)
#bincounts = []
#for s in survnos:
#    bincount = []
#    for n in sampnos:
#        count = [0]*nbins
#        for i in range(0,ngals[s][n]):
#            count[choice(binnos, trueNz[s])] += 1
#            bincount.append(count)
#    bincounts.append(bincount)
#
#plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
#logplotcounts = np.log(np.array(plotcounts))

#test case: all galaxies in survey have same true redshift
chosenbin = np.argmax(plotrealistic_pdf)#random.sample(range(0,35),1)
chosenbins = chosenbin*nsurvs
bincounts = []
for s in survnos:
    bincount = []
    for n in sampnos:
        count = [0]*nbins
        count[chosenbin] = ngals[s][n]
        bincount.append(np.array(count))
    bincounts.append(np.array(bincount))
bincounts = np.array(bincounts)

#plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
#logplotcounts = np.log(np.array(plotcounts))

sampPz = np.array([[bincounts[s][n]/ngals[s][n]/zdif for n in sampnos] for s in survnos])
logsampPz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampPz]))

sampNz = np.array([[bincounts[s][n]/zdif for n in sampnos] for s in survnos])
logsampNz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampNz]))

#define flat distribution for N(z)
avgprob = 1./nbins/zdif
logavgprob = m.log(avgprob)
flat = [seed*avgprob for seed in seed_ngals]
logflat = [m.log(f) for f in flat]
flatNz = [np.array([f]*nbins) for f in flat]
logflatNz = [np.array([lf]*nbins) for lf in logflat]
flatPz = [np.array([avgprob]*nbins) for s in survnos]
logflatPz = np.log(flatPz)#[np.array([avg_prob]*nbins) for s in survnos]

#turn bin numbers into redshifts for histogram later
idealZs = np.array([[[zmids[k] for k in binnos] for j in range(0,int(round(trueNz[s][k])))] for s in survnos])

#assign actual redshifts uniformly within each bin
trueZs = np.array([[[random.uniform(zlos[k],zhis[k]) for k in binnos for j in range(0,bincounts[s][n][k])] for n in sampnos] for s in survnos])

##test case: all galaxies have same true redshift
#trueZs = np.array([[[zmids[k] for k in binnos for j in range(0,bincounts[s][n][k])] for n in sampnos] for s in survnos])

#jitter zs to simulate inaccuracy
sigZs = np.array([[[zdif*(trueZs[s][n][j]+1.) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])#zdif*(trueZs+1.)
shiftZs = np.array([[[random.gauss(trueZs[s][n][j],sigZs[s][n][j]) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])

##test case: perfect observations
#shiftZs = trueZs
#sigZs = np.array([[[zdif*(trueZs[s][n][j]+1.) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])

#write out the data into a "catalog"
#broken re: nsamps>1
for s in survnos:
    for n in sampnos:
        catfile = open(topdirs[s][n]+'datacat.p','w')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        catdat = zip(shiftZs[s],sigZs[s])
        cPickle.dump(catdat,catfile)
        catfile.close()

import scipy as sp
from scipy import stats
import random

#read in the data from a catalog
#broken re: nsamps>1
obsZs,obserrs = [],[]
for s in survnos:
    for n in sampnos:
        catfile = open(topdirs[s][n]+'datacat.p','r')
        catdat = cPickle.load(catfile)
        obsZ,obserr = zip(*catdat)
        obsZs.append(obsZ)
        obserrs.append(obserr)
        catfile.close()

#define additional bins for points thrown out of Sheldon range by shifting
minshift = min([min([min(shift) for shift in shiftZ]) for shiftZ in shiftZs])
maxshift = max([max([max(shift) for shift in shiftZ]) for shiftZ in shiftZs])

binfront = [min(zlos)+x*zdif for x in range(int(m.floor((minshift-min(zlos))/zdif)),0)]
binback = [max(zhis)+x*zdif for x in range(1,int(m.ceil((maxshift-max(zhis))/zdif)))]
binends = np.array(binfront+sorted(set(zlos+zhis))+binback)
binlos = binends[:-1]
binhis = binends[1:]
new_nbins = len(binends)-1
new_binnos = range(0,new_nbins)
binmids = [(binends[k]+binends[k+1])/2. for k in new_binnos]

#define true N(z),P(z) for plotting given number of galaxies
full_trueNz = [np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(trueNz[s],np.array([sys.float_info.epsilon]*len(binback))))) for s in survnos]
full_logtrueNz = [[m.log(full_trueNz[s][k]) for k in new_binnos] for s in survnos]
full_truePz = [np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(truePz[s],np.array([sys.float_info.epsilon]*len(binback))))) for s in survnos]
full_logtruePz = [[m.log(full_truePz[s][k]) for k in new_binnos] for s in survnos]

#define flat N(z),P(z)
full_flatNz = [np.array([f]*new_nbins) for f in flat]
full_logflatNz = [np.array([lf]*new_nbins) for lf in logflat]
full_flatPz = [np.array([avgprob]*new_nbins) for s in survnos]
full_logflatPz = [np.array([logavgprob]*new_nbins) for s in survnos]

#bin the true and observed samples
sampNz = []
obsNz = []
for s in survnos:
    samp = []
    obs = []
    for n in sampnos:
        samphist = [sys.float_info.epsilon]*len(binmids)
        obshist = [sys.float_info.epsilon]*len(binmids)
        for j in range(0,ngals[s][n]):
            for k in new_binnos:
                if binends[k]<=trueZs[s][n][j] and trueZs[s][n][j]<binends[k+1]:
                    samphist[k]+=1./zdif
                if binends[k]<=shiftZs[s][n][j] and shiftZs[s][n][j]<binends[k+1]:
                    obshist[k]+=1./zdif
        samp.append(samphist)
        obs.append(obshist)
    sampNz.append(samp)
    obsNz.append(obs)
sampNz = np.array(sampNz)
obsNz = np.array(obsNz)
obsPz = np.array([[obsNz[s][n]/ngals[s][n] for n in sampnos] for s in survnos])

logsampNz = np.log(sampNz)#)[[[m.log(trueNz[s][n][k]) for k in new_binnos] for n in range(0,ndraws)]
logobsNz = np.log(obsNz)#[[m.log(shiftNz[s][n][k]) for k in new_binnos] for n in range(0,ndraws)]
logobsPz = np.log(obsPz)

#calculate mean and rms errors (over many draws) on bin heights

avgsamp,avgobs,rmssamp,rmsobs = [],[],[],[]

for s in survnos:
    samp = np.transpose(sampNz[s])
    obs = np.transpose(obsNz[s])
    avgsamp.append([np.mean(z) for z in samp])
    avgobs.append([np.mean(z) for z in obs])
    rmssamp.append([np.sqrt(np.mean(np.square(z*zdif))) for z in samp])
    rmsobs.append([np.sqrt(np.mean(np.square(z*zdif))) for z in obs])

avgsamp,avgobs,rmssamp,rmsobs = np.array(avgsamp),np.array(avgobs),np.array(rmssamp),np.array(rmsobs)
maxsamp,minsamp = avgsamp+rmssamp,avgsamp-rmssamp
maxobs,minobs = avgobs+rmsobs,avgobs-rmsobs

#generate gaussian likelihood function per galaxy per sample per survey to simulate imprecision
logpobs,pobs = [],[]
#n = rando#for one draw when this step is slow
for s in survnos:
    logpob,pob = [],[]
    for n in sampnos:
        lps,ps = [],[]
        for j in range(0,ngals[s][n]):
            func = sp.stats.norm(loc=obsZs[s][n][j],scale=obserrs[s][n][j])
            lo = np.array([max(sys.float_info.epsilon,func.cdf(binends[k])) for k in new_binnos])
            hi = np.array([max(sys.float_info.epsilon,func.cdf(binends[k+1])) for k in new_binnos])
            spread = (hi-lo)/zdif
            #normalize probabilities to integrate (not sum)) to 1
            summed = sum(spread)
            p = spread/summed/zdif
            logp = [m.log(max(p_i,sys.float_info.epsilon)) for p_i in p]
            lps.append(logp)
            ps.append(p)
        pob.append(ps)
        logpob.append(lps)
    pobs.append(pob)
    logpobs.append(logpob)
pobs = np.array(pobs)
logpobs = np.array(logpobs)

#permit varying number of parameters for testing
ndim = new_nbins
ndims = [ndim]#5*np.arange(0,7)+5
nlens = len(ndims)
lenno = 0#set parameter dimensions for now
#survno = 0#one survey size for now
dimnos = range(0,ndims[lenno])

#thin the chain
howmany = 25

#how many walkers
nwalkers = 2*new_nbins
walknos = range(0,nwalkers)

#set up number of iterations
maxiters = int(5e3)#int(1e4)#[seed_ngals[s]*1e3 for s in survnos]#int(5e3)
miniters = int(1e3)#[maxiters/seed_ngals[s] for s in survnos]#
nruns = maxiters/miniters

runnos = range(0,nruns)
plot_iters = [(r+1)*miniters for r in runnos]
#plot_x = [r*miniters for r in range(1,nruns+1)]
#plot_x_all = np.arange(0,maxiters)
iters_all = maxiters/howmany
plot_iters_all = howmany*np.arange(0,iters_all)
iters_each = miniters/howmany
range_iters_each = np.arange(0,iters_each)
plot_iters_each = howmany*range_iters_each
plot_iters_ranges = [plot_iters_all[r*iters_each:(r+1)*iters_each] for r in runnos]#[(r+1)*plot_iters_each for r in runnos]
randsamps = random.sample(walknos,ncolors)

import emcee
import StringIO
import hickle as hkl

#generate prior distribution for each survey size
logmus = [full_logflatNz[s][:ndims[lenno]] for s in survnos]
#logmus = [full_logflatPz[s][:ndims[lenno]] for s in survnos]

#MVN prior as class
class mvn(object):
    def __init__(self,mean,cov):
        self.dims = len(mean)
        self.mean = mean
        self.cov = cov
        assert np.all(sp.linalg.eigh(self.cov)[0] >= 0.)
        self.icov = np.linalg.pinv(self.cov,rcond = sys.float_info.epsilon)
        #assert np.all(sp.linalg.eigh(self.icov)[0] >= 0.)
        (self.logdetsign,self.logdet) =  np.linalg.slogdet(self.cov)
        #self.logdet = np.log(-1.*np.linalg.det(self.cov))
        #assert self.logdetsign >= 0.
    def logpdf(self,x):
        delta = x-self.mean
        c = np.dot(delta, np.dot(self.icov, delta))
        prob = -0.5*c
        return prob
    def sample_ps(self,W):
        #outsamp = [sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov) for n in range(0,N)]
        outsamp = [thing for thing in np.array(np.random.multivariate_normal(self.mean,self.cov,W))]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp#,sampprobs
    def sample_gm(self,W):
        #rando = sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov)
        #sigma = 100.#[np.median(self.cov[i]) for i in range(0,dims)]
        outsamp = [self.mean+np.random.randn(self.dims) for w in range(0,W)]#[rando+np.random.randn(dims) for n in range(0,N)]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp#,sampprobs
    def sample_gs(self,W):
        #rando = sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov)
        rando = np.random.multivariate_normal(self.mean,self.cov)
        #rando = self.sample_dumb(1)[0]
        #sigma = 100.#[np.median(self.cov[i]) for i in range(0,dims)]
        outsamp = [rando+np.random.randn(self.dims) for w in range(0,W)]#[rando+np.random.randn(dims) for n in range(0,N)]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp,rando#,sampprobs,rando

#this covariance can produce N(z) for realistic data
# q=1.#0.5
# e=0.1/zdif**2
# tiny=q*1e-6
# covmats = [np.array([[q*m.exp(-0.5*e*(zmids[a]-zmids[b])**2.) for a in range(0,ndim)] for b in range(0,ndim)])+tiny*np.identity(ndim) for ndim in ndims]
# priordists = [mvn(logmus[s],covmats[lenno]) for s in survnos]

#this covariance can produce N(z) for the test data
covmats = [np.identity(n) for n in ndims]
priordists = [mvn(logmus[s],covmats[lenno]) for s in survnos]

#generate initial values for walkers
iguesses_ps = [priordists[s].sample_ps(nwalkers) for s in survnos]#[0]
iguesses_gm = [priordists[s].sample_gm(nwalkers) for s in survnos]#[0]
gs_guesses = [priordists[s].sample_gs(nwalkers) for s in survnos]#[0]
iguesses_gs,randos_gs = [guess[0] for guess in gs_guesses],[guess[1] for guess in gs_guesses]
iguesses = [[iguesses_ps[s],iguesses_gm[s],iguesses_gs[s]] for s in survnos]
means = [[logmus[s][0:ndims[lenno]],logmus[s][0:ndims[lenno]],randos_gs[s][0:ndims[lenno]]] for s in survnos]
setups = ['Prior Samples','Gaussian Ball Around Mean','Gaussian Ball Around Prior Sample']
ntests = len(setups)
testnos = range(0,ntests)
setdirs = ['ps/','gm/','gs/']
inpaths = [[[topdirs[s][n]+setdirs[t] for t in testnos] for n in sampnos] for s in survnos]

#posterior distribution we want to sample as class
class post(object):
    def __init__(self,idist,xvals,yprobs):#data are logged posteriors (nsamps*nbins), idist is mvn object
        self.prior = idist
        self.priormean = idist.mean
        self.xgrid = xvals
#        self.ndims = len(xvals)-1
#        self.dims = range(0,self.ndims)
        self.postprobs = yprobs
        self.ndats = len(yprobs)
#        self.lndats = np.log(self.ndats)
        self.dats = range(0,self.ndats)
        self.difs = self.xgrid[1:]-self.xgrid[:-1]#np.array([self.xgrid[k+1]-self.xgrid[k] for k in self.dims])
        self.lndifs = np.log(self.difs)
        self.constterm = self.lndifs-self.priormean
    def priorprob(self,theta):#this is proportional to log probability
        return self.prior.logpdf(theta)
    def lnprob(self,theta):#speed this up some more with matrix magic?
        constterms = theta+self.constterm
        sumterm = self.priorprob(theta)-np.dot(np.exp(theta),self.difs)
        #assert (sumterm <= 0.), ('theta='+str(theta)+', lnprob='+str(sumterm))
        for j in self.dats:
            #logterm = sp.misc.logsumexp(self.postprobs[j]+constterms)#shockingly slower!
            #logterm = np.logaddexp(self.postprobs[j]+constterms)#only works for two terms
            logterm = np.log(np.sum(np.exp(self.postprobs[j]+constterms)))
            sumterm += logterm
        #have been getting positive lnprob values (i.e. probabilities>1), get reasonable samples if capped at 0 but still investigating
        #assert (sumterm <= 0.), ('theta='+str(theta)+', lnprob='+str(sumterm))
        #in run from which plots were generated, the following was uncommented!
        #if sumterm <= 0.:
        #    return sumterm
        #else:
        #    return 0.
        return sumterm

#set posterior distributions we want to sample
posts = [[post(priordists[s],binends,logpobs[s][n]) for n in sampnos] for s in survnos]
#create the samplers
samplers = [[emcee.EnsembleSampler(nwalkers,ndims[lenno],posts[s][n].lnprob) for n in sampnos] for s in survnos]

#set up directory structure
outdirs = ['times','fracs','probs','chains']
nstats = len(outdirs)
statnos = range(0,nstats)
outpaths = [[[[inpaths[s][n][t]+outdirs[i] for i in statnos] for t in testnos] for n in sampnos] for s in survnos]
for s in outpaths:
    for n in s:
        for t in n:
            for i in t:
                if not os.path.exists(i):
                    os.makedirs(i)
filenames = [str(x)+'.h' for x in plot_iters]
outnames = [[[[[os.path.join(outpaths[s][n][t][i],filenames[r]) for r in runnos] for i in statnos] for t in testnos] for n in sampnos] for s in survnos]

calctime = os.path.join(topdir,'calctimer.txt')
plottime = os.path.join(topdir,'plottimer.txt')

#import matplotlib.pyplot as plt

print 'SETUP DONE'
