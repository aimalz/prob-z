import timeit

start_time = timeit.default_timer()

import pdb
#pdb.set_trace()
import numpy as np
import os
import cPickle
import pyfits

#generate number of galaxies to draw
#for consistency, must have more than one survey size
seed_ngals = [2,20]#2*np.arange(1,6)#[1,10]#can generate for different survey sizes
nsurvs = len(seed_ngals)
survnos = range(0,nsurvs)
nsamps = 1#instantiations of the survey, more than 1 breaks some things...
sampnos = range(0,nsamps)
ngals = [nsamps*[seed_ngals[s]] for s in survnos]
#for poisson sampling instead of set survey size -- small number tests fail when sample size is 0!
#ngals = [[np.random.poisson(seed_ngals[s]) for n in sampnos] for s in survnos]

if os.path.isfile('topdir.p'):
  topdir = cPickle.load(open('topdir.p','rb'))
else:
  topdir = 'test'+str(round(timeit.default_timer()))
  cPickle.dump(topdir,open('topdir.p','wb'))
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
allzs = sorted(set(zlos+zhis))

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

#generate an instantiation of N(z)
def maketrue(nsurvs,nsamps):
  #set up for random selection of galaxies per bin
  def cdf(weights):
   tot = sum(weights)
   result = []
   cumsum = 0
   for w in weights:
       cumsum += w
       result.append(cumsum / tot)
   return result
  def choice(pop, weights):
   assert len(pop) == len(weights)
   cdf_vals = cdf(weights)
   x = random.random()
   index = bisect.bisect(cdf_vals, x)
   return pop[index]

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
  chosenbin = np.argmax(plotrealistic_pdf)#choice(binnos, realistic_pdf)#random.sample(range(0,35),1)#np.argmax(plotrealistic_pdf)
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

  for s in survnos:
    for n in sampnos:
        truefile = open(topdirs[s][n]+'true.p','wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        cPickle.dump([bincounts[s][n],(sampPz[s][n],logsampPz[s][n]),(sampNz[s][n],logsampNz[s][n])],truefile)
        truefile.close()

  return(bincounts,(sampPz,logsampPz),(sampNz,logsampNz))

#generate sampled truth and save it for later
if not np.all([[os.path.exists(topdirs[s][n]+'true.p') for n in sampnos] for s in survnos]):
  bincounts,(sampPz,logsampPz),(sampNz,logsampNz) = maketrue(nsurvs,nsamps)
else:#if truth already exists, use that
  bincounts,sampPz,logsampPz,sampNz,logsampNz = [],[],[],[],[]
  for s in survnos:
      bincount,sampP,logsampP,sampN,logsampN = [],[],[],[],[]
      for n in sampnos:
        truefile = open(topdirs[s][n]+'true.p','rb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        [b,(sP,lsP),(sN,lsN)] = cPickle.load(truefile)
        bincount.append(b)
        sampP.append(sP)
        logsampP.append(lsP)
        sampN.append(sN)
        logsampN.append(lsN)
        truefile.close()
      bincounts.append(bincount)
      sampPz.append(sampP)
      logsampPz.append(logsampP)
      sampNz.append(sampN)
      logsampNz.append(logsampN)

#define flat distribution for N(z)
avgprob = 1./nbins/zdif
logavgprob = m.log(avgprob)
flat = [seed*avgprob for seed in seed_ngals]
logflat = [m.log(f) for f in flat]
flatNz = [np.array([f]*nbins) for f in flat]
logflatNz = [np.array([lf]*nbins) for lf in logflat]
flatPz = [np.array([avgprob]*nbins) for s in survnos]
logflatPz = np.log(flatPz)#[np.array([avg_prob]*nbins) for s in survnos]

#generate the catalog of individual galaxy posteriors
def makecat(bincounts):

#  #turn bin numbers into redshifts for histogram later
#  idealZs = np.array([[[zmids[k] for k in binnos] for j in range(0,int(round(trueNz[s][k])))] for s in survnos])

  #assign actual redshifts uniformly within each bin
  #trueZs = np.array([[np.array([random.uniform(zlos[k],zhis[k]) for k in binnos for j in range(0,bincounts[s][n][k])]) for n in sampnos] for s in survnos])

  #test case: all galaxies have same true redshift
  trueZs = np.array([[np.array([zmids[k] for k in binnos for j in range(0,bincounts[s][n][k])]) for n in sampnos] for s in survnos])

  #jitter zs to simulate inaccuracy
  modZs = [[trueZs[s][n]+1. for n in sampnos] for s in survnos]
  varZs = np.array([[[zdif*modZs[s][n][j] for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])#zdif*(trueZs+1.)
  shiftZs = np.array([[[random.gauss(trueZs[s][n][j],varZs[s][n][j]) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])
  sigZs = np.array([[[abs(random.gauss(varZs[s][n][j],varZs[s][n][j])) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])
  #print(modZs,varZs,sigZs)
  #pdb.set_trace()

  #test case: perfect observations
  #shiftZs = trueZs
  #sigZs = np.array([[[zdif*(trueZs[s][n][j]+1.) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])

  #write out the data into a "catalog"
  #broken re: nsamps>1
  for s in survnos:
    for n in sampnos:
        catfile = open(topdirs[s][n]+'datacat.p','wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        catdat = zip(shiftZs[s][n],sigZs[s][n])
        cPickle.dump(catdat,catfile)
        catfile.close()

  return(shiftZs,sigZs)

#"observe" nsamps samples from nsurvs surveys
if not np.all([[os.path.exists(topdirs[s][n]+'datacat.p') for n in sampnos] for s in survnos]):
   obsZs,obserrs = makecat(bincounts)
else:#if truth already exists, use that
  obsZs,obserrs = [],[]
  for s in survnos:
    obsZ,obserr = [],[]
    for n in sampnos:
        catfile = open(topdirs[s][n]+'datacat.p','rb')
        catdat = cPickle.load(catfile)
        oZ,oerr = zip(*catdat)
        obsZ.append(oZ)
        obserr.append(oerr)
        catfile.close()
    obsZs.append(obsZ)
    obserrs.append(obserr)

import scipy as sp
from scipy import stats
import random

#print(obsZs)
#pdb.set_trace()

#define additional bins for points thrown out of Sheldon range by shifting
minobs = min([min([min(obs) for obs in obsZ]) for obsZ in obsZs])
maxobs = max([max([max(obs) for obs in obsZ]) for obsZ in obsZs])

binfront = [min(zlos)+x*zdif for x in range(int(m.floor((minobs-min(zlos))/zdif)),0)]
binback = [max(zhis)+x*zdif for x in range(1,int(m.ceil((maxobs-max(zhis))/zdif)))]
binends = np.array(binfront+sorted(set(zlos+zhis))+binback)
binlos = binends[:-1]
binhis = binends[1:]
new_nbins = len(binends)-1
new_binnos = range(0,new_nbins)
binmids = (binhis+binlos)/2.#[(binends[k]+binends[k+1])/2. for k in new_binnos]

#define true N(z),P(z) for plotting given number of galaxies
full_trueNz = [np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(trueNz[s],np.array([sys.float_info.epsilon]*len(binback))))) for s in survnos]
full_logtrueNz = [[m.log(full_trueNz[s][k]) for k in new_binnos] for s in survnos]
full_truePz = [np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(truePz[s],np.array([sys.float_info.epsilon]*len(binback))))) for s in survnos]
full_logtruePz = [[m.log(full_truePz[s][k]) for k in new_binnos] for s in survnos]

#define flat N(z),P(z) for plotting
full_flatNz = [np.array([f]*new_nbins) for f in flat]
full_logflatNz = [np.array([lf]*new_nbins) for lf in logflat]
full_flatPz = [np.array([avgprob]*new_nbins) for s in survnos]
full_logflatPz = [np.array([logavgprob]*new_nbins) for s in survnos]

#define sampled N(z),P(z) for plotting
full_sampNz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(sampNz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]
full_logsampNz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(logsampNz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]#[[np.log(full_sampNz[s][n]) for n in sampnos] for s in survnos]
full_sampPz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(sampPz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]
full_logsampPz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(logsampPz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]#[[np.log(full_sampPz[s][n]) for n in sampnos] for s in survnos]

#bin "observed" (point estimate) N(z)
histNz = []
for s in survnos:
    histN = []
    for n in sampnos:
        hist = [sys.float_info.epsilon]*len(binmids)
        for j in range(0,ngals[s][n]):
            for k in new_binnos:
                if binends[k]<=obsZs[s][n][j] and obsZs[s][n][j]<binends[k+1]:
                    hist[k]+=1./zdif
        histN.append(hist)
    histNz.append(histN)
histNz = np.array(histNz)
loghistNz = np.log(histNz)
histPz = histNz/ngals
loghistPz = np.log(histPz)

# #calculate mean and rms errors (over many draws) on bin heights
# avgsamp,avgobs,rmssamp,rmsobs = [],[],[],[]

# for s in survnos:
#     samp = np.transpose(sampNz[s])
#     hist = np.transpose(histNz[s])
#     avgsamp.append([np.mean(z) for z in samp])
#     avghist.append([np.mean(z) for z in hist])
#     rmssamp.append([np.sqrt(np.mean(np.square(z*zdif))) for z in samp])
#     rmshist.append([np.sqrt(np.mean(np.square(z*zdif))) for z in hist])

# avgsamp,avghist,rmssamp,rmshist = np.array(avgsamp),np.array(avghist),np.array(rmssamp),np.array(rmshist)
# maxsamp,minsamp = avgsamp+rmssamp,avgsamp-rmssamp
# maxhist,minhist = avghist+rmshist,avghist-rmshist

#generate gaussian likelihood function per galaxy per sample per survey to simulate imprecision
#simultaneously generate sheldon "posterior"
logpobs,pobs,sheldon = [],[],[]
for s in survnos:
    logpob,pob,sheldon_s = [],[],[]
    for n in sampnos:
        lps,ps= [],[]
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
        sheldonprep = np.sum(np.array(pob),axis=1)
        sheldon_n = [max(sys.float_info.epsilon,sheldonprep[0][k]) for k in new_binnos]
        sheldon_s.append(sheldon_n)
    pobs.append(pob)
    logpobs.append(logpob)
    sheldon.append(sheldon_s)
pobs = np.array(pobs)
logpobs = np.array(logpobs)
logsheldon = np.log(np.array(sheldon))
#print(np.shape(sheldon))#len(sheldon),len(sheldon[0]),len(sheldon[0][0]),len(sheldon[0][0][0]))
#print(np.shape(sheldon,logsheldon)
#pdb.set_trace()

#permit varying number of parameters for testing
ndim = new_nbins
ndims = [ndim]#5*np.arange(0,7)+5
nlens = len(ndims)
lenno = 0#set parameter dimensions for now
#survno = 0#one survey size for now
dimnos = range(0,ndims[lenno])

#prepare for MCMC

#how many walkers
nwalkers = 2*new_nbins
walknos = range(0,nwalkers)

#set up number of iterations
maxiters = int(3e3)#int(1e4)#[seed_ngals[s]*1e3 for s in survnos]#int(5e3)
miniters = int(1e3)#[maxiters/seed_ngals[s] for s in survnos]#
nruns = maxiters/miniters

#thin the chain
howmany = miniters/10#must be integer such that miniters mod howmany = 0

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
logmus = full_logflatNz#[full_logflatNz[s][:ndims[lenno]] for s in survnos]
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

# #this covariance can produce N(z) for realistic data
# q=1.#0.5
# e=0.1/zdif**2
# tiny=q*1e-6
# covmats = [np.array([[q*m.exp(-0.5*e*(binmids[a]-binmids[b])**2.) for a in range(0,ndim)] for b in range(0,ndim)])+tiny*np.identity(ndim) for ndim in ndims]
# priordists = [mvn(logmus[s],covmats[lenno]) for s in survnos]

#this covariance can produce N(z) for delta function test
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

# #delta function prior for test case
# priordists = [delta(logmus[s]) for s in survnos]

# #generate initial values for walkers in delta function case
# iguesses = [priordists[s].sample_ps(nwalkers) for s in survnos]
# iguesses = [[iguesses[s]] for s in survnos]
# means = [[logmus[s][0:ndims[lenno]]] for s in survnos]
# setups = ['Prior Samples']
# ntests = len(setups)
# testnos = range(0,ntests)
# setdirs = ['ps/']
# inpaths = [[[topdirs[s][n]+setdirs[t] for t in testnos] for n in sampnos] for s in survnos]

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
        sumterm = self.priorprob(theta)-np.dot(np.exp(theta),self.difs)#this should sufficiently penalize poor samples but somehow fails on large datasets
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
#fitness = [[os.path.join(topdirs[s][n],'fitness.txt') for n in sampnos] for s in survnos]
#allnames_prep = [os.path.join(topdirs[s][n],'fitness.txt') for n in sampnos for s in survnos]
#allnames_prep.append(calctime)
#allnames_prep.append(plottime)
for i in [calctime,plottime]:#allnames_prep:
  if os.path.exists(i):
    os.remove(i)

#import matplotlib.pyplot as plt
#ymin = np.log(sys.float_info.epsilon)
ymax = [np.log(seed_ngals[s]/zdif) for s in survnos]
ymax_e = np.exp(np.array(ymax))

elapsed = timeit.default_timer() - start_time
print 'setup complete: '+str(elapsed)
