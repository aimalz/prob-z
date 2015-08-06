import os
import random
import bisect
import cPickle
import numpy as np
import sys
import math as m
import scipy as sp
from scipy import stats
import emcee

def cdf(weights):
    tot = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
      cumsum += w
      result.append(cumsum/tot)
    return result

def choice(pop, weights):
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]

#posterior distribution we will want to sample as class
class post(object):

    def __init__(self,idist,xvals,yprobs):#data are logged posteriors (ngals*nbins), idist is mvn object
        self.prior = idist
        self.priormean = idist.mean
        self.xgrid = np.array(xvals)
        self.difs = self.xgrid[1:]-self.xgrid[:-1]#np.array([self.xgrid[k+1]-self.xgrid[k] for k in self.dims])
        self.lndifs = np.log(self.difs)#np.array([m.log(max(self.difs[k],sys.float_info.epsilon)) for k in self.dims])
        self.ndims = len(self.difs)
        self.dims = range(0,self.ndims)
        self.postprobs = yprobs
        self.ndats = len(yprobs)
#        self.lndats = np.log(self.ndats)
        self.dats = range(0,self.ndats)
        #print('difs'+str(self.difs))
        #print(self.lndifs)
        #print(self.priormean)
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

class persamp(object):

  def __init__(self,meta,p_run,s_run,n):

    self.p = p_run.p
    self.s = s_run.s
    self.n = n
    #sample some number of galaxies
    self.nsamps = meta.samps
    self.ngals = s_run.seed#nsamps*[seed]
   #for poisson sampling instead of set survey size -- small number tests fail when sample size is 0!
   #ngals = np.random.poisson(seed)#[[np.random.poisson(seed) for n in sampnos] for s in survnos]
    self.galnos = range(0,self.ngals)
    #print([j for j in self.galnos])

    self.topdir_n = s_run.topdir_s+'/'+str(self.n)+'-'+str(self.ngals)
    if not os.path.exists(self.topdir_n):
      os.makedirs(self.topdir_n)

    def maketrue(self):

      if os.path.exists(os.path.join(self.topdir_n,'true.p')):
        truefile = open(os.path.join(self.topdir_n,'true.p'),'rb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        [count,(sampPz,logsampPz),(sampNz,logsampNz)] = cPickle.load(truefile)
        truefile.close()
      else:
      ##randomly select bin counts from true N(z)
      # count = [0]*nbins
      # for i in range(0,ngals):
      #     count[choice(binnos, truePz)] += 1
      #
      #plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
      #logplotcounts = np.log(np.array(plotcounts))

      #test case: all galaxies in survey have same true redshift
        chosenbin = np.argmax(p_run.truePz)#choice(binnos, realistic_pdf)#random.sample(range(0,35),1)#np.argmax(plotrealistic_pdf)
        count = [0]*p_run.ndims
        count[chosenbin] = self.ngals
        count = np.array(count)

      #plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
      #logplotcounts = np.log(np.array(plotcounts))

        sampNz = count/meta.zdif
        logsampNz = np.log(np.array([max(o,sys.float_info.epsilon) for o in sampNz]))

      #sampNz = np.array([[bincounts[s][n]/zdif for n in sampnos] for s in survnos])
      #logsampNz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampNz]))

        sampPz = sampNz/self.ngals#count/ngal/zdif
        logsampPz = np.log(np.array([max(o,sys.float_info.epsilon) for o in sampPz]))

      #sampPz = np.array([[bincounts[s][n]/ngals[s][n]/zdif for n in sampnos] for s in survnos])
      #logsampPz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampPz]))

      #for s in survnos:
      #  for n in sampnos:
        truefile = open(os.path.join(self.topdir_n,'true.p'),'wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        cPickle.dump([count,(sampPz,logsampPz),(sampNz,logsampNz)],truefile)
        truefile.close()

      return([count,(sampPz,logsampPz),(sampNz,logsampNz)])

    [self.count,(self.sampPz,self.logsampPz),(self.sampNz,self.logsampNz)] = maketrue(self)

    #  #turn bin numbers into redshifts for histogram later
      #  idealZs = np.array([[zmids[k] for k in binnos] for j in range(0,int(round(trueNz[k])))])

      #assign actual redshifts uniformly within each bin
      #trueZs = ([random.uniform(zlos[k],zhis[k]) for k in binnos for j in range(0,count[k])])

      #test case: all galaxies have same true redshift
    self.trueZs = np.array([p_run.zmids[k] for k in p_run.dimnos for j in range(0,self.count[k])])

    #generate the catalog of individual galaxy posteriors
    def makecat(self):

      if os.path.exists(os.path.join(self.topdir_n,'datacat.p')):
        catfile = open(os.path.join(self.topdir_n,'datacat.p'),'rb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        [shiftZs,sigZs] = cPickle.load(catfile)
        catfile.close()
      else:

      #jitter zs to simulate inaccuracy
        modZs = self.trueZs+1.#[[trueZs[s][n]+1. for n in sampnos] for s in survnos]
        varZs = [meta.zdif*modZs[j] for j in self.galnos]# for n in sampnos] for s in survnos])#zdif*(trueZs+1.)
        shiftZs = np.array([random.gauss(self.trueZs[j],varZs[j]) for j in self.galnos])
        sigZs = np.array([abs(random.gauss(varZs[j],varZs[j])) for j in self.galnos])
      #print(modZs,varZs,sigZs)
      #pdb.set_trace()

      #test case: perfect observations
      #shiftZs = trueZs
      #sigZs = np.array([[[zdif*(trueZs[s][n][j]+1.) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])

      #write out the data into a "catalog"
      #broken re: nsamps>1
        catfile = open(os.path.join(self.topdir_n,'datacat.p'),'wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        cPickle.dump([shiftZs,sigZs],catfile)
        catfile.close()

      return(shiftZs,sigZs)

    self.obsZs,self.obserrs = makecat(self)

    def setup_pdfs(self):
    #redefine bins
      minobs = min(self.obsZs)
      maxobs = max(self.obsZs)
      binfront = np.array([min(p_run.zlos)+x*meta.zdif for x in range(int(m.floor((minobs-min(p_run.zlos))/meta.zdif)),0)])
      binback = np.array([max(p_run.zhis)+x*meta.zdif for x in range(1,int(m.ceil((maxobs-max(p_run.zhis))/meta.zdif)))])
      binends = np.unique(np.concatenate((binfront,p_run.allzs,binback),axis=0))
      #print('binends:'+str(binends))
      binlos = binends[:-1]
      binhis = binends[1:]
      nbins = len(binends)-1
      binnos = range(0,nbins)
      binmids = (binhis+binlos)/2.#[(binends[k]+binends[k+1])/2. for k in binnos]

      return([binfront,binback,binends,binlos,binhis,nbins,binnos,binmids])

    self.binfront,self.binback,self.binends,self.binlos,self.binhis,self.nbins,self.binnos,self.binmids = setup_pdfs(self)

    def makepdfs(self):

    #generate gaussian likelihood function per galaxy per sample per survey to simulate imprecision
    #simultaneously generate sheldon "posterior"
      pobs = []
      logpobs = []
      obshist = [sys.float_info.epsilon]*self.nbins
      for j in self.galnos:
        func = sp.stats.norm(loc=self.obsZs[j],scale=self.obserrs[j])
        lo = np.array([max(sys.float_info.epsilon,func.cdf(self.binends[k])) for k in self.binnos])
        hi = np.array([max(sys.float_info.epsilon,func.cdf(self.binends[k+1])) for k in self.binnos])
        spread = (hi-lo)#/meta.zdif,sys.float_info.epsilon)
      #normalize probabilities to integrate (not sum)) to 1
        summed = max(sum(spread),sys.float_info.epsilon)
        pob = spread/summed/meta.zdif
        logpob = [m.log(max(p_i,sys.float_info.epsilon)) for p_i in pob]
        logpobs.append(logpob)
        pobs.append(pob)
        for k in self.binnos:
          if self.obsZs[j]>self.binends[k] and self.obsZs[j]<self.binends[k+1]:
            obshist[k]+=1.
      pobs = np.array(pobs)
      logpobs = np.array(logpobs)
      stackprep = np.sum(np.array(pobs),axis=0)
      stack = np.array([max(sys.float_info.epsilon,stackprep[k]) for k in self.binnos])
      logstack = np.log(stack)
      obshist = np.array(obshist)
      logobshist = np.log(obshist)

      return([pobs,logpobs],[stack,logstack],[obshist,logobshist])

    [self.pobs,self.logpobs],[self.stack,self.logstack],[self.obshist,self.logobshist] = makepdfs(self)

    def makesummary(self):

      #define true N(z),P(z) for plotting given number of galaxies
      full_trueNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),s_run.trueNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
      full_logtrueNz = np.log(full_trueNz)
      #full_truePz = [np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(truePz[s],np.array([sys.float_info.epsilon]*len(binback))))) for s in survnos]
      #full_logtruePz = [[m.log(full_truePz[s][k]) for k in binnos] for s in survnos]

      #define flat N(z),P(z) for plotting
      full_flatNz = np.array([1./meta.zdif/self.nbins]*self.nbins)#[np.array([p_run.avgprob]*nbins) for f in flat]
      full_logflatNz = np.log(full_flatNz)#[np.array([lf]*nbins) for lf in logflat]
      #full_flatPz = [np.array([avgprob]*nbins) for s in survnos]
      #full_logflatPz = [np.array([logavgprob]*nbins) for s in survnos]
      #print(full_logflatNz)

      #define sampled N(z),P(z) for plotting
      full_sampNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.sampNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
      #full_logsampNz = np.log(full_sampNz)
      full_logsampNz = np.concatenate((np.array([m.log(sys.float_info.epsilon)]*len(self.binfront)),self.logsampNz,np.array([m.log(sys.float_info.epsilon)]*len(self.binback))),axis=0)
      #full_sampPz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(sampPz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]
      #full_logsampPz = [[np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(logsampPz[s][n],np.array([sys.float_info.epsilon]*len(binback))))) for n in sampnos] for s in survnos]#[[np.log(full_sampPz[s][n]) for n in sampnos] for s in survnos]

      return([(full_trueNz,full_logtrueNz),(full_flatNz,full_logflatNz),(full_sampNz,full_logsampNz)])

    [(self.full_trueNz,self.full_logtrueNz),(self.full_flatNz,self.full_logflatNz),(self.full_sampNz,self.full_logsampNz)] = makesummary(self)

    #print(np.shape(self.full_logflatNz))

    self.priordist = meta.mvn(self.full_logflatNz,np.identity(self.nbins))

    # q=1.#0.5
    # e=0.1/p_run.zdif**2
    # tiny=q*1e-6
    # covmat = np.array([[q*m.exp(-0.5*e*(self.binmids[a]-self.binmids[b])**2.) for a in range(0,p_run.nbins)] for b in range(0,p_run.nbins)])+tiny*np.identity(p_run.nbins) for ndim in ndims]
    # priordist = meta.mvn(self.full_logflatNz,covmat)

    #how many walkers
    self.nwalkers = 2*self.nbins
    self.walknos = range(0,self.nwalkers)

    self.postdist = post(self.priordist,self.binends,self.logpobs)

    self.sampler = emcee.EnsembleSampler(self.nwalkers,p_run.ndims,self.postdist.lnprob)

