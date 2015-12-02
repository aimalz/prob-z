"""
per-sim module generates the data for one run
"""

import numpy as np
import sys
import math as m
import os
import scipy as sp
from scipy import stats
import csv
import itertools
import timeit

from insim import setup
from utilsim import *

class pertest(object):
    def __init__(self, meta):

        self.meta = meta

        # perparam differs from meta
        self.ndims = meta.params
        self.allzs = meta.allzs[:self.ndims+1]
        self.zlos = self.allzs[:-1]
        self.zhis = self.allzs[1:]
        self.zmids = (self.zlos+self.zhis)/2.
        self.zavg = sum(self.zmids)/self.ndims

        # define realistic underlying P(z) for this number of parameters
        self.realsum = sum(meta.realistic[:self.ndims])
        self.realistic_pdf = np.array([meta.realistic[k]/self.realsum/meta.zdifs[k] for k in xrange(0,self.ndims)])
        self.physPz = self.realistic_pdf
        self.logphysPz = np.array([m.log(max(pPz,sys.float_info.epsilon)) for pPz in self.physPz])

        # define flat P(z) for this number of parameters
        self.avgprob = 1./self.ndims/meta.zdif
        self.logavgprob = m.log(self.avgprob)
        self.flatPz = [self.avgprob]*self.ndims
        self.logflatPz = [self.logavgprob]*self.ndims

        # set true value of N(z) for this survey size
        self.seed = meta.survs
        self.physNz = self.seed*self.realistic_pdf
        self.logphysNz = [m.log(max(x,sys.float_info.epsilon)) for x in self.physNz]

        # define flat distribution for N(z)
        self.flat = self.seed*self.avgprob
        self.logflat = m.log(self.flat)
        self.flatNz = np.array([self.flat]*self.ndims)
        self.logflatNz = np.array([self.logflat]*self.ndims)

        self.choosen()
        self.choosebins()
        self.choosetrue()
        self.makedat()
        self.makecat()
        self.fillsummary()
        self.savedat()

        print(self.meta.name+' simulated data')

    def choosen(self):

        # sample some number of galaxies, poisson or set
        if self.meta.poisson == True:
            np.random.seed(seed=self.ndims)
            self.ngals = np.random.poisson(self.seed)
        else:
            self.ngals = self.seed
        self.randos = np.random.choice(self.ngals,len(self.meta.colors),replace=False)

    def choosebins(self):

        count = [0]*self.ndims

        #test all galaxies in survey have same true redshift vs. sample from physPz
        if self.meta.random == True:
            np.random.seed(seed=self.ndims)
            for j in range(0,self.ngals):
                count[choice(xrange(self.ndims), self.physPz)] += 1
        else:
            chosenbin = np.argmax(self.physPz)
            count[chosenbin] = self.ngals

        self.count = np.array(count)

        self.sampNz = self.count/self.meta.zdif
        self.logsampNz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampNz]))

        self.sampPz = self.sampNz/self.ngals
        self.logsampPz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampPz]))

    def choosetrue(self):

        # assign actual redshifts either uniformly or identically to mean
        if self.meta.uniform == True:
            np.random.seed(seed=self.ndims)
            self.trueZs = np.array([np.random.uniform(self.zlos[k],self.zhis[k]) for k in xrange(self.ndims) for j in xrange(self.count[k])])
        else:
            self.trueZs = np.array([self.zmids[k] for k in xrange(self.ndims) for j in xrange(self.count[k])])

    def makedat(self):

      # define 1+z and variance to use for sampling z
        modZs = self.trueZs+1.
        varZs = modZs*self.meta.zdif

        # we can re-calculate npeaks later from shiftZs or sigZs.
        if self.meta.shape == True:
            np.random.seed(seed=self.ndims)
            self.npeaks = np.array([np.random.randint(1,self.ndims-1) for j in xrange(self.ngals)])
        else:
            self.npeaks = [1]*self.ngals

        # jitter zs to simulate inaccuracy, choose variance randomly for eah peak
        np.random.seed(seed=self.ndims)
        shiftZs = np.array([[np.random.normal(loc=self.trueZs[j],scale=varZs[j]) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

        # standard deviation of peaks directly dependent on true redshift vs Gaussian
        if self.meta.sigma == True or self.meta.shape == True:
            np.random.seed(seed=self.ndims)
            sigZs = np.array([[max(sys.float_info.epsilon,np.random.normal(loc=varZs[j],scale=varZs[j])) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])
        else:
            sigZs = np.array([[varZs[j] for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

        self.minobs = min(min(shiftZs))
        self.maxobs = max(max(shiftZs))

        # for consistency
        self.obserrs = sigZs
        self.obsZs = shiftZs

    def setup_pdfs(self):

        self.binfront = np.array([])#np.array([min(self.zlos)+x*self.meta.zdif for x in range(int(m.floor((self.minobs-min(self.zlos))/self.meta.zdif)),0)])#np.array([])
        self.binback = np.array([])#np.array([max(self.zhis)+x*self.meta.zdif for x in range(1,int(m.ceil((self.maxobs-max(self.zhis))/self.meta.zdif)))])#np.array([])
        self.binends = np.unique(np.concatenate((self.binfront,self.allzs,self.binback),axis=0))
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.nbins = len(self.binends)-1
        self.binnos = range(0,self.nbins)
        self.binmids = (self.binhis+self.binlos)/2.
        self.bindifs = self.binhis-self.binlos
        self.bindif = sum(self.bindifs)/self.nbins

        #nontrivial interim prior
        if self.meta.interim == True:
            interim = sp.stats.poisson.pmf(xrange(self.nbins),2.0)
        else:
            #self.interim = float(self.ngals)/self.bindifs/self.nbins
            interim = self.logflatNz#self.realistic_pdf
            interim = np.array([sys.float_info.epsilon]*len(self.binfront)+list(interim)+[sys.float_info.epsilon]*len(self.binback))#np.array([z**2*np.exp(-(z/0.5)**1.5) for z in self.binmids])/self.bindifs
        suminterim = np.sum(interim)
        self.interim = float(self.ngals)*interim/suminterim/self.bindifs
        self.loginterim = np.log(self.interim)

    def makecat(self):

        self.setup_pdfs()

        pobs = []
        logpobs = []
        mapzs = []
        expzs = []

        for j in xrange(self.ngals):
            allsummed = np.array([sys.float_info.epsilon]*self.nbins)
            for pn in xrange(self.npeaks[j]):
                func = sp.stats.norm(loc=self.obsZs[j][pn],scale=self.obserrs[j][pn])
                # these should be two slices of the same array, rather than having two separate list comprehensions
                lo = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[:-1]])
                hi = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[1:]])
                spread = abs(self.obsZs[j][pn]-self.trueZs[j])*(hi-lo)

                # normalize probabilities to integrate (not sum)) to 1
                allsummed += spread

            pob = self.interim*allsummed
            pob = pob/self.bindifs/sum(pob)

            # sample posterior if noisy observation
            if self.meta.noise == True:
                spob = [sys.float_info.epsilon]*self.nbins
                for k in xrange(self.nbins):
                    spob[choice(self.binnos, pob)] += 1.
                pob = np.array(spob)/sum(spob)/self.meta.zdif

            mapz = self.binmids[np.argmax(pob)]
            expz = sum(self.binmids*self.bindifs*pob)
            logpob = [m.log(max(p_i,sys.float_info.epsilon)) for p_i in pob]
            logpobs.append(logpob)
            pobs.append(pob)
            mapzs.append(mapz)
            expzs.append(expz)
        self.pobs = np.array(pobs)
        self.logpobs = np.array(logpobs)
        self.mapzs = np.array(mapzs)
        self.expzs = np.array(expzs)

        # generate full Sheldon, et al. 2011 "posterior"
        stackprep = np.sum(np.array(pobs),axis=0)
        self.stack = np.array([max(sys.float_info.epsilon,stackprep[k]) for k in self.binnos])
#         print(np.dot(self.stack,self.bindifs))
        self.logstack = np.log(self.stack)

        # generate MAP N(z)
        self.mapNz = [sys.float_info.epsilon]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpobs]
        for z in mappreps:
            self.mapNz[z] += 1./self.bindifs[z]
#         print(np.dot(self.mapNz,self.bindifs))
        self.logmapNz = np.log(self.mapNz)

        # generate expected value N(z)
        expprep = [sum(z) for z in self.binmids*self.pobs*self.bindifs]
        self.expNz = [sys.float_info.epsilon]*self.nbins
        for z in expprep:
              for k in xrange(self.nbins):
                  if z > self.binlos[k] and z < self.binhis[k]:
                      self.expNz[k] += 1./self.bindifs[k]
#         print(np.dot(self.expNz,self.bindifs))
        self.logexpNz = np.log(self.expNz)

    # generate summary quantities for plotting
    def fillsummary(self):

        # define interim prior N(z),P(z) for plotting
        self.full_interim = np.array([max(i,sys.float_info.epsilon) for i in self.interim])#*self.bindifs
        self.full_loginterim = np.log(self.full_interim)

        # define true N(z),P(z) for plotting given number of galaxies
        self.full_physNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.physNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logphysNz = np.log(self.full_physNz)
        self.full_physPz = self.full_physNz/sum(self.full_physNz)
        self.full_logphysPz = np.log(self.full_physPz)

        phys = self.ngals*self.meta.realistic/sum(self.meta.realistic)
        logphys = np.log(phys)
        self.kl_physPz = self.calckl(logphys,self.full_logphysNz)
        self.lik_true = self.calclike(self.full_logphysNz)

        # define sampled N(z),P(z) for plotting
        self.full_sampNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.sampNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logsampNz = np.concatenate((np.array([m.log(sys.float_info.epsilon)]*len(self.binfront)),self.logsampNz,np.array([m.log(sys.float_info.epsilon)]*len(self.binback))),axis=0)
        self.lik_samp = self.calclike(self.full_logsampNz)

        self.avgNz = (self.full_sampNz+self.full_interim)/2.
        self.logavgNz = np.log(self.avgNz)
        self.lik_avgNz = self.calclike(self.logavgNz)
        self.kl_avgNz = self.calckl(self.logavgNz,self.full_logsampNz)

        #summary stats
        vsstack = self.stack-self.full_sampNz
        self.vsstack = np.dot(vsstack,vsstack)/self.nbins
        vslogstack = self.logstack-self.full_logsampNz
        self.vslogstack = np.dot(vslogstack,vslogstack)/self.nbins
        stackPz = self.stack/sum(self.stack)
        logstackPz = np.log(stackPz)
        self.kl_stack = self.calckl(self.logstack,self.full_logsampNz)
        self.lik_stack = self.calclike(self.logstack)

        vsmapNz = self.mapNz-self.full_sampNz
        self.vsmapNz = np.dot(vsmapNz,vsmapNz)/self.nbins
        vslogmapNz = self.logmapNz-self.full_logsampNz
        self.vslogmapNz = np.dot(vslogmapNz,vslogmapNz)/self.nbins
        mapPz = self.mapNz/sum(self.mapNz)
        logmapPz = np.log(mapPz)
        self.kl_mapNz = self.calckl(self.logmapNz,self.full_logsampNz)
        self.lik_mapNz = self.calclike(self.logmapNz)

        vsexpNz = self.expNz-self.full_sampNz
        self.vsexpNz = np.dot(vsexpNz,vsexpNz)/self.nbins
        vslogexpNz = self.logexpNz-self.full_logsampNz
        self.vslogexpNz = np.dot(vslogexpNz,vslogexpNz)/self.nbins
        expPz = self.expNz/sum(self.expNz)
        logexpPz = np.log(expPz)
        self.kl_expNz = self.calckl(self.logexpNz,self.full_logsampNz)
        self.lik_expNz = self.calclike(self.logexpNz)

        vsinterim = self.full_interim-self.full_sampNz
        self.vsinterim = np.dot(vsinterim,vsinterim)/self.nbins
        vsloginterim = self.full_loginterim-self.full_logsampNz
        self.vsloginterim = np.dot(vsloginterim,vsloginterim)/self.nbins
        interimPz = self.interim/sum(self.interim)
        loginterimPz = np.log(interimPz)
        self.kl_interim = self.calckl(self.loginterim,self.full_logsampNz)
        self.lik_interim = self.calclike(self.full_loginterim)

        self.cands = np.array([self.loginterim,self.logstack,self.logmapNz])#,self.logexpNz])
        self.liks = np.array([self.lik_interim,self.lik_stack,self.lik_mapNz])#,self.lik_expNz])
        self.start = self.cands[np.argmax(self.liks)]
        self.lik_mleNz,self.mle = self.makemle('slsqp')#'cobyla','slsqp'
        self.kl_mleNz = self.calckl(self.mle,self.full_logsampNz)

    # KL Divergence test
    def calckl(self,lpn,lqn):
        pn = np.exp(lpn)*self.bindifs
        qn = np.exp(lqn)*self.bindifs
        p = pn/np.sum(pn)
        q = qn/np.sum(qn)
        logp = np.log(p)
        logq = np.log(q)
        klpq = np.sum(p*(logp-logq))
        klqp = np.sum(q*(logq-logp))
        return(round(klpq,3),round(klqp,3))

    def calclike(self,theta):
        constterm = np.log(self.bindifs)-self.full_loginterim
        constterms = theta+constterm
        sumterm = -1.*np.dot(np.exp(theta),self.bindifs)
        for j in xrange(self.ngals):
            logterm = np.log(np.sum(np.exp(self.logpobs[j]+constterms)))
            sumterm += logterm
        return sumterm

    def makemle(self,arg):
        start_time = timeit.default_timer()
        if arg == 'cobyla':
            def cons1(theta):
                return np.dot(np.exp(theta),self.bindifs)-0.5*self.ngals
            def cons2(theta):
                return 1.5*self.ngals-np.dot(np.exp(theta),self.bindifs)
            def cons3(theta):
                return np.exp(theta)
            def cons4(theta):
                return self.ngals-np.exp(theta)
            def minlf(theta):
                return -1.*self.calclike(theta)
            loc = sp.optimize.fmin_cobyla(minlf,self.start,cons=(cons1,cons2,cons3,cons4),maxfun=self.ngals**2)
            like = self.calclike(loc)
        if arg == 'slsqp':
            def cons1(theta):
                return np.dot(np.exp(theta),self.bindifs)-0.5*self.ngals
            def cons2(theta):
                return 1.5*self.ngals-np.dot(np.exp(theta),self.bindifs)
            def minlf(theta):
                return -1.*self.calclike(theta)
            bounds = [(-sys.float_info.epsilon,np.log(self.ngals)) for k in xrange(self.nbins)]
            loc = sp.optimize.fmin_slsqp(minlf,self.start,ieqcons=([cons1,cons2]),bounds=bounds,iter=self.ngals)#,epsilon=1.)
            like = self.calclike(loc)
        print(self.start,loc,self.logsampNz)
        elapsed = timeit.default_timer() - start_time
        print(str(self.ngals)+' galaxies for '+self.meta.name+' MLE by '+arg+' in '+str(elapsed))
        return(like,loc)

    def savedat(self):

        with open(os.path.join(self.meta.simdir,'logdata.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            out.writerow(self.full_loginterim)
            for line in self.logpobs:
                out.writerow(line)
        with open(os.path.join(self.meta.simdir,'logtrue.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            out.writerow(self.full_loginterim)
            #out.writerow(self.full_logphysNz)
            trueZs = [[z] for z in self.trueZs]
            for item in trueZs:
                out.writerow(item)
