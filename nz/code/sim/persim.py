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
import random

from insim import setup
import utilsim as us

class pertest(object):
    """
    pertest class takes in a setup object and writes a catalog of data
    """

    def __init__(self, meta):

        self.meta = meta

        self.readin()
        self.choosen()
        self.choosetrue()
        self.makedat()
        self.makecat()
        self.fillsummary()
        self.savedat()

        print(self.meta.name+' simulated data')

    def readin(self):

        # take in desired subset of redshift bins
        self.ndims = self.meta.params
        self.allzs = self.meta.allzs[:self.ndims+1]
        self.zlos = self.allzs[:-1]
        self.zhis = self.allzs[1:]
        self.zdifs = self.zhis-self.zlos
        self.zdif = sum(self.zdifs)/self.ndims
        self.zmids = (self.zlos+self.zhis)/2.
        self.zavg = sum(self.zmids)/self.ndims

        # define target survey size
        self.seed = self.meta.survs

        # define realistic underlying P(z) for this number of parameters and N(z) for this survey size
        self.real = us.gmix(self.meta.real,(self.zlos[0],self.zhis[-1]))
#         self.phsPz,self.logphsPz = us.normed(self.meta.realistic[:self.ndims],self.zdifs)
#         self.phsNz = self.seed*self.phsPz
#         self.logphsNz = us.safelog(self.phsNz)

    def choosen(self):

        # sample some number of galaxies, poisson or set
        if self.meta.poisson == True:
            np.random.seed(seed=self.ndims)
            self.ngals = np.random.poisson(self.seed)
        else:
            self.ngals = self.seed
        self.randos = np.random.choice(self.ngals,len(self.meta.colors),replace=False)

    def choosetrue(self):

#         count = [0]*self.ndims

        #test all galaxies in survey have same true redshift vs. sample from physPz
        if self.meta.random == True:
            np.random.seed(seed=self.ndims)
#             for j in range(0,self.ngals):
#                 count[us.choice(xrange(self.ndims), self.phsPz)] += 1
            self.truZs = self.real.sample(self.ngals)
        else:
#             chosenbin = np.argmax(self.phsPz)
#             count[chosenbin] = self.ngals
            self.truZs = np.array([(self.allzs[0]+self.allzs[-1])/2.]*self.ngals)
        random.seed(self.ndims)
        random.shuffle(self.truZs)

#         self.count = np.array(count)

#         self.truNz = self.count/self.zdif
#         self.logtruNz = us.safelog(self.truNz)#np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampNz]))

#         self.truPz = self.truNz/self.ngals
#         self.logtruPz = us.safelog(self.truPz)#np.log(np.array([max(o,sys.float_info.epsilon) for o in self.truPz]))

#     def choosetrue(self):

#         # assign actual redshifts either uniformly or identically to mean
#         if self.meta.uniform == True:
#             np.random.seed(seed=self.ndims)
#             self.truZs = np.array([np.random.uniform(self.zlos[k],self.zhis[k]) for k in xrange(self.ndims) for j in xrange(self.count[k])])
#         else:
#             self.truZs = np.array([self.zmids[k] for k in xrange(self.ndims) for j in xrange(self.count[k])])

    def makedat(self):

      # define 1+z and variance to use for sampling z
        self.modZs = self.truZs+1.
        self.varZs = self.modZs*self.zdif

        # we can re-calculate npeaks later from shiftZs or sigZs.
        if self.meta.shape == True:
            np.random.seed(seed=self.ndims)
            self.npeaks = np.array([np.random.randint(1,self.ndims-1) for j in xrange(self.ngals)])
        else:
            self.npeaks = [1]*self.ngals

        # jitter zs to simulate inaccuracy, choose variance randomly for eah peak
        np.random.seed(seed=self.ndims)
        self.obsZs = np.array([[np.random.normal(loc=self.truZs[j],scale=self.varZs[j]) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

        # standard deviation of peaks directly dependent on true redshift vs Gaussian
        if self.meta.sigma == True or self.meta.shape == True:
            np.random.seed(seed=self.ndims)
            self.sigZs = np.array([[max(sys.float_info.epsilon,np.random.normal(loc=self.varZs[j],scale=self.varZs[j])) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])
        else:
            self.sigZs = np.array([[self.varZs[j] for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

        self.minobs = min(min(self.obsZs))
        self.maxobs = max(max(self.obsZs))

    def setup_pdfs(self):

        self.binfront = np.array([])#np.array([min(self.zlos)+x*self.zdif for x in range(int(m.floor((self.minobs-min(self.zlos))/self.zdif)),0)])#np.array([])
        self.binback = np.array([])#np.array([max(self.zhis)+x*self.zdif for x in range(1,int(m.ceil((self.maxobs-max(self.zhis))/self.zdif)))])#np.array([])
        self.binends = np.unique(np.concatenate((self.binfront,self.allzs,self.binback),axis=0))
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.nbins = len(self.binends)-1
        self.binnos = range(0,self.nbins)
        self.binmids = (self.binhis+self.binlos)/2.
        self.bindifs = self.binhis-self.binlos
        self.bindif = sum(self.bindifs)/self.nbins

        # define flat P(z) for this number of parameters and N(z) for this survey size
        self.fltPz,self.logfltPz = us.normed([1.]*self.nbins,self.bindifs)
        self.fltNz = self.seed*self.fltPz
        self.logfltNz = us.safelog(self.fltNz)

        # define underlying P(z) for this number of parameters and N(z) for this survey size
        self.phsPz = self.real.binned(self.binends)
        self.logphsPz = us.safelog(self.phsPz)
        self.phsNz = self.seed*self.phsPz
        self.logphsNz = us.safelog(self.phsNz)

        #nontrivial interim prior
        if self.meta.interim == 'flat':
            intNz = self.logfltNz
        elif self.meta.interim == 'multimodal':
            intNz = self.logphsNz
        elif self.meta.interim == 'unimodal':
            intNz = sp.stats.poisson.pmf(xrange(self.nbins),2.0)
        elif self.meta.interim == 'bimodal':
            x = self.ndims
            intNz = sp.stats.pareto.pdf(np.arange(1.,2.,1./self.nbins),x)+sp.stats.pareto.pdf(np.arange(1.,2.,1./self.nbins)[::-1],x)
#         sumintNz = np.sum(intNz)
        self.intNz = float(self.ngals)*intNz/np.dot(intNz,self.bindifs)#sumintNz/self.bindifs
        self.logintNz = us.safelog(self.intNz)
        self.intPz,self.logintPz = us.normed(self.intNz,self.bindifs)

        self.truNz,bins = np.histogram(self.truZs,bins=self.binends)
        self.truNz = self.truNz/self.bindifs
        self.logtruNz = us.safelog(self.truNz)
        self.truPz,self.logtruPz = us.normed(self.truNz,self.bindifs)

    def makecat(self):

        self.setup_pdfs()

        pobs = []
        logpobs = []
        mapzs = []
        expzs = []

        for j in xrange(self.ngals):
            allsummed = np.array([sys.float_info.epsilon]*self.nbins)
            for pn in xrange(self.npeaks[j]):
                func = sp.stats.norm(loc=self.obsZs[j][pn],scale=self.sigZs[j][pn])
                # these should be two slices of the same array, rather than having two separate list comprehensions
                lo = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[:-1]])
                hi = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[1:]])
                spread = abs(self.obsZs[j][pn]-self.truZs[j])*(hi-lo)

                # normalize probabilities to integrate (not sum)) to 1
                allsummed += spread

            pob = self.intPz*allsummed
            pob = pob/np.dot(pob,self.bindifs)

            # sample posterior if noisy observation
            if self.meta.noise == True:
                spob = [sys.float_info.epsilon]*self.nbins
                for k in xrange(self.nbins):
                    spob[us.choice(self.binnos, pob)] += 1.
                pob = np.array(spob)/sum(spob)/self.zdif

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
        stkNzprep = np.sum(np.array(pobs),axis=0)
        self.stkNz = np.array([max(sys.float_info.epsilon,stkNzprep[k]) for k in self.binnos])
        self.logstkNz = np.log(self.stkNz)
        self.stkPz,self.logstkPz = us.normed(self.stkNz,self.bindifs)

        # generate MAP N(z)
        self.mapNz = [sys.float_info.epsilon]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpobs]
        for z in mappreps:
            self.mapNz[z] += 1./self.bindifs[z]
        self.logmapNz = np.log(self.mapNz)
        self.mapPz,self.logmapPz = us.normed(self.mapNz,self.bindifs)

#         # generate expected value N(z)
#         expprep = [sum(z) for z in self.binmids*self.pobs*self.bindifs]
#         self.expNz = [sys.float_info.epsilon]*self.nbins
#         for z in expprep:
#               for k in xrange(self.nbins):
#                   if z > self.binlos[k] and z < self.binhis[k]:
#                       self.expNz[k] += 1./self.bindifs[k]
#         self.logexpNz = np.log(self.expNz)
#         self.expPz,self.logexpPz = us.normed(self.expNz,self.bindifs)

    # generate summary quantities for plotting
    def fillsummary(self):

        self.kl_phsNz = self.calckl(self.logphsNz,self.logtruNz)
        self.lik_phsNz = self.calclike(self.logphsNz)
        self.kl_truNz = self.calckl(self.logtruNz,self.logtruNz)
        self.lik_truNz = self.calclike(self.logtruNz)

        self.vslogstkNz,self.vsstkNz = self.calcvar(self.logstkNz)
        self.kl_stkNz = self.calckl(self.logstkNz,self.logtruNz)
        self.lik_stkNz = self.calclike(self.logstkNz)

        self.vslogmapNz,self.vsmapNz = self.calcvar(self.logmapNz)
        self.kl_mapNz = self.calckl(self.logmapNz,self.logtruNz)
        self.lik_mapNz = self.calclike(self.logmapNz)

#         self.vslogexpNz,self.vsexpNz = self.calcvar(self.logexpNz)
#         self.kl_expNz = self.calckl(self.logexpNz,self.logtruNz)
#         self.lik_expNz = self.calclike(self.logexpNz)

        self.vslogintNz,self.vsintNz = self.calcvar(self.logintNz)
        self.kl_intNz = self.calckl(self.logintNz,self.logtruNz)
        self.lik_intNz = self.calclike(self.logintNz)

        self.cands = np.array([self.logintNz,self.logstkNz,self.logmapNz])#,self.logexpNz])
        self.liks = np.array([self.lik_intNz,self.lik_stkNz,self.lik_mapNz])#,self.lik_expNz])
        self.start = self.cands[np.argmax(self.liks)]
        self.lik_mleNz,self.mle = self.makemle('slsqp')#'cobyla','slsqp'
        self.kl_mleNz = self.calckl(self.mle,self.logtruNz)

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
        constterm = np.log(self.bindifs)-self.logintNz
        constterms = theta+constterm
        sumterm = -1.*np.dot(np.exp(theta),self.bindifs)
        for j in xrange(self.ngals):
            logterm = np.log(np.sum(np.exp(self.logpobs[j]+constterms)))
            sumterm += logterm
        return sumterm

    def calcvar(self,theta):
        vslog = theta-self.logtruNz
        vslog = np.dot(vslog,vslog)/self.nbins
        vs = np.exp(theta)-self.truNz
        vs = np.dot(vs,vs)/self.nbins
        return(vslog,vs)

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
            bounds = [(-sys.float_info.epsilon,np.log(self.ngals/min(self.bindifs))) for k in xrange(self.nbins)]
            loc = sp.optimize.fmin_slsqp(minlf,self.start,ieqcons=([cons1,cons2]),bounds=bounds,iter=self.ngals)#,epsilon=1.)
            like = self.calclike(loc)
        elapsed = timeit.default_timer() - start_time
        print(str(self.ngals)+' galaxies for '+self.meta.name+' MLE by '+arg+' in '+str(elapsed)+': '+str(loc))
        return(like,loc)

    def savedat(self):

        with open(os.path.join(self.meta.simdir,'logdata.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            out.writerow(self.logintNz)
            for line in self.logpobs:
                out.writerow(line)
        with open(os.path.join(self.meta.simdir,'logtrue.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            out.writerow(self.logintNz)
            #out.writerow(self.full_logphsNz)
            truZs = [[z] for z in self.truZs]
            for item in truZs:
                out.writerow(item)
