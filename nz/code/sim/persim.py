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
import cPickle as cpkl

from insim import setup
import utilsim as us

class pertest(object):
    """
    pertest class takes in a setup object and writes a catalog of data
    """

    def __init__(self, meta):

        self.meta = meta

        self.seed = self.meta.allnbins

        start_time = timeit.default_timer()
        self.readin()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' constructed true N(z) in '+str(elapsed))

        start_time = timeit.default_timer()
        self.choosen()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' N chosen in '+str(elapsed))

        start_time = timeit.default_timer()
        self.makelf()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' parameters chosen in '+str(elapsed))

        start_time = timeit.default_timer()
        self.choosetrue()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' true zs chosen in '+str(elapsed))
#         self.prepinterim()

        start_time = timeit.default_timer()
        self.makecat()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' catalog constructed in '+str(elapsed))

        start_time = timeit.default_timer()
        self.savedat()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' data saved in '+str(elapsed))

        start_time = timeit.default_timer()
        self.fillsummary()
        elapsed = timeit.default_timer() - start_time
        print(self.meta.name+' summary stats calculated in '+str(elapsed))

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
        self.surv = self.meta.surv

        # define realistic underlying P(z) for this number of parameters and N(z) for this survey size
        self.real = us.gmix(self.meta.real,(self.allzs[0],self.allzs[-1]))
#         self.phsPz,self.logphsPz = us.normed(self.meta.realistic[:self.ndims],self.zdifs)
#         self.phsNz = self.surv*self.phsPz
#         self.logphsNz = us.safelog(self.phsNz)

        # define underlying P(z) for this number of parameters and N(z) for this survey size
#         self.z_cont = np.arange(self.allzs[0],self.allzs[-1],1./self.surv)
#         self.phsPz = self.real.pdf(self.z_cont)
# #         self.phsPz = self.real.binned(self.binends)
#         self.logphsPz = us.safelog(self.phsPz)

    def choosen(self):

        # sample some number of galaxies, poisson or set
        if self.meta.poisson == True:
            np.random.seed(seed=self.seed)
            self.ngals = np.random.poisson(self.surv)
        else:
            self.ngals = self.surv
        np.random.seed(seed=self.seed)
        self.randos = np.random.choice(self.ngals,len(self.meta.colors),replace=False)

#         self.phsNz = self.ngals*self.phsPz
#         self.logphsNz = us.safelog(self.phsNz)

    def choosetrue(self):

        # test all galaxies in survey have same true redshift vs. sample from physPz
#         if self.meta.random == True:
#             np.random.seed(seed=self.seed)
#             self.truZs = self.real.sample(self.ngals)
#         else:
        if self.meta.random == False:
            center = (self.allzs[0]+self.allzs[-1])/2.
            self.meta.real = np.array([np.array([center,1./self.surv,1.])])
        self.real = us.gmix(self.meta.real,(self.allzs[0],self.allzs[-1]))
#             self.truZs = np.array([center]*self.ngals)
        np.random.seed(seed=self.seed)
        self.truZs = self.real.sample(self.ngals)
#         np.random.seed(seed=self.seed)
#         np.random.shuffle(self.truZs)

        # jitter peak zs given sigma to simulate inaccuracy
        np.random.seed(seed=self.seed)
        self.shift = np.array([[np.random.normal(loc=0.,scale=self.varZs[j][p]) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])
        self.obsZs = np.array([[self.truZs[j]+self.shift[j][p] for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

        self.minobs = min(min(self.obsZs))
        self.maxobs = max(max(self.obsZs))

    def makelf(self):

        # choose npeaks before sigma so you know how many to pick
        if self.meta.shape == True:
            np.random.seed(seed=self.seed)
            weights = [1./k**self.meta.noisefact for k in xrange(1,self.ndims)]
            self.npeaks = np.array([us.choice(xrange(1,self.ndims),weights) for j in xrange(self.ngals)])#np.array([np.random.randint(1,self.ndims-1) for j in xrange(self.ngals)])
            #self.npeaks = [1]*self.ngals
        else:
            self.npeaks = [1]*self.ngals

        # choose random sigma
        sigval = self.meta.noisefact*self.zdif
        self.var = us.tnorm(sigval,sigval,[0.,self.allzs[-1]])
        # self.modZs = np.array([self.var.rvs(self.npeaks[j]) for j in xrange(self.ngals)])#self.truZs+1.
        self.varZs = np.array([self.var.rvs(self.npeaks[j]) for j in xrange(self.ngals)])#*self.zdif
        # np.random.shuffle(self.modZs)

        # test increasing sigma associated with increasing z
        if self.meta.sigma == True:# or self.meta.shape == True:
            np.random.seed(seed=self.seed)
#             self.sigZs = np.array([[max(sys.float_info.epsilon,np.random.normal(loc=self.varZs[j],scale=self.varZs[j])) for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])
            self.sigZs = np.array([self.var.rvs(self.npeaks[j]) for j in xrange(self.ngals)])
        else:
            self.sigZs = self.varZs#np.array([[self.varZs[j] for p in xrange(self.npeaks[j])] for j in xrange(0,self.ngals)])

    def setup_pdfs(self):

        self.binfront = np.array([])#np.array([min(self.zlos)+x*self.zdif for x in range(int(m.floor((self.minobs-min(self.zlos))/self.zdif)),0)])#np.array([])
        self.binback = np.array([])#np.array([max(self.zhis)+x*self.zdif for x in range(1,int(m.ceil((self.maxobs-max(self.zhis))/self.zdif)))])#np.array([])
        self.binends = np.unique(np.concatenate((self.binfront,self.allzs,self.binback),axis=0))
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.nbins = len(self.binends)-1
        self.binmids = (self.binhis+self.binlos)/2.
        self.bindifs = self.binhis-self.binlos
        self.bindif = sum(self.bindifs)/self.nbins
        self.binrange = self.binends[-1]-self.binends[0]

#         define flat P(z) for this number of parameters and N(z) for this survey size
        self.fltPz = us.normed([1.]*self.nbins,self.bindifs)
        self.logfltPz = us.safelog(self.fltPz)
        self.fltNz = self.ngals*self.fltPz
        self.logfltNz = us.safelog(self.fltNz)

        self.truNz,bins = np.histogram(self.truZs,bins=self.binends)
        self.truPz = us.normed(self.truNz,self.bindifs)
        self.logtruPz = us.safelog(self.truPz)
        self.truNz = self.ngals*self.truPz
        self.logtruNz = us.safelog(self.truNz)

#         self.zgrid = np.arange(self.zlos[0],self.zhis[-1]+1./100,1./100)
#         self.gridmids = (self.zgrid[1:]+self.zgrid[:-1])/2.
#         self.griddifs = self.zgrid[1:]-self.zgrid[:-1]
#         global p

        #nontrivial interim prior
        if self.meta.interim == 'flat':
            fun = 1.
            intP = self.fltPz
#             intP = sp.stats.uniform(loc=self.binends[0],scale=self.binrange)
#             p = np.array([fun for z in self.gridmids])
        elif self.meta.interim == 'unimodal':
#            percentile(self.binends,0)
            fun = us.tnorm(min(self.binends),(max(self.binends)-min(self.binends))/5.,(min(self.binends),max(self.binends)))#sp.stats.norm(np.percentile(self.binends,25),np.sqrt(np.mean(self.binends)))
#             intP = sp.stats.poisson.pmf(xrange(self.nbins),2.0)
#             intP = sp.stats.poisson(2.0)
            intP = np.array([z*fun.pdf(z) for z in self.binmids])
            intP = intP-min(intP)+1./self.ngals/self.bindif
#             p = np.array([z*fun.pdf(z) for z in self.gridmids])
#             p = p-min(p)+1./self.ngals/self.bindif
        elif self.meta.interim == 'bimodal':
            mulo = np.percentile(self.binends,20)
            muhi = np.percentile(self.binends,80)
            funlo = us.tnorm(mulo,(max(self.binends)-min(self.binends))/7.,(min(self.binends),max(self.binends)))#sp.stats.norm(np.percentile(self.binends,75),np.sqrt(np.mean(self.binends)))
            funhi = us.tnorm(muhi,(max(self.binends)-min(self.binends))/3.5,(min(self.binends),max(self.binends)))
#             x = self.nbins
#             intP = sp.stats.pareto.pdf(np.arange(1.,2.,1./x),x)+sp.stats.pareto.pdf(np.arange(1.,2.,1./x)[::-1],x)
#             intP = sp.stats.pareto(self.nbins)
            intP = np.array([funlo.pdf(z)+funhi.pdf(z) for z in self.binmids])
            intP = intP-min(intP)+1./self.ngals#(1.+self.binmids*(max(pdf)-pdf))**2
#             p = np.array([1.25*funlo.pdf(z)+funhi.pdf(z) for z in self.gridmids])
#             p = p-min(p)+1./self.ngals/self.bindif
#         elif self.meta.interim == 'multimodal':
#             intP = self.real.binned(self.binends)
# #             intP = self.real
        self.intPz = us.normed(intP,self.bindifs)
        self.logintPz = us.safelog(self.intPz)
        self.intNz = float(self.ngals)*self.intPz
        self.logintNz = us.safelog(self.intNz)

    def makecat(self):

        self.setup_pdfs()

        pdfs = []
        logpdfs = []
        mapZs = []
        expZs = []

        for j in xrange(self.ngals):
#             allsummed = np.array([0.]*self.nbins)
#             for pn in xrange(self.npeaks[j]):
#                 func = us.tnorm(self.obsZs[j][pn],self.sigZs[j][pn],(self.allzs[0],self.allzs[-1]))
#                 cdfs = np.array([func.cdf(binend) for binend in self.binends])
#                 spread = cdfs[1:]-cdfs[:-1]

#                 allsummed += spread

#             pdf = self.intPz*allsummed
#             # normalize probabilities to integrate (not sum)) to 1
#             pdf = pdf/max(np.dot(pdf,self.bindifs),sys.float_info.epsilon)

#             # sample posterior if noisy observation
#             if self.meta.noise == True or self.meta.shape == True:
#                 spdf = [0]*self.nbins
#                 for k in xrange(self.nbins):
#                     spdf[us.choice(xrange(self.nbins), pdf)] += 1
#                 pdf = np.array(spdf)/np.dot(spdf,self.bindifs)
            pdf = self.makepdfs(j,self.binends,self.intPz)
#             j = self.randos[0]
#             lf = self.makelfs(j,self.zgrid)
#             #lf = np.array([np.array([l[zo]*l[zt] for zo in us.lrange(self.gridmids)]) for zt in us.lrange(self.gridmids)])
#             lfs.append(lf)
            mapZ = self.binmids[np.argmax(pdf)]
            expZ = sum(self.binmids*self.bindifs*pdf)
            logpdf = us.safelog(pdf)
            logpdfs.append(logpdf)
            pdfs.append(pdf)
            mapZs.append(mapZ)
            expZs.append(expZ)
        self.pdfs = np.array(pdfs)
        self.logpdfs = np.array(logpdfs)
        self.mapZs = np.array(mapZs)
        self.expZs = np.array(expZs)

        # generate full Sheldon, et al. 2011 "posterior"
        self.stkNz = np.sum(np.array(pdfs),axis=0)
        self.logstkNz = us.safelog(self.stkNz)
        self.stkPz = us.normed(self.stkNz,self.bindifs)
        self.logstkPz = us.safelog(self.stkPz)

        # generate MAP N(z)
        self.mapNz = [0]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpdfs]
        for z in mappreps:
            self.mapNz[z] += 1
        self.mapNz = self.mapNz/self.bindifs
        self.logmapNz = us.safelog(self.mapNz)
        self.mapPz = us.normed(self.mapNz,self.bindifs)
        self.logmapPz = us.safelog(self.mapPz)

        # generate expected value N(z)
        expprep = [sum(z) for z in self.binmids*self.pdfs*self.bindifs]
        self.expNz = [sys.float_info.epsilon]*self.nbins
        for z in expprep:
              for k in xrange(self.nbins):
                  if z > self.binlos[k] and z < self.binhis[k]:
                      self.expNz[k] += 1
        self.expNz = self.expNz/self.bindifs
        self.logexpNz = np.log(self.expNz)
        self.expPz = us.normed(self.expNz,self.bindifs)
        self.logexpPz = us.safelog(self.expNz)

    # generate summary quantities for plotting
    def fillsummary(self):

#         self.kl_phsNz = self.calckl(self.logphsNz,self.logtruNz)
#         self.lik_phsNz = self.calclike(self.logphsNz)
        self.kl_truNz = self.calckl(self.logtruNz,self.logtruNz)
        self.lik_truNz = self.calclike(self.logtruNz)

        self.vslogstkNz,self.vsstkNz = self.calcvar(self.logstkNz)
        self.kl_stkNz = self.calckl(self.logstkNz,self.logtruNz)
        self.lik_stkNz = self.calclike(self.logstkNz)

        self.vslogmapNz,self.vsmapNz = self.calcvar(self.logmapNz)
        self.kl_mapNz = self.calckl(self.logmapNz,self.logtruNz)
        self.lik_mapNz = self.calclike(self.logmapNz)

        self.vslogexpNz,self.vsexpNz = self.calcvar(self.logexpNz)
        self.kl_expNz = self.calckl(self.logexpNz,self.logtruNz)
        self.lik_expNz = self.calclike(self.logexpNz)

        self.vslogintNz,self.vsintNz = self.calcvar(self.logintNz)
        self.kl_intNz = self.calckl(self.logintNz,self.logtruNz)
        self.lik_intNz = self.calclike(self.logintNz)

        self.cands = np.array([self.logintNz,self.logstkNz,self.logmapNz])#,self.logexpNz])
        self.liks = np.array([self.lik_intNz,self.lik_stkNz,self.lik_mapNz])#,self.lik_expNz])
        self.start = self.logtruNz#self.cands[np.argmax(self.liks)]
        self.lik_mmlNz,self.logmmlNz = self.makemml('fmin')#'bfgs_b','bfgs','cobyla','fmin','powell','slsqp'
        self.mmlNz = np.exp(self.logmmlNz)
        self.kl_mmlNz = self.calckl(self.logmmlNz,self.logtruNz)

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
            logterm = np.log(np.sum(np.exp(self.logpdfs[j]+constterms)))
            sumterm += logterm
        return(round(sumterm))

    def calcvar(self,theta):
        vslog = theta-self.logtruNz
        vslog = np.dot(vslog,vslog)/self.nbins
        vs = np.exp(theta)-self.truNz
        vs = np.dot(vs,vs)/self.nbins
        return(vslog,vs)

    def makemml(self,arg):
        start_time = timeit.default_timer()

        def minlf(theta):
            return -1.*self.calclike(theta)
        def maxruns():
            return(self.surv)#**2)

        if arg == 'fmin':
            loc = sp.optimize.fmin(minlf,self.start,maxiter=maxruns(),maxfun=maxruns(), disp=True)
        like = self.calclike(loc)
        elapsed = timeit.default_timer() - start_time
        print(str(self.ngals)+' galaxies for '+self.meta.name+' MMLE by '+arg+' in '+str(elapsed)+': '+str(loc))

        with open(os.path.join(self.meta.simdir,'logmmle.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow([like])
            out.writerow(loc)

        return(like,loc)

    def savedat(self):

        with open(os.path.join(self.meta.simdir,'logdata.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            out.writerow(self.logintNz)
            for line in self.logpdfs:
                out.writerow(line)
        with open(os.path.join(self.meta.simdir,'logtrue.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            truZs = [[z] for z in self.truZs]
            for item in truZs:
                out.writerow(item)
        with open(os.path.join(self.meta.simdir,'truth.p'),'wb') as cpfile:
            cpkl.dump(self.meta.real, cpfile)

    def makepdfs(self,j,grid,intp):
        difs = grid[1:]-grid[:-1]
        allsummed = np.array([0.]*(len(grid)-1))
        for pn in xrange(self.npeaks[j]):
            func = us.tnorm(self.obsZs[j][pn],self.sigZs[j][pn],(min(grid),max(grid)))
            cdfs = np.array([func.cdf(binend) for binend in grid])
            spread = cdfs[1:]-cdfs[:-1]

            allsummed += spread

        pdf = intp*allsummed
        # normalize probabilities to integrate (not sum)) to 1
        pdf = pdf/max(np.dot(pdf,difs),sys.float_info.epsilon)

        # sample posterior if noisy observation

        if self.meta.noise == True:# or self.meta.shape == True:
            spdf = [0]*(len(grid)-1)
            for k in xrange(len(grid)-1):
                spdf[us.choice(xrange(len(grid)-1), pdf)] += 1
            pdf = np.array(spdf)/np.dot(spdf,difs)

        return(pdf)
