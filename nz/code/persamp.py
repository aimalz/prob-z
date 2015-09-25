# one of these objects per instantiation of a survey

import os
import random
import cPickle
import numpy as np
import sys
import math as m
import scipy as sp
from scipy import stats
import emcee
from inputs import setup
import key
from util import *

# eventually break this up into multiple files...
# import datagen

# define class for instantiation of survey
class persamp(object):
    path_builder = path("{topdir}/{p}/{s}/{n}")

    def __init__(self,meta,s_run,n):
        self.s_run = s_run
        self.p_run = self.s_run.p_run
        self.p = s_run.p
        self.s = s_run.s
        self.n = n
        self.meta = meta
        self.key = s_run.key.add(n=self.n)
        self.path_builder = persamp.path_builder.fill(topdir = meta.topdir, p=self.p, s = self.s, n = self.n)
        self.true_path_builder = path("")

        # sample some number of galaxies, poisson or set
        if meta.poisson[self.n]:
          self.ngals = ngals = np.random.poisson(s_run.seed)#[[np.random.poisson(seed) for n in sampnos] for s in survnos]
        else:
          self.ngals = s_run.seed
        print('ngals='+str(self.ngals))

        self.filltrue()
        self.fillcat()
        self.setup_pdfs()
        self.fillpdfs()

        self.fillsummary()

#         q = 1.#0.5
#         e = 0.15/self.meta.zdif**2
#         tiny = q*1e-6
#         self.covmat = np.array([[q*m.exp(-0.5*e*(self.binmids[a]-self.binmids[b])**2.) for a in xrange(0,self.nbins)] for b in xrange(0,self.nbins)])+tiny*np.identity(self.nbins)
        self.covmat = np.identity(self.nbins)
        self.priordist = mvn(self.full_logflatNz,self.covmat)

        #self.priordist = mvn(self.full_logflatNz,np.identity(self.nbins))
        ## add ourself to the list of things that our parent knows about
        s_run.n_runs.append(self)
        self.i_runs = []

        #how many walkers
        self.nwalkers = 2*self.nbins
        self.walknos = xrange(self.nwalkers)

        self.postdist = post(self.priordist, self.binends, self.logpobs)

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nbins, self.postdist.lnprob)

    # associated directory for this instantiation
    def get_dir(self):
        return self.path_builder.construct()

    # set true redshifts
    def filltrue(self):
#         on_disk = self.meta.olddata#self.key.load_true(self.meta.topdir)
#         if on_disk:# is not None:
#             self.count = on_disk['count']
#             self.sampNz = on_disk['sampNz']
#             self.logsampNz = on_disk['logsampNz']
#             self.sampPz = on_disk['sampPz']
#             self.logsampPz = on_disk['logsampPz']
#             return

        count = [0]*self.p_run.ndims

        #test all galaxies in survey have same true redshift vs. sample from truePz
        if self.meta.random[self.n]:
            for j in range(0,self.ngals):
              count[choice(xrange(self.p_run.ndims), self.p_run.truePz)] += 1
              #count[choice(xrange(self.p_run.ndims), self.p_run.flatPz)] += 1
        else:
            chosenbin = np.argmax(self.p_run.truePz)
            count[chosenbin] = self.ngals
        #print('count='+str(count))

        self.count = np.array(count)

        self.sampNz = self.count/self.meta.zdif
        self.logsampNz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampNz]))

        self.sampPz = self.sampNz/self.ngals#count/ngal/zdif
        self.logsampPz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampPz]))

        # assign actual redshifts either uniformly or identically to mean
        if self.meta.uniform[self.n]:
            self.trueZs = np.array([random.uniform(self.p_run.zlos[k],self.p_run.zhis[k]) for k in xrange(self.p_run.ndims) for j in xrange(count[k])])
        else:
            self.trueZs = np.array([self.p_run.zmids[k] for k in xrange(self.p_run.ndims) for j in xrange(self.count[k])])
        #print('trueZs='+str(self.trueZs))

        self.key.store_true(self.meta.topdir,
                            {'count': self.count,
                             'trueZs': self.trueZs,
                             'sampNz': self.sampNz,
                             'logsampNz': self.logsampNz,
                             'sampPz': self.sampPz,
                             'logsampPz': self.logsampPz})

        print('simulated sample '+str(self.n+1)+' of '+str(self.ngals)+' galaxies')

    # generate the catalog of individual galaxy posteriors
    def fillcat(self):

#         on_disk = self.meta.olddata#self.key.load_cat(self.meta.topdir)
#         if on_disk is not None:
#             self.obsZs = on_disk['obsZs']
#             self.obserror = on_disk['obserror']
#             self.minobs = min(min(self.obsZs))
#             self.maxobs = max(max(self.obsZs))
#             return

        # define 1+z and variance to use for sampling z
        modZs = self.trueZs+1.#[[trueZs[s][n]+1. for n in sampnos] for s in survnos]
        varZs = [self.meta.zdif*modZs[j] for j in xrange(self.ngals)]# for n in sampnos] for s in survnos])#zdif*(trueZs+1.)

        # we can re-calculate npeaks later from shiftZs or sigZs.
        if self.meta.shape[self.n]:
            npeaks = [random.randrange(1,self.p_run.ndims,1) for j in xrange(self.ngals)]
        else:
            npeaks = [1]*self.ngals

        # jitter zs to simulate inaccuracy, choose variance randomly for eah peak
        shiftZs = np.array([[np.random.normal(loc=self.trueZs[j],scale=varZs[j]) for p in xrange(npeaks[j])] for j in xrange(0,self.ngals)])
        sigZs = np.array([[abs(random.gauss(varZs[j],m.sqrt(varZs[j]))) for p in xrange(npeaks[j])] for j in xrange(0,self.ngals)])

        self.minobs = min(min(shiftZs))
        self.maxobs = max(max(shiftZs))

        # for consistency
        self.obserror = sigZs
        self.obsZs = shiftZs

        # write out the data into catalog
        self.key.store_cat(self.meta.topdir,
                           {'obsZs' : self.obsZs,
                            'obserror' : self.obserror})

        print('observed sample '+str(self.n+1)+' of '+str(self.ngals)+' galaxies')

    # make new bins to accommodate posteriors
    def setup_pdfs(self):
        self.binfront = np.array([min(self.p_run.zlos)+x*self.meta.zdif for x in range(int(m.floor((self.minobs-min(self.p_run.zlos))/self.meta.zdif)),0)])
        self.binback = np.array([max(self.p_run.zhis)+x*self.meta.zdif for x in range(1,int(m.ceil((self.maxobs-max(self.p_run.zhis))/self.meta.zdif)))])
        self.binends = np.unique(np.concatenate((self.binfront,self.p_run.allzs,self.binback),axis=0))
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.nbins = len(self.binends)-1
        self.binnos = range(0,self.nbins)
        self.binmids = (self.binhis+self.binlos)/2.#[(binends[k]+binends[k+1])/2. for k in binnos]

    # make posteriors out of catalog
    def fillpdfs(self):

        self.obsdata = zip(self.obsZs,self.obserror)
        pobs = []
        logpobs = []
        mapzs = []

        for (obsZ, obserr) in self.obsdata:
            allsummed = [sys.float_info.epsilon]*self.nbins
            npeaks = len(obsZ)
            for pn in lrange(obsZ):
                func = sp.stats.norm(loc=obsZ[pn],scale=obserr[pn])
                # these should be two slices of the same array, rather than having two separate list comprehensions
                lo = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[:-1]])
                hi = np.array([max(sys.float_info.epsilon,func.cdf(binend)) for binend in self.binends[1:]])
                spread = (hi-lo)
                # normalize probabilities to integrate (not sum)) to 1
                onesummed = max(sum(spread),sys.float_info.epsilon)
                allsummed += spread/onesummed

            pob = allsummed/self.meta.zdif/npeaks

            # sample posterior if noisy observation
            if self.meta.noise[self.n]:
                spob = [sys.float_info.epsilon]*self.nbins
                for k in xrange(2*self.nbins):#self.binnos:
                    spob[choice(self.binnos, pob)] += 1.
                pob = np.array(spob)/sum(spob)/self.meta.zdif

            mapz = self.binmids[np.argmax(pob)]
            logpob = [m.log(max(p_i,sys.float_info.epsilon)) for p_i in pob]
            logpobs.append(logpob)
            pobs.append(pob)
            mapzs.append(mapz)
        self.pobs = np.array(pobs)
        self.logpobs = np.array(logpobs)
        self.mapzs = np.array(mapzs)

        # generate full Sheldon, et al. 2011 "posterior"
        stackprep = np.sum(np.array(pobs),axis=0)
        self.stack = np.array([max(sys.float_info.epsilon,stackprep[k]) for k in self.binnos])
        self.logstack = np.log(self.stack)

        print('processed sample '+str(self.n+1)+' of '+str(self.ngals)+' galaxies')

    # generate summary quantities for plotting
    def fillsummary(self):

        #define true N(z),P(z) for plotting given number of galaxies
        self.full_trueNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.s_run.trueNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logtrueNz = np.log(self.full_trueNz)

        #define flat N(z),P(z) for plotting
        self.full_flatNz = np.array([self.s_run.seed/self.meta.zdif/self.nbins]*self.nbins)
        self.full_logflatNz = np.log(self.full_flatNz)

        #define sampled N(z),P(z) for plotting
        self.full_sampNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.sampNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logsampNz = np.concatenate((np.array([m.log(sys.float_info.epsilon)]*len(self.binfront)),self.logsampNz,np.array([m.log(sys.float_info.epsilon)]*len(self.binback))),axis=0)
