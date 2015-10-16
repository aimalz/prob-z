"""
per-sim module generates the data for one run
"""

import numpy as np
import sys
import math as m
import os
import random
import scipy as sp
from scipy import stats
import csv
from insim import setup
from utilsim import *

class pertest(object):
    def __init__(self, meta):

        self.meta = meta

        # perparam differs from meta
        self.ndims = meta.params
        self.allzs = meta.allzs[:self.ndims+1]#sorted(set(self.zlos+self.zhis))
        self.zlos = self.allzs[:-1]#meta.allzlos[:self.ndims]
        self.zhis = self.allzs[1:]#meta.allzhis[:self.ndims]
        self.zmids = (self.zlos+self.zhis)/2.
        self.zavg = sum(self.zmids)/self.ndims

        # define realistic underlying P(z) for this number of parameters
        self.realsum = sum(meta.realistic[:self.ndims])
        self.realistic_pdf = np.array([meta.realistic[k]/self.realsum/meta.zdifs[k] for k in xrange(0,self.ndims)])
        self.truePz = self.realistic_pdf
        self.logtruePz = np.array([m.log(max(tPz,sys.float_info.epsilon)) for tPz in self.truePz])

        # define flat P(z) for this number of parameters
        self.avgprob = 1./self.ndims/meta.zdif
        self.logavgprob = m.log(self.avgprob)
        self.flatPz = [self.avgprob]*self.ndims
        self.logflatPz = [self.logavgprob]*self.ndims

        # set true value of N(z) for this survey size
        self.seed = meta.survs
        self.trueNz = self.seed*self.realistic_pdf
        self.logtrueNz = [m.log(max(x,sys.float_info.epsilon)) for x in self.trueNz]

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

        print('generated data')

    def choosen(self):

        # sample some number of galaxies, poisson or set
        if self.meta.poisson:
            self.ngals = np.random.poisson(self.seed)#[[np.random.poisson(seed) for n in sampnos] for s in survnos]
        else:
            self.ngals = self.seed

    def choosebins(self):

        count = [0]*self.ndims

        #test all galaxies in survey have same true redshift vs. sample from truePz
        if self.meta.random:
            for j in range(0,self.ngals):
              count[choice(xrange(self.ndims), self.truePz)] += 1
              #count[choice(xrange(self.p_run.ndims), self.p_run.flatPz)] += 1
        else:
            chosenbin = np.argmax(self.truePz)
            count[chosenbin] = self.ngals
        #print('count='+str(count))

        self.count = np.array(count)

        self.sampNz = self.count/self.meta.zdif
        self.logsampNz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampNz]))

        self.sampPz = self.sampNz/self.ngals#count/ngal/zdif
        self.logsampPz = np.log(np.array([max(o,sys.float_info.epsilon) for o in self.sampPz]))

    def choosetrue(self):

        # assign actual redshifts either uniformly or identically to mean
        if self.meta.uniform:
            self.trueZs = np.array([random.uniform(self.zlos[k],self.zhis[k]) for k in xrange(self.ndims) for j in xrange(self.count[k])])
        else:
            self.trueZs = np.array([self.zmids[k] for k in xrange(self.ndims) for j in xrange(self.count[k])])
        #print('trueZs='+str(self.trueZs))

    def makedat(self):

      # define 1+z and variance to use for sampling z
        modZs = self.trueZs+1.#[[trueZs[s][n]+1. for n in sampnos] for s in survnos]
        varZs = self.meta.zdif*modZs#[j] for j in xrange(self.ngals)]# for n in sampnos] for s in survnos])#zdif*(trueZs+1.)

        # we can re-calculate npeaks later from shiftZs or sigZs.
        if self.meta.shape:
            npeaks = [random.randrange(1,self.ndims,1) for j in xrange(self.ngals)]
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

    def setup_pdfs(self):

        self.binfront = np.array([min(self.zlos)+x*self.meta.zdif for x in range(int(m.floor((self.minobs-min(self.zlos))/self.meta.zdif)),0)])
        self.binback = np.array([max(self.zhis)+x*self.meta.zdif for x in range(1,int(m.ceil((self.maxobs-max(self.zhis))/self.meta.zdif)))])
        self.binends = np.unique(np.concatenate((self.binfront,self.allzs,self.binback),axis=0))
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.nbins = len(self.binends)-1
        self.binnos = range(0,self.nbins)
        self.binmids = (self.binhis+self.binlos)/2.#[(binends[k]+binends[k+1])/2. for k in binnos]
        self.bindifs = self.binhis-self.binlos
        self.bindif = sum(self.bindifs)/self.nbins

    def makecat(self):

        self.setup_pdfs()

        self.obsdata = zip(self.obsZs,self.obserror)
        pobs = []
        logpobs = []
        mapzs = []
        expzs = []

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
            if self.meta.noise:
                spob = [sys.float_info.epsilon]*self.nbins
                for k in xrange(2*self.nbins):#self.binnos:
                    spob[choice(self.binnos, pob)] += 1.
                pob = np.array(spob)/sum(spob)/self.meta.zdif

            mapz = self.binmids[np.argmax(pob)]
            expz = sum(self.binmids*pob*self.meta.zdif)
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
        self.logstack = np.log(self.stack)

        # generate MAP N(z)
        self.mapNz = [sys.float_info.epsilon]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpobs]
        for z in mappreps:
              self.mapNz[z] += 1./self.bindifs[z]
        self.logmapNz = np.log(self.mapNz)

        # generate expected value N(z)
        expprep = [sum(z) for z in self.binmids*np.exp(self.pobs)*self.bindifs]
        self.expNz = [sys.float_info.epsilon]*self.nbins
        for z in expprep:
              for k in xrange(self.nbins):
                  if z > self.binlos[k] and z < self.binhis[k]:
                      self.expNz[k] += 1./self.bindifs[k]
        self.logexpNz = np.log(self.expNz)

    # generate summary quantities for plotting
    def fillsummary(self):

        #define true N(z),P(z) for plotting given number of galaxies
        self.full_trueNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.trueNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logtrueNz = np.log(self.full_trueNz)

        #define flat N(z),P(z) for plotting
        self.full_flatNz = np.array([self.seed/self.meta.zdif/self.nbins]*self.nbins)
        self.full_logflatNz = np.log(self.full_flatNz)

        #define sampled N(z),P(z) for plotting
        self.full_sampNz = np.concatenate((np.array([sys.float_info.epsilon]*len(self.binfront)),self.sampNz,np.array([sys.float_info.epsilon]*len(self.binback))),axis=0)
        self.full_logsampNz = np.concatenate((np.array([m.log(sys.float_info.epsilon)]*len(self.binfront)),self.logsampNz,np.array([m.log(sys.float_info.epsilon)]*len(self.binback))),axis=0)


    def savedat(self):

        with open(os.path.join(self.meta.simdir,'logdata.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            for line in self.logpobs:
                out.writerow(line)
        with open(os.path.join(self.meta.simdir,'logtrue.csv'),'wb') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            out.writerow(self.binends)
            #out.writerow(self.full_logtrueNz)
            trueZs = [[z] for z in self.trueZs]
            for item in trueZs:
                out.writerow(item)
