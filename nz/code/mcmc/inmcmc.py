"""
in-mcmc module contains parameters controlling one run of MCMC p(z) inference program
"""

# TO DO: add docstrings

import os
import cPickle as cpkl
import math as m
import numpy as np
import scipy as sp
import shutil
import emcee
import itertools
import timeit
import sklearn as skl
import sys
import csv
import multiprocessing as mp

import utilmcmc as um
import keymcmc as key
from permcmc import pertest
import statmcmc as stats

# setup object contains inputs of necessary parameters for code
class setup(object):
    """
    setup object specifies all parameters controlling one run of p(z) inference program
    """
    def __init__(self, input_address):
#         self.pool = pool
        # name the test for which this is setup object
        self.key = key.key(t=input_address)
        print('initial key = '+str(self.key))

        # read input parameters
        self.testdir = os.path.join('..','tests')
        with open(os.path.join(self.testdir,input_address)) as infile:
            lines = (line.split(None) for line in infile)
            indict   = {defn[0]:defn[1] for defn in lines}

        # load top directory location
#         with open(os.path.join(self.testdir,'topdirs.p')) as topdirs:
#             self.topdir = cpkl.load(topdirs)[input_address]
        self.inadd = input_address[:-4]
        self.testdir = os.path.join('..','tests')
        self.updir = os.path.join(self.testdir,self.inadd)
        self.datadir = os.path.join(self.updir,'data')

        # make directory into which to put output of this test
        self.topdir = os.path.join(self.updir,'mcmc')
#         if os.path.exists(self.topdir):
#             shutil.rmtree(self.topdir)
# #             os.remove(os.path.join(self.topdir,'samples.csv'))
# #             os.remove(os.path.join(self.topdir,'calctimer.txt'))
# #             os.remove(os.path.join(self.topdir,'plottimer.txt'))
# #         else:
        if not os.path.exists(self.topdir):
            os.makedirs(self.topdir)

        iterplace = os.path.join(self.topdir,'iterno.p')
        if os.path.exists(iterplace):
            iterfile = open(iterplace)
            self.iterno = cpkl.load(iterfile)

        # create files for outputting timing data for performance evaluation
        self.calctime = os.path.join(self.topdir, 'calctimer.txt')
#         if os.path.exists(self.calctime):
#             os.remove(self.calctime)
        self.plottime = os.path.join(self.topdir, 'plottimer.txt')
#         if os.path.exists(self.plottime):
#             os.remove(self.plottime)
#         self.iotime = os.path.join(self.testdir, 'iotimer.txt')
#         if os.path.exists(self.iotime):
#             os.remove(self.iotime)

        # load and parse data
        self.proc_data()

        # name of test for plots
        if 'name' in indict:
            self.name = indict['name']
        else:
            self.name = 'Test'

        # initialization schemes
        if 'inits' in indict:
            self.inits = indict['inits'][0]
        else:
            self.inits = 'gs'#corresponding to 'ps', 'gm'

        # construct alternative estimators
        self.alternatives()

        # generate prior distribution
        self.make_prior(indict)

        # set parameters for MCMC
        self.setup_mcmc(indict)

        # colors for plots
        self.colors='brgycm'

        # write important information about the test to a file
        outdict = {
            'topdir': self.topdir,
            'binends': self.binends,
            'logpdfs': self.logpdfs,
            'inits': self.inits,
            'miniters': self.miniters,
            'thinto': self.thinto,
            'ivals': self.ivals,
            'mean': self.mean,
            'covmat': self.covmat
            }

        with open(os.path.join(self.topdir,'README.md'), 'a') as readme:
            readme.write('\n')
            readme.write(repr(outdict))

        print(self.name+' ingested inputs and initialized sampling')

    def proc_data(self):

        with open(os.path.join(self.datadir,'logdata.csv'),'rb') as csvfile:
            tuples = (line.split(None) for line in csvfile)
            alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]

        self.binends = np.array(alldata[0])
        self.nbins = len(self.binends)-1
        self.binlos = self.binends[:-1]
        self.binhis = self.binends[1:]
        self.bindifs = self.binhis-self.binlos
        self.bindif = sum(self.bindifs)/self.nbins
        self.binmids = (self.binlos+self.binhis)/2.

        # number of walkers
        self.nwalkers = 2*self.nbins

        # read in and normalize PDFS
        self.logpdfs = np.array(alldata[2:])
        self.pdfs = np.exp(self.logpdfs)
        sums = np.sum(self.pdfs*self.bindifs,axis=1)
        self.pdfs = np.divide(self.pdfs,sums[:, np.newaxis])
        self.logpdfs = um.safelog(self.pdfs)

#         sums = self.logpdfs.sum(axis=0)
#         for gal in self.logpdfs:
#             gal = gal/self.bindifs
#             gal = gal/sum(gal*self.bindifs)
#             assert sum(gal*self.bindifs)==1.

        self.ngals = len(self.logpdfs)

        self.logintNz = np.array(alldata[1])
        self.intNz = np.exp(self.logintNz)
        self.lik_intNz = self.calclike(self.logintNz)

        self.fltNz = np.array([float(self.ngals)/float(self.nbins)/self.bindif]*self.nbins)
        self.logfltNz = np.log(self.fltNz)

        self.truZs = None
        self.truNz,self.logtruNz = None,None
        self.truPz,self.logtruPz = None,None
        self.zrange = np.arange(self.binends[0],self.binends[-1],1./self.ngals)[:, np.newaxis]

        if os.path.exists(os.path.join(self.datadir,'logtrue.csv')):
            with open(os.path.join(self.datadir,'logtrue.csv'),'rb') as csvfile:
                tuples = (line.split(None) for line in csvfile)
                trudata = [float(pair[k]) for k in range(0,len(pair)) for pair in tuples]
            self.truZs = np.array(trudata)

            bw=0.04
            kde = skl.neighbors.KernelDensity(kernel='gaussian', bandwidth=bw)
            self.trange = self.truZs[:, np.newaxis]
            self.trukde = kde.fit(self.trange)
            self.lPz_range = self.trukde.score_samples(self.zrange)
            self.Pz_range = np.exp(self.lPz_range)
            self.Nz_range = self.ngals*self.Pz_range
            self.lNz_range = um.safelog(self.Nz_range)

            truNz = [sys.float_info.epsilon]*self.nbins
            for z in self.truZs:
                for k in xrange(self.nbins):
                    if z > self.binlos[k] and z < self.binhis[k]:
                        truNz[k] += 1./self.bindifs[k]
            self.truNz = np.array(truNz)
            self.logtruNz = np.log(self.truNz)
            self.truPz = self.truNz/np.sum(self.truNz)
            self.logtruPz = np.log(self.truPz)

        self.samples = os.path.join(self.topdir, 'samples.csv')
#         with open(self.samples,'wb') as csvfile:
#             out = csv.writer(csvfile,delimiter=' ')
#             out.writerow(self.binends)

        return

    def make_prior(self,indict):

#         self.cands = [self.logintNz,self.logstkNz,self.logmapNz]#,self.logexpNz])
#         self.liks = [self.lik_intNz,self.lik_stkNz,self.lik_mapNz]#,self.lik_expNz])
        self.start = self.logintNz#self.cands[np.argmax(np.array(self.liks))]
        self.lik_mmlNz,self.logmmlNz = self.makemml()
        self.mmlNz = np.exp(self.logmmlNz)

        self.q = None
        self.e = None
        self.t = None
        # prior specification
        if 'priormean' in indict and 'priorcov' in indict:
            mean = indict['priormean']
            self.mean = np.array([float(mean[i]) for i in range(0,self.nbins)])
            covmat = indict['priorcov']
            self.covmat = np.reshape(np.array([float(covmat[i]) for i in range(0,self.nbins**2)]),(self.nbins,self.nbins))
        else:
            self.mean = self.logintNz#self.logmmlNz#self.logstkNz#self.logfltNz
            self.q = 1.0#self.bindif
            self.e = 100.#1./self.bindif**2
            self.t = self.q*1e-5
            self.covmat = np.array([[self.q*np.exp(-0.5*self.e*(self.binmids[a]-self.binmids[b])**2.) for a in xrange(0,self.nbins)] for b in xrange(0,self.nbins)])+self.t*np.identity(self.nbins)

        self.priordist = um.mvn(self.mean,self.covmat)

        # posterior specification for sampler and alternatives
        self.postdist = um.post(self.priordist, self.binends, self.logpdfs,self.logintNz)

        # sampler specification
        self.nps = mp.cpu_count()
#         print 'lnprob_ext: ' + str(self.postdist.lnprob_ext)
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nbins, self.postdist.lnprob_ext, args=[self.postdist])#, threads=self.nps, pool=self.pool)
        #generate initial values for walkers
        if self.inits == 'ps':
            self.ivals,self.mean = self.priordist.sample_ps(self.nwalkers)
            self.init_names = 'Prior Samples'

        elif self.inits == 'gm':
            self.ivals,self.mean = self.priordist.sample_gm(self.nwalkers)
            self.init_names = 'Gaussian Ball Around Mean'

        elif self.inits == 'gs':
            self.ivals,self.mean = self.priordist.sample_gs(self.nwalkers)
            self.init_names = 'Gaussian Ball Around Prior Sample'

        self.ivals_dir = os.path.join(self.topdir,'ivals.p')

        with open(self.ivals_dir,'wb') as ival_file:
            cpkl.dump(self.ivals,ival_file)

#         start_time = timeit.default_timer()
#         print('starting optimization at '+str(start_time))
#         self.cands,self.liks = [],[]
#         for samp in self.priordist.sample_ps(self.nwalkers)[0]:
#             self.cands.append(samp)
#             self.liks.append(self.postdist.lnlike(samp))
#         print(self.liks)
#         self.logstart = self.cands[np.argmax(np.array(self.liks))]
#         print(self.logstart)
#         self.lik_mml,self.mml = self.makemml('slsqp')
#         print(self.mml)
#         elapsed = timeit.default_timer() - start_time
#         print('optimized '+str(self.ngals)+' in '+str(elapsed))

        return

    def calclike(self,theta):
        constterm = np.log(self.bindifs)-self.logintNz
        constterms = theta+constterm
        sumterm = -1.*np.dot(np.exp(theta),self.bindifs)
        for j in xrange(self.ngals):
            logterm = np.log(np.sum(np.exp(self.logpdfs[j]+constterms)))
            sumterm += logterm
        return sumterm

    def makemml(self):

        if os.path.exists(os.path.join(self.datadir,'logmmle.csv')):
            with open(os.path.join(self.datadir,'logmmle.csv'),'rb') as csvfile:
                tuples = (line.split(None) for line in csvfile)
                mmldata = np.array([[float(pair[k]) for k in range(0,len(pair))] for pair in tuples])
            like = mmldata[0]
            loc = mmldata[1]
        else:
            def minlf(theta):
                return -1.*self.calclike(theta)
            def maxruns():
                return(self.ngals*2**self.nbins)
            start_time = timeit.default_timer()
            loc = sp.optimize.fmin(minlf,self.start,maxiter=maxruns(),maxfun=maxruns(), disp=True)
            like = self.calclike(loc)
            elapsed = timeit.default_timer() - start_time
            with open(os.path.join(self.datadir,'logmmle.csv'),'wb') as csvfile:
                out = csv.writer(csvfile,delimiter=' ')
                out.writerow([elapsed])
                out.writerow([like])
                out.writerow(loc)
            with open(self.calctime,'w') as calctimer:
                calctimer.write(str(elapsed)+' MMLE for '+str(self.nbins)+'\n')
                calctimer.close()
              #print(str(self.ngals)+' galaxies for '+self.name+' MMLE in '+str(elapsed)+': '+str(loc))
        return(like,loc)

    def alternatives(self):

#         # generate full Sheldon, et al. 2011 "posterior"
        stkprep = np.sum(np.array(self.pdfs),axis=0)
        self.stkNz = np.array([max(sys.float_info.epsilon,stkprep[k]) for k in xrange(self.nbins)])
        self.logstkNz = np.log(self.stkNz)
        self.lik_stkNz = self.calclike(self.logstkNz)

#         # generate MAP N(z)
#         self.mapNz = [sys.float_info.epsilon]*self.nbins
#         mappreps = [np.argmax(l) for l in self.logpdfs]
#         for m in mappreps:
#               self.mapNz[m] += 1./self.bindifs[m]
#         self.logmapNz = np.log(self.mapNz)
#         self.lik_mapNz = self.calclike(self.logmapNz)

# #         # generate expected value N(z)
# #         expprep = [sum(z) for z in self.binmids*self.pdfs*self.bindifs]
# #         self.expNz = [sys.float_info.epsilon]*self.nbins
# #         for z in expprep:
# #               for k in xrange(self.nbins):
# #                   if z > self.binlos[k] and z < self.binhis[k]:
# #                       self.expNz[k] += 1./self.bindifs[k]
# #         self.logexpNz = np.log(self.expNz)
# #         self.lik_expNz = self.calclike(self.logexpNz)

#         return

    def setup_mcmc(self,indict):

        # enable plotting without sampling
        if 'plotonly' in indict:
            self.plotonly = bool(int(indict['plotonly'][0]))
            if self.plotonly == True:
                self.iterno = self.key.load_iterno(self.topdir)
        else:
            self.plotonly = bool(0)

        if 'miniters' in indict:
            self.miniters = 10**int(indict['miniters'])
        else:
            self.miniters = int(1e3)

        if 'thinto' in indict:
            self.thinto = int(indict['thinto'])
        else:
            self.thinto = 1

        assert(self.miniters%self.thinto==0)

        self.ntimes = self.miniters / self.thinto

        if 'factor' in indict:
            self.factor = int(indict['factor'])
        else:
            self.factor = 2

        #assert(self.ntimes > self.nwalkers)
        # autocorrelation time mode
        if 'mode' in indict:
            self.mode = int(indict['mode'])
        else:
            self.mode = 'bins'#'walkers'

        # what outputs of emcee will we be saving?
        self.stats = [ stats.stat_both(self),
                       stats.stat_chains(self),
                       stats.stat_probs(self),
                       stats.stat_fracs(self),
                       stats.stat_times(self) ]
        return

    # retrieve last saved state
    def get_last_state(self):
        iterno = self.key.load_iterno(self.topdir)
        print('getting state:' + str(self.key))
        state = self.key.add(r = iterno).load_state(self.topdir)
        if state is not None:
            print ('state restored at: {}/{} runs', state.runs, self.key.r)
            return state
        return pertest(self)

    # retrieve all saved states
    def get_all_states(self):
        iterno = self.key.load_iterno(self.meta.topdir)
        if iterno is None:
            print ("Oops, I couldn't find the number of iterations, assuming 0")
            return []
        print('getting state:' + str(self.key))
        # check this: is there an off-by-one here?
        states = [self.key.add(r=r).load_state(self.topdir) for r in xrange(iterno)]
        return states

    # update goodness of fit tests calculated at each set of iterations
    def load_fitness(self, category):
        iterno = self.key.load_iterno(self.topdir)
        vars = ['tot_ls', 'tot_s', 'var_ls', 'var_s']
        retval = {var : [] for var in vars}
        if iterno is None:
            print ("Oops, I couldn't find the number of iterations, assuming 0")
            return retval
        # TO DO: this currently returns a list of tuples, rather than a tuple of lists.
        fitness_list =  self.key.load_stats(self.topdir, category, iterno)
        for per_iter in fitness_list:
            print ("per_iter: {}".format(per_iter))
            for var in vars:
                if isinstance(per_iter[var], list):
                    retval[var].extend(per_iter[var])
                else:
                    retval[var].append(per_iter[var])
        return retval

    # sample this using information defined for each run of MCMC
    def samplings(self):
        return pertest(self).samplings()

    def plotonlues(self):
        return pertest(self).plotonlies()
