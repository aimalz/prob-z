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

from utilmcmc import *
import keymcmc as key
from permcmc import pertest
import statmcmc as stats

# setup object contains inputs of necessary parameters for code
class setup(object):
    """
    setup object specifies all parameters controlling one run of p(z) inference program
    """
    def __init__(self,input_address):

        # name the test for which this is setup object
        self.key = key.key(t=input_address)

        # read input parameters
        self.testdir = os.path.join('..','tests')
        with open(os.path.join(self.testdir,input_address)) as infile:
            lines = (line.split(None) for line in infile)
            indict   = {defn[0]:defn[1] for defn in lines}

        # load top directory location
        with open(os.path.join(self.testdir,'topdirs.p')) as topdirs:
            self.topdir = cpkl.load(topdirs)[input_address]
        self.datadir = os.path.join(self.topdir,'data')

        # make directory into which to put output of this test
        self.topdir = os.path.join(self.topdir,'mcmc')
        if os.path.exists(self.topdir):
            shutil.rmtree(self.topdir)
        os.makedirs(self.topdir)

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
            'logpobs': self.logpobs,
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

        # create files for outputting timing data for performance evaluation
        self.calctime = os.path.join(self.testdir, 'calctimer.txt')
        if os.path.exists(self.calctime):
            os.remove(self.calctime)
        self.plottime = os.path.join(self.testdir, 'plottimer.txt')
        if os.path.exists(self.plottime):
            os.remove(self.plottime)
#         self.iotime = os.path.join(self.testdir, 'iotimer.txt')
#         if os.path.exists(self.iotime):
#             os.remove(self.iotime)

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

        self.logpobs = np.array(alldata[2:])
        self.pobs = np.exp(self.logpobs)
        self.ngals = len(self.logpobs)

        self.logintNz = np.array(alldata[1])
        self.intNz = np.exp(self.logintNz)
        self.lik_intNz = self.calclike(self.logintNz)

        self.flatNz = np.array([float(self.ngals)/float(self.nbins)/self.bindif]*self.nbins)
        self.logflatNz = np.log(self.flatNz)

        self.trueZs = None
        self.trueNz,self.logtrueNz = None,None
        self.truePz,self.logtruePz = None,None
        if os.path.exists(os.path.join(self.datadir,'logtrue.csv')):
            with open(os.path.join(self.datadir,'logtrue.csv'),'rb') as csvfile:
                tuples = (line.split(None) for line in csvfile)
                trudata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
            self.trueZs = np.array(trudata[2:])

            trueNz = [sys.float_info.epsilon]*self.nbins
            for z in self.trueZs:
                for k in xrange(self.nbins):
                    if z[0] > self.binlos[k] and z[0] < self.binhis[k]:
                        trueNz[k] += 1./self.bindifs[k]
            self.trueNz = np.array(trueNz)
            self.logtrueNz = np.log(self.trueNz)
            self.truePz = self.trueNz/np.sum(self.trueNz)
            self.logtruePz = np.log(self.truePz)
        return

    def make_prior(self,indict):

        self.cands = np.array([self.logintNz,self.logstack,self.logmapNz])#,self.logexpNz])
        self.liks = np.array([self.lik_intNz,self.lik_stack,self.lik_mapNz])#,self.lik_expNz])
        self.logstart = self.cands[np.argmax(self.liks)]

        self.ml,self.logmleNz = self.makemle(arg='slsqp')#'grid','cobyla','slsqp'
        self.mleNz = np.exp(self.logmleNz)

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
            self.mean = self.logmleNz#self.logstack#self.logflatNz#self.logintNz
            self.q = 1.0#self.bindif
            self.e = 100.#1./self.bindif**2
            self.t = self.q*1e-5
            self.covmat = np.array([[self.q*np.exp(-0.5*self.e*(self.binmids[a]-self.binmids[b])**2.) for a in xrange(0,self.nbins)] for b in xrange(0,self.nbins)])+self.t*np.identity(self.nbins)

        self.priordist = mvn(self.mean,self.covmat)

        # posterior specification for sampler and alternatives
        self.postdist = post(self.priordist, self.binends, self.logpobs,self.logintNz)

        # sampler specification
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nbins, self.postdist.lnprob)

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

        return

    def calclike(self,theta):
        constterm = np.log(self.bindifs)-self.logintNz
        constterms = theta+constterm
        sumterm = -1.*np.dot(np.exp(theta),self.bindifs)
        for j in xrange(self.ngals):
            logterm = np.log(np.sum(np.exp(self.logpobs[j]+constterms)))
            sumterm += logterm
        return sumterm

    def makemle(self,arg):
        start_time = timeit.default_timer()
        if arg == 'grid':
            def cons1(theta):
                if np.dot(np.exp(theta),self.bindifs) >= 0.5*self.ngals:
                    return True
                else:
                    return False
            def cons2(theta):
                if np.dot(np.exp(theta),self.bindifs) <= 1.5*self.ngals:
                    return True
                else:
                    return False
            r = 0.5
            grid = np.arange(0.,np.log(self.ngals/self.bindif)+r,r)
            vals = itertools.combinations_with_replacement(grid,self.nbins)
            grids = []
            for g in vals:
                if cons1(g) and cons2(g):
                    grids.append(g)
            locs,likes = [],[]
            for val in grids:
                locs.append(val)
                likes.append(self.calclike(val))
            locs = np.array(locs)
            likes = np.array(likes)
            ind = np.argmax(likes)
            like = likes[ind]
            loc = locs[ind]
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
            loc = sp.optimize.fmin_cobyla(minlf,self.logstart,cons=(cons1,cons2,cons3,cons4),maxfun=100000)
            like = self.calclike(loc)
        if arg == 'slsqp':
            def cons1(theta):
                return np.dot(np.exp(theta),self.bindifs)-0.5*self.ngals
            def cons2(theta):
                return 1.5*self.ngals-np.dot(np.exp(theta),self.bindifs)
            def minlf(theta):
                return -1.*self.calclike(theta)
            bounds = [(-sys.float_info.epsilon,np.log(self.ngals)) for k in xrange(self.nbins)]
            loc = sp.optimize.fmin_slsqp(minlf,self.logstart,ieqcons=([cons1,cons2]),bounds=bounds,iter=100)
            like = self.calclike(loc)
        elapsed = timeit.default_timer() - start_time
        print(str(self.ngals)+' galaxies for '+self.name+' MLE by '+arg+' in '+str(elapsed))
        return(like,loc)

    def alternatives(self):

        # generate full Sheldon, et al. 2011 "posterior"
        stackprep = np.sum(np.array(self.pobs),axis=0)
        self.stack = np.array([max(sys.float_info.epsilon,stackprep[k]) for k in xrange(self.nbins)])
        self.logstack = np.log(self.stack)
        self.lik_stack = self.calclike(self.logstack)

        # generate MAP N(z)
        self.mapNz = [sys.float_info.epsilon]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpobs]
        for m in mappreps:
              self.mapNz[m] += 1./self.bindifs[m]
        self.logmapNz = np.log(self.mapNz)
        self.lik_mapNz = self.calclike(self.logmapNz)

        # generate expected value N(z)
        expprep = [sum(z) for z in self.binmids*self.pobs*self.bindifs]
        self.expNz = [sys.float_info.epsilon]*self.nbins
        for z in expprep:
              for k in xrange(self.nbins):
                  if z > self.binlos[k] and z < self.binhis[k]:
                      self.expNz[k] += 1./self.bindifs[k]
        self.logexpNz = np.log(self.expNz)
        self.lik_expNz = self.calclike(self.logexpNz)

        return

    def setup_mcmc(self,indict):

        if 'miniters' in indict:
            self.miniters = int(indict['miniters'])
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
