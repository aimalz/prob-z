"""
in-mcmc module contains parameters controlling one run of MCMC p(z) inference program
"""

# TO DO: add docstrings

import os
import cPickle as cpkl
import math as m
import numpy as np
import shutil
import emcee
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

        self.key = key.key(t=input_address)

        # read input parameters
        self.testdir = os.path.join('..','tests')
        with open(os.path.join(self.testdir,input_address)) as infile:
            lines = (line.split(None) for line in infile)
            indict   = {defn[0]:defn[1] for defn in lines}

        # load data
        with open(os.path.join(self.testdir,'topdirs.p')) as topdirs:
            self.topdir = cpkl.load(topdirs)[input_address]
        self.datadir = os.path.join(self.topdir,'data')

        self.topdir = os.path.join(self.topdir,'mcmc')
        if os.path.exists(self.topdir):
            shutil.rmtree(self.topdir)
        os.makedirs(self.topdir)

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

        self.loginterim = np.array(alldata[1])
        self.interim = np.exp(self.loginterim)

        self.logpobs = np.array(alldata[2:])
        self.pobs = np.exp(self.logpobs)
        self.ngals = len(self.logpobs)
        self.flatNz = np.array([float(self.ngals)/float(self.nbins)/self.bindif]*self.nbins)
        self.logflatNz = np.log(self.flatNz)

        self.trueZs = None
        self.trueNz = None
        self.logtrueNz = None
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

        # generate full Sheldon, et al. 2011 "posterior"
        stackprep = np.sum(np.array(self.pobs),axis=0)
        self.stack = np.array([max(sys.float_info.epsilon,stackprep[k]) for k in xrange(self.nbins)])
        self.logstack = np.log(self.stack)

        # generate MAP N(z)
        self.mapNz = [sys.float_info.epsilon]*self.nbins
        mappreps = [np.argmax(l) for l in self.logpobs]
        for m in mappreps:
              self.mapNz[m] += 1./self.bindifs[m]
        self.logmapNz = np.log(self.mapNz)

        # generate expected value N(z)
        expprep = [sum(z) for z in self.binmids*self.pobs*self.bindifs]
        self.expNz = [sys.float_info.epsilon]*self.nbins
        for z in expprep:
              for k in xrange(self.nbins):
                  if z > self.binlos[k] and z < self.binhis[k]:
                      self.expNz[k] += 1./self.bindifs[k]
        self.logexpNz = np.log(self.expNz)
        #print('logexpNz='+str(self.logexpNz))

        # how many walkers
        self.nwalkers = 2*self.nbins
        #self.walknos = xrange(self.nwalkers)

        # prior specification
        if 'priormean' in indict and 'priorcov' in indict:
            mean = indict['priormean']
            self.mean = np.array([float(mean[i]) for i in range(0,self.nbins)])
            covmat = indict['priorcov']
            self.covmat = np.reshape(np.array([float(covmat[i]) for i in range(0,self.nbins**2)]),(self.nbins,self.nbins))
        else:
#             if 'prior' in indict and bool(int(indict['prior'][0])) == True:
#                 mean = self.ngals*np.array([z**2*np.exp(-(z/0.5)**1.5) for z in self.binmids])/self.bindifs
#                 self.mean = np.log(np.array([max(x,0.) for x in mean]))
#             else:
            self.mean = self.loginterim
            if 'random' in indict and bool(int(indict['random'])) == True:
#                 q = 1.
#                 e = 1./self.bindif**2
#                 tiny = q*1e-6
#                 self.covmat = np.array([[q*np.exp(-0.5*e*(self.binmids[a]-self.binmids[b])**2.) for a in xrange(0,self.nbins)] for b in xrange(0,self.nbins)])+tiny*np.identity(self.nbins)
                self.covmat = np.identity(self.nbins)
            else:
                self.covmat = np.identity(self.nbins)

        self.priordist = mvn(self.mean,self.covmat)

        # posterior specification
        self.postdist = post(self.priordist, self.binends, self.logpobs,self.loginterim)

        # sampler specification
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nbins, self.postdist.lnprob)

        # initialization schemes
        if 'inits' in indict:
            self.inits = indict['inits']
        else:
            self.inits = 'gs'#corresponding to 'ps', 'gm'

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

        # parameters for MCMC
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
        self.factor = 2

        #assert(self.ntimes > self.nwalkers)

#         # what outputs of emcee will we be saving?
        self.stats = [ stats.stat_chains(self),
                       stats.stat_probs(self),
                       stats.stat_fracs(self),
                       stats.stat_times(self) ]

        # colors for plots
        self.colors='brgycm'

        # autocorrelation time mode
        if 'mode' in indict:
            self.mode = int(indict['mode'])
        else:
            self.mode = 'bins'#'walkers'

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

        self.calctime = os.path.join(self.testdir, 'calctimer.txt')
        if os.path.exists(self.calctime):
            os.remove(self.calctime)
        self.plottime = os.path.join(self.testdir, 'plottimer.txt')
        if os.path.exists(self.plottime):
            os.remove(self.plottime)
#         self.iotime = os.path.join(self.testdir, 'iotimer.txt')
#         if os.path.exists(self.iotime):
#             os.remove(self.iotime)

        print('ingested inputs and initialized sampling')

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
            print ("BLOODY HELL, I couldn't find the number of iterations, assuming 0")
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
            print ("BLOODY HELL, I couldn't find the number of iterations, assuming 0")
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
