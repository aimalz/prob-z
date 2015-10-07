"""
perinit module permits varying initial conditions for MCMC
"""

import os
import hickle as hkl
import cPickle
import statistics
from subprocess import call
from permcmc import *
import stats

# define class per initialization of MCMC
# will be changing this object at each iteration as it encompasses state
class perinit(object):

    def __init__(self, meta, n_run, i):

        self.meta = meta
        self.n_run = n_run
        self.s_run = self.n_run.s_run
        self.p_run = self.n_run.p_run
        self.i = i
        self.key = self.n_run.key.add(i=i)
        self.init = self.meta.inits[i]

        self.genivals()

        # what outputs of emcee will we be saving?
        self.stats = [ stats.stat_chains(self),
                       stats.stat_probs(self),
                       stats.stat_fracs(self),
                       stats.stat_times(self) ]
        n_run.i_runs.append(self)

    def genivals(self):
        if self.meta.mcmc == 0:
            on_disk = self.key.load_ivals(self.meta.topdir)
            self.iguesses = on_disk['ivals']
            self.mean = on_disk['mean']
            print('loaded initialized '+str(self.meta.init_names[self.i])+' sampling')
            return

        #generate initial values for walkers
        if self.init == 'ps':
            self.iguesses,self.mean = self.n_run.priordist.sample_ps(self.n_run.nwalkers)

        elif self.init == 'gm':
            self.iguesses,self.mean = self.n_run.priordist.sample_gm(self.n_run.nwalkers)

        elif self.init == 'gs':
            self.iguesses,self.mean = self.n_run.priordist.sample_gs(self.n_run.nwalkers)

        self.key.store_ivals(self.meta.topdir,
                            {'ivals': self.iguesses,
                             'mean': self.mean})

        print('initialized '+str(self.meta.init_names[self.i])+' sampling')

    # retrieve last saved state
    def get_last_state(self):
        iterno = self.key.load_iterno(self.meta.topdir)
        print('getting state:' + str(self.key))
        state = self.key.add(r = iterno).load_state(self.meta.topdir)
        if state is not None:
            print ('state restored at: {}/{} runs', state.runs, self.key.r)
            return state
        return permcmc(self)

    # retrieve all saved states
    def get_all_states(self):
        iterno = self.key.load_iterno(self.meta.topdir)
        if iterno is None:
            print ("BLOODY HELL, I couldn't find the number of iterations, assuming 0")
            return []
        print('getting state:' + str(self.key))
        # check this: is there an off-by-one here?
        states = [self.key.add(r=r).load_state(self.meta.topdir) for r in xrange(iterno)]
        return states

    # update goodness of fit tests calculated at each set of iterations
    def load_fitness(self, category):
        iterno = self.key.load_iterno(self.meta.topdir)
        vars = ['tot_ls', 'tot_s', 'var_ls', 'var_s']
        retval = {var : [] for var in vars}
        if iterno is None:
            print ("BLOODY HELL, I couldn't find the number of iterations, assuming 0")
            return retval
        # TO DO: this currently returns a list of tuples, rather than a tuple of lists.
        fitness_list =  self.key.load_stats(self.meta.topdir, category, iterno)
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
        return permcmc(self).samplings()
