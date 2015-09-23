# this hideous file makes the plots

import distribute
import matplotlib
import matplotlib.pyplot as plt
import math as m
import numpy as np
from util import *
import os
import statistics
import timeit
import random

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        self.plot(key)

def postplot(meta, runs, dist, idinfo):
    r_run = runs.i_runs[idinfo].getstate()
    for r in r_run.stepnos:
        runinfo = idinfo + (r,)
        dist.complete_chunk(key(runinfo))

def plotall(allinfo, ):
    pass

# plot autocorrelation times
class plotter_times(plotter):

    def __init__(self, meta, p_run, s_run, n_runs, i_runs):
        self.meta = meta
        self.p_run = p_run
        self.s_run = s_run
        self.n_runs = n_runs
        self.f = plt.figure(figsize=(5,5))
        self.sps = self.f.add_subplot(1,1,1)
        self.a_times = 2./self.p_run.ndims / self.meta.samps
        self.sps.set_title(str(self.meta.samps) + 'Autocorrelation Times for ' + str(s_run.seed) + ' galaxies')
        self.sps.set_ylabel('autocorrelation time')
        self.sps.set_xlabel('number of iterations')
        self.sps.set_ylim(0, 100)
        for i in lrange(self.meta.inits):
            self.sps.scatter([0], [-1], c=self.meta.colors[i], label=self.meta.init_names[i], linewidths=0.1)
        self.sps.legend(fontsize = 'small', loc = 'upper right')

    def plot(self, key):

        data = key.load_state(self.meta.topdir)['times']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.n_runs[key.n].nbins,#[i_run.iternos[r]]*meta.params[p_run.p],
                         plot_y,
                         c=self.meta.colors[key.i],
                         alpha=self.a_times,
                         linewidth=0.1,
                         s=self.n_runs[key.n].nbins,
                         rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          plottimer.write(str(timeit.default_timer())+' '+str(key)+' times \n')
          plottimer.close()

    def finish(self):
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.s_run.get_dir(),'times.png'),dpi=100)

# plot acceptance fractions
class plotter_fracs(plotter):

    def __init__(self, meta, p_run, s_run, n_runs, **_):
        self.meta = meta
        self.p_run = p_run
        self.s_run = s_run
        self.n_runs = n_runs
        self.a_fracs = 1./self.p_run.ndims/self.meta.samps
        self.f = plt.figure(figsize=(5,5))
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title(str(self.meta.samps) + ' Acceptance Fractions for ' + str(self.s_run.seed) + ' galaxies')
        self.sps.set_ylim(0,1)
        self.sps.set_ylabel('acceptance fraction')
        self.sps.set_xlabel('number of iterations')
        for i in lrange(self.meta.inits):
            self.sps.scatter([0],
                        [-1],
                        c=self.meta.colors[i],
                        label=self.meta.init_names[i],
                        linewidths=0.1,
                        s=self.p_run.ndims)
        self.sps.legend(fontsize = 'small')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['fracs']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.n_runs[key.n].nwalkers,#[i_run.iternos[r]] * n_run.nwalkers,
                         plot_y,
                         c=self.meta.colors[key.i],
                         alpha=self.a_fracs,
                         linewidth=0.1,
                         s=self.p_run.ndims,
                         rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          plottimer.write(str(timeit.default_timer())+' '+str(key)+' fracs \n')
          plottimer.close()

    def finish(self):
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.s_run.get_dir(),'fracs.png'),dpi=100)

# plot log probabilities of samples of full posterior
class plotter_probs(plotter):

    def __init__(self, meta, p_run, s_run, n_runs, **_):
        self.meta = meta
        self.p_run = p_run
        self.s_run = s_run
        self.n_runs = n_runs
        self.a_probs = 1./len(self.meta.colors)/self.meta.samps
        self.f = plt.figure(figsize=(5,5))
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title(r'Sample probability Evolution for $J_{0}=' + str(s_run.seed) + r'$')
        self.sps.set_ylabel('log probability of walker')
        self.sps.set_xlabel('iteration number')

        #self.sps.set_ylim(-self.s_run.seed, self.s_run.seed*m.log(self.s_run.seed))
        dummy_rec = [0]*self.meta.miniters
        for i in lrange(self.meta.inits):
            self.sps.plot([-1.] * self.meta.miniters,
                     dummy_rec,
                     c=self.meta.colors[i],
                     label = self.meta.init_names[i])
        self.sps.legend(fontsize='x-small', loc='lower right')

    def plot(self,key):
        print('probs plot key '+str(key))

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1).T

        for w in self.n_runs[key.n].walknos:
            self.sps.plot(np.arange(key.r*self.meta.miniters/self.meta.thinto,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#,(key.r+1)*self.meta.miniters),#key.i_run.eachtimenos[r],
                     plot_y[w],
                     c=self.meta.colors[key.i],
                     alpha=self.a_probs,
                     rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          plottimer.write(str(timeit.default_timer())+' '+str(key)+' probs \n')
          plottimer.close()

    def finish(self):
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.s_run.get_dir(),'probs.png'),dpi=100)

# plot full posterior samples and chain evolution
class plotter_chains(plotter):

    def __init__(self, meta, p_run, s_run, n_runs, i_runs):
        self.meta = meta
        self.p_run = p_run
        self.s_run = s_run
        self.n_runs = n_runs
        self.a_samp = 1. / len(self.meta.inits) / len(self.meta.colors)
        self.a_chain = 1. / len(self.meta.inits) / len(self.meta.colors)
        self.f_samps = plt.figure(figsize=(5*self.meta.samps, 5*2))
        self.gs_samps = matplotlib.gridspec.GridSpec(2, self.meta.samps)
        self.sps_samps = [[self.f_samps.add_subplot(self.gs_samps[l,n]) for n in xrange(self.meta.samps)] for l in range(0,2)]
        self.maxk = max(n_run.nbins for n_run in self.n_runs)
        self.f_chains = plt.figure(figsize=(5*self.maxk, 5*self.meta.samps))
        self.gs_chains = matplotlib.gridspec.GridSpec(self.meta.samps, self.maxk)
        self.sps_chains = [[self.f_chains.add_subplot(self.gs_chains[n,k]) for k in range(self.maxk)] for n in xrange(self.meta.samps)]
        self.init_data = {'meta': meta,
                          'p_run': p_run,
                          's_run': s_run,
                          'n_runs': n_runs,
                          'i_runs': i_runs}
        dummy_chain = [-1.] * meta.miniters

        for n in xrange(self.meta.samps):
            self.sps_samps[0][n].set_xlim(self.n_runs[n].binends[0]-self.meta.zdif,self.n_runs[n].binends[-1]+self.meta.zdif)
            self.sps_samps[0][n].set_ylim(-1.,m.log(self.s_run.seed/self.meta.zdif)+1.)
            self.sps_samps[0][n].set_xlabel(r'$z$')
            self.sps_samps[0][n].set_ylabel(r'$\ln N(z)$')
            self.sps_samps[0][n].set_title(str(n+1)+r' Sampled $\ln N(z)$ for $J_{0}='+str(self.s_run.seed)+r'$')
            self.sps_samps[1][n].set_xlim(self.n_runs[n].binends[0]-self.meta.zdif,self.n_runs[n].binends[-1]+self.meta.zdif)
            self.sps_samps[1][n].set_ylim(0.,self.s_run.seed/self.meta.zdif+self.s_run.seed)
            self.sps_samps[1][n].set_xlabel(r'$z$')
            self.sps_samps[1][n].set_ylabel(r'$N(z)$')
            self.sps_samps[1][n].set_title(str(n+1)+r' Sampled $N(z)$ for $J_{0}='+str(self.s_run.seed)+r'$')

            maxn = max(self.n_runs[n].binnos for n in xrange(self.meta.samps))
            sps_chain = self.sps_chains[n]
            for k in xrange(self.n_runs[n].nbins):
                sps_chain[k].set_ylim(-m.log(self.s_run.seed), m.log(self.s_run.seed / self.meta.zdif)+1)
                sps_chain[k].set_xlabel('iteration number')
                sps_chain[k].set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
                sps_chain[k].set_title('Sample {} of {} galaxies: Parameter {} of {}'.format(n+1, self.s_run.seed, k+1, self.n_runs[n].nbins))
                for i in lrange(self.meta.inits):
                    sps_chain[k].plot([-1.]*self.meta.miniters,
                                      dummy_chain,
                                      color=self.meta.colors[i],
                                      label=self.meta.init_names[i])
                sps_chain[k].legend(fontsize='small', loc='upper right')

    def plot(self,key):
        print('chains plot key '+str(key))
        if not key.burnin:
            return

        data = key.load_state(self.meta.topdir)['chains']
        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)
        plot_y_c = plot_y_ls.T

        randwalks = random.sample(self.n_runs[key.n].walknos, len(self.meta.colors))

        for x in xrange(self.meta.miniters/self.meta.thinto):
            for w in randwalks:
                self.sps_samps[0][key.n].hlines(plot_y_ls[x][w],
                                                  self.n_runs[key.n].binlos,
                                                  self.n_runs[key.n].binhis,
                                                  color=self.meta.colors[key.i],
                                                  alpha=self.a_samp,
                                                  rasterized=True)
                self.sps_samps[1][key.n].hlines(plot_y_s[x][w],
                                                  self.n_runs[key.n].binlos,
                                                  self.n_runs[key.n].binhis,
                                                  color=self.meta.colors[key.i],
                                                  alpha=self.a_samp,
                                                  rasterized=True)
                for k in self.n_runs[key.n].binnos:
                    self.sps_chains[key.n][k].plot(np.arange(key.r*self.meta.miniters/self.meta.thinto,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#i_run.eachtimenos[r],
                                                     plot_y_c[k][w],
                                                     color = self.meta.colors[key.i],
                                                     alpha = self.a_chain,
                                                     rasterized = True)
        with open(self.meta.plottime,'a') as plottimer:
            plottimer.write(str(timeit.default_timer())+' '+str(key)+' chains \n')
            plottimer.close()

    def finish(self):
        meta = self.init_data['meta']
        p_run = self.init_data['p_run']
        s_run = self.init_data['s_run']
        n_runs = self.init_data['n_runs']
        i_runs = self.init_data['i_runs']
        r_runs = [i_run.get_all_states() for i_run in i_runs]
        for n in xrange(meta.samps):
            n_run = n_runs[n]
            dummy_lnsamp = [-1.] * n_run.nbins
            dummy_samp = [0.] * n_run.nbins
            self.sps_samps[0][n].step(n_run.binhis,
                                      n_run.full_logflatNz,
                                      color='k',
                                      alpha=0.5,
                                      label=r'Flat $\ln N(z)$',
                                      linewidth=2,
                                      #where='pre',
                                      linestyle='--')

            self.sps_samps[1][n].step(n_run.binhis,
                                      n_run.full_flatNz,
                                      color='k',
                                      alpha=0.5,
                                      label=r'Flat $N(z)$',
                                      linewidth=2,
                                      #where='pre',
                                      linestyle='--')
            logdifstack = n_run.logstack - n_run.full_logsampNz
            difstack = n_run.stack - n_run.full_sampNz
            logvarstack = np.dot(logdifstack, logdifstack) / n_run.nbins
            varstack = np.dot(difstack, difstack), n_run.nbins

            self.sps_samps[0][n].step(n_run.binhis,
                                      n_run.logstack,
                                      color='k',
                                      alpha=0.5,
                                      linewidth=2,
                                      #where='pre',
                                      label=r'Stacked $\ln N(z)$ $\sigma^{2}='+str(logvarstack)+r'$')
            self.sps_samps[1][n].step(n_run.binhis,
                                      n_run.stack,
                                      color='k',
                                      alpha=0.5,
                                      linewidth=2,
                                      #where='pre',
                                      label=r'Stacked $N(z)$ $\sigma^{2}='+str(varstack[0])+r'$')
            self.sps_samps[0][n].step(n_run.binhis,
                                      n_run.full_logsampNz,
                                      color='k',
                                      linewidth=2,
                                      #where='pre',
                                      label=r'True $\ln N(z)$')
            self.sps_samps[1][n].step(n_run.binhis,
                                      n_run.full_sampNz,
                                      color='k',
                                      linewidth=2,
                                      #where='pre',
                                      label=r'True $N(z)$')

            for i in lrange(meta.inits):
                i_run = i_runs[i]
                fitness = i_run.load_fitness('chains')
                tot_ls = fitness['tot_ls']
                tot_s = fitness['tot_s']
                var_ls = fitness['var_ls']
                var_s = fitness['var_s']
                self.sps_samps[0][n].hlines(dummy_lnsamp,
                                            n_run.binlos,
                                            n_run.binhis,
                                            color=meta.colors[i],
                                            label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_ls)+r'$')
                self.sps_samps[1][n].hlines(dummy_samp,
                                            n_run.binlos,
                                            n_run.binhis,
                                            color=meta.colors[i],
                                            label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_s)+r'$')
            maxsteps = self.last_key.r
            maxiternos = np.arange(0,maxsteps)
            for k in n_run.binnos:
                self.sps_chains[n][k].step(maxiternos*meta.miniters,
                                           [n_run.full_logflatNz[k]] * maxsteps,
                                           color='k',
                                           label='Flat value',
                                           linestyle='--')
                self.sps_chains[n][k].plot(maxiternos*meta.miniters,
                                           [n_run.full_logsampNz[k]] * maxsteps,
                                           color='k',
                                           linewidth=2,
                                           label='True value')
                self.sps_chains[n][k].plot(maxiternos*meta.miniters,
                                           [n_run.logstack[k]]*maxsteps,
                                           color='k',
                                           alpha=0.5,
                                           linewidth=2,
                                           label='Stacked value')
                self.sps_chains[n][k].legend(fontsize='xx-small', loc='lower right')
                self.sps_chains[n][k].set_xlim(0,(self.last_key.r+1)*meta.miniters)
            self.sps_samps[0][n].legend(fontsize='xx-small', loc='upper left')
            self.sps_samps[1][n].legend(fontsize='xx-small', loc='upper left')
            self.f_samps.savefig(os.path.join(self.s_run.get_dir(),'samps.png'),dpi=100)
            self.f_chains.savefig(os.path.join(self.s_run.get_dir(),'chains.png'),dpi=100)

# initialize all plotters
all_plotters = [plotter_times,
               plotter_fracs,
               plotter_probs,
               plotter_chains]
