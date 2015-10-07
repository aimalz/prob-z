"""
plots module says how to make the plots
"""

# TO DO: use enumerate instead of indices and elements
# TO DO: consider seaborn for plots

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
import psutil

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        self.plot(key)

# def postplot(meta, runs, dist, idinfo):
#     r_run = runs.i_runs[idinfo].getstate()
#     for r in r_run.stepnos:
#         runinfo = idinfo + (r,)
#         dist.complete_chunk(key(runinfo))

def plotall(allinfo, ):
    pass

# plot autocorrelation times
# TO DO: check how emcee calculates these to troubleshoot 0 autocorrelation times
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
        for n in xrange(self.meta.samps):
            for i in lrange(self.meta.inits):
                self.sps.scatter([0], [-1],
                                 c=self.meta.colors[i+n],
                                 label=self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]),
                                 linewidths=0.1)
        self.sps.legend(fontsize = 'small', loc = 'upper right')

    def plot(self, key):

        data = key.load_state(self.meta.topdir)['times']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.n_runs[key.n].nbins,#[i_run.iternos[r]]*meta.params[p_run.p],
                         plot_y,
                         c=self.meta.colors[key.n+key.i],
                         alpha=self.a_times,
                         linewidth=0.1,
                         s=self.n_runs[key.n].nbins,
                         rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          process = psutil.Process(os.getpid())
          plottimer.write('times '+str(timeit.default_timer())+' '+str(key)+' mem:'+str(process.get_memory_info())+'\n')
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
        for n in xrange(self.meta.samps):
            for i in lrange(self.meta.inits):
                self.sps.scatter([0],[-1],
                        c=self.meta.colors[i+n],
                        label=self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]),
                        linewidths=0.1,
                        s=self.p_run.ndims)
        self.sps.legend(fontsize = 'small')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['fracs']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.n_runs[key.n].nwalkers,#[i_run.iternos[r]] * n_run.nwalkers,
                         plot_y,
                         c=self.meta.colors[key.i+key.n],
                         alpha=self.a_fracs,
                         linewidth=0.1,
                         s=self.p_run.ndims,
                         rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          process = psutil.Process(os.getpid())
          plottimer.write('fracs '+str(timeit.default_timer())+' '+str(key)+' mem:'+str(process.get_memory_info())+'\n')
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
        for n in xrange(self.meta.samps):
            for i in lrange(self.meta.inits):
                self.sps.plot([-1.] * self.meta.miniters,
                              dummy_rec,
                              c=self.meta.colors[i+n],
                              label = self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]))
        self.sps.legend(fontsize='x-small', loc='lower right')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1).T

        for w in self.n_runs[key.n].walknos:
            self.sps.plot(np.arange(key.r*self.meta.miniters/self.meta.thinto,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#,(key.r+1)*self.meta.miniters),#key.i_run.eachtimenos[r],
                     plot_y[w],
                     c=self.meta.colors[key.i+key.n],
                     alpha=self.a_probs,
                     rasterized=True)
        with open(self.meta.plottime,'a') as plottimer:
          process = psutil.Process(os.getpid())
          plottimer.write('probs '+str(timeit.default_timer())+' '+str(key)+' mem:'+str(process.get_memory_info())+'\n')
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
            sps_samp_log = self.sps_samps[0][n]
            sps_samp = self.sps_samps[1][n]
            sps_samp_log.set_xlim(self.n_runs[n].binends[0]-self.meta.zdif,self.n_runs[n].binends[-1]+self.meta.zdif)
            sps_samp_log.set_ylim(-1.,m.log(self.s_run.seed/self.meta.zdif)+1.)
            sps_samp_log.set_xlabel(r'$z$')
            sps_samp_log.set_ylabel(r'$\ln N(z)$')
            sps_samp_log.set_title(str(n+1)+r' Sampled $\ln N(z)$ for $J_{0}='+str(self.s_run.seed)+r'$'+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]))
            sps_samp.set_xlim(self.n_runs[n].binends[0]-self.meta.zdif,self.n_runs[n].binends[-1]+self.meta.zdif)
            sps_samp.set_ylim(0.,self.s_run.seed/self.meta.zdif+self.s_run.seed)
            sps_samp.set_xlabel(r'$z$')
            sps_samp.set_ylabel(r'$N(z)$')
            sps_samp.set_title(str(n+1)+r' Sampled $N(z)$ for $J_{0}='+str(self.s_run.seed)+r'$'+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]))

            maxn = max(self.n_runs[n].binnos for n in xrange(self.meta.samps))
            for k in xrange(self.n_runs[n].nbins):
                sps_chain = self.sps_chains[n][k]
                sps_chain.set_ylim(-m.log(self.s_run.seed), m.log(self.s_run.seed / self.meta.zdif)+1)
                sps_chain.set_xlabel('iteration number')
                sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
                sps_chain.set_title('Sample {} of {} galaxies: Parameter {} of {}'.format(n+1, self.s_run.seed, k+1, self.n_runs[n].nbins))
                for i in lrange(self.meta.inits):
                    sps_chain.plot([-1.]*self.meta.miniters,
                                      dummy_chain,
                                      color=self.meta.colors[i],
                                      label=self.meta.init_names[i])
                sps_chain.legend(fontsize='small', loc='upper right')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['chains']
        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)
        plot_y_c = plot_y_ls.T

        randwalks = random.sample(self.n_runs[key.n].walknos, len(self.meta.colors))

        with open(self.meta.plottime,'a') as plottimer:
            process = psutil.Process(os.getpid())
            plottimer.write('chains '+str(timeit.default_timer())+' '+str(key)+' mem:'+str(process.get_memory_info())+'\n')
            plottimer.close()

        if key.burnin:
            for x in xrange(self.meta.miniters/self.meta.thinto):
                for w in randwalks:
                    for k in self.n_runs[key.n].binnos:
                        self.sps_chains[key.n][k].plot(np.arange(key.r*self.meta.miniters/self.meta.thinto,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#i_run.eachtimenos[r],
                                                     plot_y_c[k][w],
                                                     color = self.meta.colors[key.i],
                                                     alpha = self.a_chain,
                                                     rasterized = True)
        else:
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


            self.sps_samps[0][n].hlines(n_run.full_logflatNz,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        linestyle='--',
                                        label=r'Flat $\ln N(z)$')
            self.sps_samps[0][n].vlines(n_run.binends[1:-1],
                                        n_run.full_logflatNz[:-1],#np.concatenate((np.array([0]),n_run.full_logflatNz)),
                                        n_run.full_logflatNz[1:],#np.concatenate((n_run.full_logflatNz,np.array([0]))),
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        linestyle='--')
#             self.sps_samps[0][n].step(n_run.binends,
#                                       n_run.full_logflatNz,
#                                       color='k',
#                                       alpha=0.5,
#                                       label=r'Flat $\ln N(z)$',
#                                       linewidth=2,
#                                       where='post',
#                                       linestyle='--')
            self.sps_samps[1][n].hlines(n_run.full_flatNz,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        linestyle='--',
                                        label=r'Flat $N(z)$')
            self.sps_samps[1][n].vlines(n_run.binends[1:-1],
                                        n_run.full_flatNz[:-1],#np.concatenate((np.array([0]),n_run.full_flatNz)),
                                        n_run.full_flatNz[1:],#np.concatenate((n_run.full_flatNz,np.array([0]))),
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        linestyle='--')
#             self.sps_samps[1][n].step(n_run.binends,
#                                       n_run.full_flatNz,
#                                       color='k',
#                                       alpha=0.5,
#                                       label=r'Flat $N(z)$',
#                                       linewidth=2,
#                                       where='post',
#                                       linestyle='--')
            logdifstack = n_run.logstack - n_run.full_logsampNz
            difstack = n_run.stack - n_run.full_sampNz
            logvarstack = round(np.dot(logdifstack, logdifstack) / n_run.nbins**2)
            varstack = round(np.dot(difstack, difstack)/ n_run.nbins**2)

            self.sps_samps[0][n].hlines(n_run.logstack,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        label=r'Stacked $\ln N(z)$ $\sigma^{2}='+str(logvarstack)+r'$')
            self.sps_samps[0][n].vlines(n_run.binends[1:-1],
                                        n_run.logstack[:-1],#np.concatenate((np.array([0]),n_run.logstack)),
                                        n_run.logstack[1:],#np.concatenate((n_run.logstack,np.array([0]))),
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2)
#             self.sps_samps[0][n].step(n_run.binends,
#                                       n_run.logstack,
#                                       color='k',
#                                       alpha=0.5,
#                                       linewidth=2,
#                                       where='post',
#                                       label=r'Stacked $\ln N(z)$ $\sigma^{2}='+str(logvarstack)+r'$')
            self.sps_samps[1][n].hlines(n_run.stack,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2,
                                        label=r'Stacked $N(z)$ $\sigma^{2}='+str(varstack)+r'$')
            self.sps_samps[1][n].vlines(n_run.binends[1:-1],
                                        n_run.stack[:-1],#np.concatenate((np.array([0]),n_run.stack)),
                                        n_run.stack[1:],#np.concatenate((n_run.stack,np.array([0]))),
                                        color='k',
                                        alpha=0.5,
                                        linewidth=2)
#             self.sps_samps[1][n].step(n_run.binends,
#                                       n_run.stack,
#                                       color='k',
#                                       alpha=0.5,
#                                       linewidth=2,
#                                       where='post',
#                                       label=r'Stacked $N(z)$ $\sigma^{2}='+str(varstack)+r'$')
            self.sps_samps[0][n].hlines(n_run.full_logsampNz,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        linewidth=2,
                                        label=r'True $\ln N(z)$')
            self.sps_samps[0][n].vlines(n_run.binends[1:-1],
                                        n_run.full_logsampNz[:-1],#np.concatenate((np.array([0]),n_run.full_logsampNz)),
                                        n_run.full_logsampNz[1:],#np.concatenate((n_run.full_logsampNz,np.array([0]))),
                                        color='k',
                                        linewidth=2)
#             self.sps_samps[0][n].step(n_run.binends,
#                                       n_run.full_logsampNz,
#                                       color='k',
#                                       linewidth=2,
#                                       where='post',
#                                       label=r'True $\ln N(z)$')
            self.sps_samps[1][n].hlines(n_run.full_sampNz,
                                        n_run.binlos,
                                        n_run.binhis,
                                        color='k',
                                        linewidth=2,
                                        label=r'True $N(z)$')
            self.sps_samps[1][n].vlines(n_run.binends[1:-1],
                                        n_run.full_sampNz[:-1],#np.concatenate((np.array([0]),n_run.full_sampNz)),
                                        n_run.full_sampNz[1:],#np.concatenate((n_run.full_logsampNz,np.array([0]))),
                                        color='k',
                                        linewidth=2)
#             self.sps_samps[1][n].step(n_run.binends,
#                                       n_run.full_sampNz,
#                                       color='k',
#                                       linewidth=2,
#                                       where='post',
#                                       label=r'True $N(z)$')

            maxsteps = self.last_key.r
            maxiternos = np.arange(0,maxsteps)
            for i in lrange(meta.inits):
                i_run = i_runs[i]
                fitness = i_run.load_fitness('chains')
                tot_ls = round(fitness['tot_ls'][-1])#/maxsteps)
                print(fitness['tot_ls'])
                tot_s = round(fitness['tot_s'][-1])#/maxsteps)
                print(fitness['tot_s'])
                #var_ls = round(fitness['var_ls'])
                #var_s = round(fitness['var_s'])
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
            for k in n_run.binnos:
                self.sps_chains[n][k].step(maxiternos*meta.miniters,
                                           [n_run.full_logflatNz[k]]*maxsteps,
                                           color='k',
                                           label='Flat value',
                                           linestyle='--')
                self.sps_chains[n][k].plot(maxiternos*meta.miniters,
                                           [n_run.full_logsampNz[k]]*maxsteps,
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
