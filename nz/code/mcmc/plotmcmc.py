"""
plot-mcmc module makes all plots including multiprocessed
"""

# TO DO: split up datagen and pre-run plots
import distribute
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import timeit
import random
import math as m
import statistics
import psutil
from util import *
from keymcmc import key

def timesaver(meta,name,key):
    with open(meta.plottime,'a') as plottimer:
        process = psutil.Process(os.getpid())
        plottimer.write(name+' '+str(timeit.default_timer())+' '+str(key)+' mem:'+str(process.get_memory_info())+'\n')
    return

# make all plots not needing MCMC
def initial_plots(runs):
    for run in runs.keys():
        meta = runs[run]
        plot_pdfs(meta)
        plot_priorsamps(meta)
        plot_ivals(meta)
        timesaver(meta,'iplot',meta.key)
        #print('initial plots completed')

# plot some individual posteriors
def plot_pdfs(meta):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Observed galaxy posteriors')
    #sps.set_title('shape='+str(meta.shape)+', noise='+str(meta.noise))
    randos = random.sample(xrange(meta.ngals),len(meta.colors))#n_run.ngals
    for r in lrange(randos):
        sps.step(meta.binmids,meta.pobs[randos[r]],where='mid',color=meta.colors[r])#,alpha=a)
        #sps.vlines(test.trueZs[randos[r]],0.,max(test.pobs[randos[r]]),color=meta.colors[r],linestyle='--')
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
    sps.set_ylim(0.,1./meta.bindif)
    f.savefig(os.path.join(meta.topdir,'samplepzs.png'))
    return

# plot some samples from prior for one instantiation of survey
def plot_priorsamps(meta):
    priorsamps = np.array(meta.priordist.sample_ps(len(meta.colors))[0])
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title(r'Prior samples for $J='+str(meta.ngals)+r'$')
    sps.set_xlabel(r'$z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlim(meta.binends[0]-meta.bindif,meta.binends[-1]+meta.bindif)#,s_run.seed)#max(n_run.full_logflatNz)+m.log(s_run.seed/meta.zdif)))
    sps.step(meta.binmids,meta.logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
    for c in lrange(meta.colors):
        sps.step(meta.binmids,priorsamps[c],color=meta.colors[c],where='mid')
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.topdir, 'priorsamps.png'))
    return

# plot initial values for all initialization procedures
def plot_ivals(meta):
    f = plt.figure(figsize=(5, 5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Initialization of '+str(meta.nwalkers)+' walkers')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlabel(r'$z$')
    sps.set_title(meta.init_names)

    for ival in meta.ivals:
        sps.step(meta.binlos,ival,
                alpha=0.5,)
    sps.step(meta.binmids,meta.mean,color='k',linewidth=2,where='mid')
    f.savefig(os.path.join(meta.topdir,'initializations.png'),dpi=100)
    return

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        print('last key is '+str(self.last_key))
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

    def __init__(self, meta):#p_run, s_run, n_runs, i_runs):
        self.meta = meta
        self.f = plt.figure(figsize=(5,5))
        self.sps = self.f.add_subplot(1,1,1)
        self.a_times = 2./self.meta.nbins
        self.sps.set_title('Autocorrelation Times for ' + str(self.meta.nbins) + ' dimensions')
        self.sps.set_ylabel('autocorrelation time')
        self.sps.set_xlabel('number of iterations')
        self.sps.set_ylim(0, 100)
#         for n in xrange(self.meta.samps):
#             for i in lrange(self.meta.inits):
#                 self.sps.scatter([0], [-1],
#                                  c=self.meta.colors[i+n],
#                                  label=self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]),
#                                  linewidths=0.1)
#         self.sps.legend(fontsize = 'small', loc = 'upper right')

    def plot(self, key):

        data = key.load_state(self.meta.topdir)['times']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.meta.nbins,#[i_run.iternos[r]]*meta.params[p_run.p],
                         plot_y,
                         c='k',#self.meta.colors[key.n+key.i],
                         alpha=self.a_times,
                         linewidth=0.1,
                         s=self.meta.nbins,
                         rasterized=True)
        timesaver(self.meta,'times',key)
        #print('last key '+str(self.last_key))


    def finish(self):
        timesaver(self.meta,'times-start',self.meta.topdir)
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.meta.topdir,'times.png'),dpi=100)
        timesaver(self.meta,'times-finish',self.meta.topdir)

# plot acceptance fractions
class plotter_fracs(plotter):

    def __init__(self, meta):#p_run, s_run, n_runs, **_):
        self.meta = meta
        self.a_fracs = 2./self.meta.nwalkers
        self.f = plt.figure(figsize=(5,5))
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title('Acceptance Fractions for ' + str(self.meta.nwalkers) + ' walkers')
        self.sps.set_ylim(0,1)
        self.sps.set_ylabel('acceptance fraction')
        self.sps.set_xlabel('number of iterations')
#         for n in xrange(self.meta.samps):
#             for i in lrange(self.meta.inits):
#                 self.sps.scatter([0],[-1],
#                         c=self.meta.colors[i+n],
#                         label=self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]),
#                         linewidths=0.1,
#                         s=self.p_run.ndims)
#         self.sps.legend(fontsize = 'small')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['fracs']
        plot_y = data.T

        self.sps.scatter([(key.r+1)*self.meta.miniters]*self.meta.nwalkers,#[i_run.iternos[r]] * n_run.nwalkers,
                         plot_y,
                         c='k',#self.meta.colors[key.i+key.n],
                         alpha=self.a_fracs,
                         linewidth=0.1,
                         s=self.meta.nbins,
                         rasterized=True)
        timesaver(self.meta,'fracs',key)

    def finish(self):
        timesaver(self.meta,'fracs-start',self.meta.topdir)
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.meta.topdir,'fracs.png'),dpi=100)
        timesaver(self.meta,'fracs-finish',self.meta.topdir)

# plot log probabilities of samples of full posterior
class plotter_probs(plotter):

    def __init__(self, meta):#p_run, s_run, n_runs, **_):
        self.meta = meta
        self.a_probs = 2./self.meta.nwalkers
        self.f = plt.figure(figsize=(5,5))
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title('Probability Evolution for ' + str(meta.nwalkers) + ' walkers')
        self.sps.set_ylabel('log probability of walker')
        self.sps.set_xlabel('iteration number')

        #self.sps.set_ylim(-self.s_run.seed, self.s_run.seed*m.log(self.s_run.seed))
        dummy_rec = [0]*self.meta.miniters
#         for n in xrange(self.meta.samps):
#             for i in lrange(self.meta.inits):
#                 self.sps.plot([-1.] * self.meta.miniters,
#                               dummy_rec,
#                               c=self.meta.colors[i+n],
#                               label = self.meta.init_names[i]+'\n shape='+str(self.meta.shape[n])+', noise='+str(self.meta.noise[n]))
#         self.sps.legend(fontsize='x-small', loc='lower right')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1).T

        for w in xrange(self.meta.nwalkers):
            self.sps.plot(np.arange(key.r*self.meta.miniters/self.meta.thinto,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#,(key.r+1)*self.meta.miniters),#key.i_run.eachtimenos[r],
                     plot_y[w],
                     c=self.meta.colors[key.r%len(self.meta.colors)],
                     alpha=self.a_probs,
                     rasterized=True)
        timesaver(self.meta,'probs',key)

    def finish(self):
        timesaver(self.meta,'probs-start',key)
        self.sps.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)
        timesaver(self.meta,'probs-finish',key)

# plot full posterior samples and chain evolution
class plotter_chains(plotter):

    def __init__(self, meta):#p_run, s_run, n_runs, i_runs):
        self.meta = meta
        self.a_samp = 1./self.meta.nwalkers
        self.a_chain = 1./ len(self.meta.colors)
        self.f_samps = plt.figure(figsize=(5, 10))
#         self.gs_samps = matplotlib.gridspec.GridSpec(2,1)
        self.sps_samps = [self.f_samps.add_subplot(2,1,l+1) for l in xrange(0,2)]
        self.f_chains = plt.figure(figsize=(5*self.meta.nbins, 5))
#         self.gs_chains = matplotlib.gridspec.GridSpec(1, self.meta.nbins)
        self.sps_chains = [self.f_chains.add_subplot(1,self.meta.nbins,k+1) for k in xrange(self.meta.nbins)]
        self.init_data = {'meta': meta}
        self.randwalks = random.sample(xrange(self.meta.nwalkers),len(self.meta.colors))
        dummy_chain = [-1.] * self.meta.miniters

        #for n in xrange(self.meta.samps):
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]
        sps_samp_log.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
        sps_samp_log.set_ylim(-1.,m.log(self.meta.ngals/self.meta.bindif)+1.)
        sps_samp_log.set_xlabel(r'$z$')
        sps_samp_log.set_ylabel(r'$\ln N(z)$')
        sps_samp_log.set_title(r'Samples of $\ln N(z)$')
        sps_samp.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
        sps_samp.set_ylim(0.,self.meta.ngals/self.meta.bindif+self.meta.ngals)
        sps_samp.set_xlabel(r'$z$')
        sps_samp.set_ylabel(r'$N(z)$')
        sps_samp.set_title(r'Samples of $N(z)$')

        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.set_ylim(-m.log(self.meta.ngals), m.log(self.meta.ngals / self.meta.bindif)+1)
            sps_chain.set_xlabel('iteration number')
            sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
            sps_chain.set_title(r'$\ln N(z)$ Parameter {} of {}'.format(k, self.meta.nbins))
#             for i in lrange(self.meta.inits):
#                 sps_chain.plot([-1.]*self.meta.miniters,
#                                   dummy_chain,
#                                   color=self.meta.colors[i],
#                                   label=self.meta.init_names[i])
#             sps_chain.legend(fontsize='small', loc='upper right')

    def plot(self,key):

        start_time = timeit.default_timer()
        data = key.load_state(self.meta.topdir)['chains']
        elapsed = timeit.default_timer() - start_time
        with open(self.meta.iotime,'a') as iotimer:
            process = psutil.Process(os.getpid())
            iotimer.write('plot '+str(key)+str(elapsed)+'\n')


        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)
        plot_y_c = plot_y_ls.T

        randsteps = random.sample(xrange(self.meta.ntimes),self.meta.nwalkers)

        for w in self.randwalks:
            for x in xrange(self.meta.ntimes):
                for k in xrange(self.meta.nbins):
                    self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#i_run.eachtimenos[r],
                                             plot_y_c[k][w],
                                             color = self.meta.colors[key.r%len(self.meta.colors)],
                                             alpha = self.a_chain,
                                             rasterized = True)

        if key.burnin == False:
#             for w in randwalks:
#                 for x in xrange(self.meta.ntimes):
#                     for k in xrange(self.meta.nbins):
#                         self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.miniters/self.meta.thinto)*self.meta.thinto,#i_run.eachtimenos[r],
#                                                      plot_y_c[k][w],
#                                                      #color = self.meta.colors[key.i],
#                                                      alpha = self.a_chain,
#                                                      rasterized = True)
            for w in self.randwalks:
                for x in randsteps:
                    self.sps_samps[0].hlines(plot_y_ls[x][w],
                                                  self.meta.binlos,
                                                  self.meta.binhis,
                                                  color=self.meta.colors[key.r%len(self.meta.colors)],
                                                  alpha=self.a_samp,
                                                  rasterized=True)
                    self.sps_samps[1].hlines(plot_y_s[x][w],
                                                  self.meta.binlos,
                                                  self.meta.binhis,
                                                  color=self.meta.colors[key.r%len(self.meta.colors)],
                                                  alpha=self.a_samp,
                                                  rasterized=True)
        timesaver(self.meta,'chains',key)

    def finish(self):
        timesaver(self.meta,'chains-start',key)

        self.sps_samps[0].hlines(self.meta.logflatNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                    linestyle=':',
                                    label=r'Flat $\ln N(z)$')
        self.sps_samps[0].vlines(self.meta.binends[1:-1],
                                    self.meta.logflatNz[:-1],#np.concatenate((np.array([0]),n_run.full_logflatNz)),
                                    self.meta.logflatNz[1:],#np.concatenate((n_run.full_logflatNz,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                    linestyle=':')
        self.sps_samps[1].hlines(self.meta.flatNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                    linestyle=':',
                                    label=r'Flat $N(z)$')
        self.sps_samps[1].vlines(self.meta.binends[1:-1],
                                    self.meta.flatNz[:-1],#np.concatenate((np.array([0]),n_run.full_flatNz)),
                                    self.meta.flatNz[1:],#np.concatenate((n_run.full_flatNz,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                    linestyle=':')
        self.sps_samps[0].hlines(self.meta.logstack,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                    linestyle='-',
                                    label=r'Stacked $\ln N(z)$ $\sigma^{2}=$')#+str(logvarstack)+r'$')
        self.sps_samps[0].vlines(self.meta.binends[1:-1],
                                    self.meta.logstack[:-1],#np.concatenate((np.array([0]),n_run.logstack)),
                                    self.meta.logstack[1:],#np.concatenate((n_run.logstack,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2)
        self.sps_samps[1].hlines(self.meta.stack,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                 linestyle='-',
                                    linewidth=2,
                                    label=r'Stacked $N(z)$ $\sigma^{2}=$')#+str(varstack)+r'$')
        self.sps_samps[1].vlines(self.meta.binends[1:-1],
                                    self.meta.stack[:-1],#np.concatenate((np.array([0]),n_run.stack)),
                                    self.meta.stack[1:],#np.concatenate((n_run.stack,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                 linestyle='-',
                                    linewidth=2)
        self.sps_samps[0].hlines(self.meta.logmapNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                 linestyle='--',
                                    label=r'MAP $\ln N(z)$ $\sigma^{2}=$')#+str(logvarstack)+r'$')
        self.sps_samps[0].vlines(self.meta.binends[1:-1],
                                    self.meta.logmapNz[:-1],#np.concatenate((np.array([0]),n_run.logstack)),
                                    self.meta.logmapNz[1:],#np.concatenate((n_run.logstack,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                 linestyle='--',
                                    linewidth=2)
        self.sps_samps[1].hlines(self.meta.mapNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.5,
                                    linewidth=2,
                                 linestyle='--',
                                    label=r'MAP $N(z)$ $\sigma^{2}=$')#+str(varstack)+r'$')
        self.sps_samps[1].vlines(self.meta.binends[1:-1],
                                    self.meta.mapNz[:-1],#np.concatenate((np.array([0]),n_run.stack)),
                                    self.meta.mapNz[1:],#np.concatenate((n_run.stack,np.array([0]))),
                                    color='k',
                                    #alpha=0.5,
                                 linestyle='--',
                                    linewidth=2)
        self.sps_samps[0].hlines(self.meta.logexpNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.25,
                                    linewidth=2,
                                 linestyle='-.',
                                    label=r'$E(z) \ln N(z)$ $\sigma^{2}=$')#+str(logvarstack)+r'$')
        self.sps_samps[0].vlines(self.meta.binends[1:-1],
                                    self.meta.logexpNz[:-1],#np.concatenate((np.array([0]),n_run.logstack)),
                                    self.meta.logexpNz[1:],#np.concatenate((n_run.logstack,np.array([0]))),
                                    color='k',
                                    #alpha=0.25,
                                 linestyle='-.',
                                    linewidth=2)
        self.sps_samps[1].hlines(self.meta.expNz,
                                    self.meta.binlos,
                                    self.meta.binhis,
                                    color='k',
                                    #alpha=0.25,
                                    linewidth=2,
                                 linestyle='-.',
                                    label=r'$E(z) N(z)$ $\sigma^{2}=$')#+str(varstack)+r'$')
        self.sps_samps[1].vlines(self.meta.binends[1:-1],
                                    self.meta.expNz[:-1],#np.concatenate((np.array([0]),n_run.stack)),
                                    self.meta.expNz[1:],#np.concatenate((n_run.stack,np.array([0]))),
                                    color='k',
                                    #alpha=0.25,
                                 linestyle='-.',
                                    linewidth=2)

        maxsteps = self.last_key.r+1
        maxiternos = np.arange(0,maxsteps)
    #        for i in lrange(meta.inits):
        fitness = self.meta.load_fitness('chains')
        tot_ls = round(fitness['tot_ls'][-1])#/maxsteps)
        print(fitness['tot_ls'])
        tot_s = round(fitness['tot_s'][-1])#/maxsteps)
        print(fitness['tot_s'])
                #var_ls = round(fitness['var_ls'])
                #var_s = round(fitness['var_s'])
#         self.sps_samps[0].hlines(dummy_lnsamp,
#                                         meta.binlos,
#                                         meta.binhis,
#                                         color=meta.colors[i],
#                                         label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_ls)+r'$')
#         self.sps_samps[1].hlines(dummy_samp,
#                                         self.meta.binlos,
#                                         self.meta.binhis,
#                                         color=meta.colors[i],
#                                         label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_s)+r'$')
        for k in xrange(self.meta.nbins):
            self.sps_chains[k].step(maxiternos*self.meta.miniters,
                                       [self.meta.logflatNz[k]]*maxsteps,
                                        color='k',
                                        label='Flat value',
                                    linewidth=2,
                                        linestyle=':')
#             self.sps_chains[k].plot(maxiternos*meta.miniters,
#                                        [self.meta.logsampNz[k]]*maxsteps,
#                                        color='k',
#                                        linewidth=2,
#                                        label='True value')
            self.sps_chains[k].plot(maxiternos*self.meta.miniters,
                                       [self.meta.logstack[k]]*maxsteps,
                                       color='k',
                                       #alpha=0.5,
                                       linewidth=2,
                                      linestyle='-',
                                       label='Stacked value')
            self.sps_chains[k].plot(maxiternos*self.meta.miniters,
                                       [self.meta.logmapNz[k]]*maxsteps,
                                       color='k',
                                       #alpha=0.5,
                                       linewidth=2,
                                    linestyle='--',
                                       label='MAP value')
            self.sps_chains[k].plot(maxiternos*self.meta.miniters,
                                       [self.meta.logexpNz[k]]*maxsteps,
                                       color='k',
                                       #alpha=0.25,
                                       linewidth=2,
                                    linestyle='-.',
                                       label=r'$E(z)$ value')
            self.sps_chains[k].legend(fontsize='xx-small', loc='lower right')
            self.sps_chains[k].set_xlim(0,(self.last_key.r+1)*self.meta.miniters)
        self.sps_samps[0].legend(fontsize='xx-small', loc='upper left')
        self.sps_samps[1].legend(fontsize='xx-small', loc='upper left')
        start_time = timeit.default_timer()
        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)
        elapsed = timeit.default_timer() - start_time
        with open(self.meta.iotime,'a') as iotimer:
            process = psutil.Process(os.getpid())
            iotimer.write('save samp '+str(key)+str(elapsed)+'\n')
        start_time = timeit.default_timer()
        self.f_chains.savefig(os.path.join(self.meta.topdir,'chains.png'),dpi=100)
        elapsed = timeit.default_timer() - start_time
        with open(self.meta.iotime,'a') as iotimer:
            process = psutil.Process(os.getpid())
            iotimer.write('save chain '+str(key)+str(elapsed)+'\n')
        timesaver(self.meta,'chains-finish',key)

# initialize all plotters
all_plotters = [plotter_times
               ,plotter_fracs
               ,plotter_probs
               ,plotter_chains
                ]
