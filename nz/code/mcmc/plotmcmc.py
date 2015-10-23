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
import cPickle as cpkl
from utilmcmc import *
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
    randos = random.sample(xrange(meta.ngals),len(meta.colors))
    for r in lrange(randos):
        sps.step(meta.binmids,meta.pobs[randos[r]],where='mid',color=meta.colors[r%len(meta.colors)])
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
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
    sps.set_title(meta.init_names)
    sps.step(meta.binmids,meta.mean,color='k',where='mid')
    for ival in meta.ivals:
        sps.step(meta.binmids,ival,alpha=0.5,where='mid')
    f.savefig(os.path.join(meta.topdir,'initializations.png'),dpi=100)
    return

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        print('last key is '+str(self.last_key))
        self.plot(key)

# plot autocorrelation times and acceptance fractions
class plotter_timefrac(plotter):

    def __init__(self, meta):
        self.meta = meta

        self.f_times = plt.figure(figsize=(5,5))
        self.sps_times = self.f_times.add_subplot(1,1,1)
        self.a_times = float(len(self.meta.colors))/self.meta.nwalkers
        if self.meta.mode == 'bins':
            self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nbins) + ' bins')
        if self.meta.mode == 'walkers':
            self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nwalkers) + ' walkers')
        self.sps_times.set_ylabel('autocorrelation time')
        self.sps_times.set_xlabel('number of iterations')
        self.sps_times.set_ylim(0, 100)

        self.f_fracs = plt.figure(figsize=(5,5))
        self.a_fracs = float(len(self.meta.colors))/self.meta.nwalkers
        self.sps_fracs = self.f_fracs.add_subplot(1,1,1)
        self.sps_fracs.set_title('Acceptance Fractions for ' + str(self.meta.nwalkers) + ' walkers')
        self.sps_fracs.set_ylim(0,1)
        self.sps_fracs.set_ylabel('acceptance fraction')
        self.sps_fracs.set_xlabel('number of iterations')

    def plot(self, key):

        time_data = key.load_state(self.meta.topdir)['times']
        plot_y_times = time_data

        if self.meta.mode == 'bins':
            plot_x_times = [(key.r+1)*self.meta.miniters]*self.meta.nbins
        if self.meta.mode == 'walkers':
            plot_x_times = [(key.r+1)*self.meta.miniters]*self.meta.nwalkers
        self.sps_times.scatter(plot_x_times,
                               plot_y_times,
                               c='k',
                               alpha=self.a_times,
                               linewidth=0.1,
                               s=self.meta.nbins,
                               rasterized=True)
        timesaver(self.meta,'times-done',key)

        frac_data = key.load_state(self.meta.topdir)['fracs']
        plot_y_fracs = frac_data.T

        self.sps_fracs.scatter([(key.r+1)*self.meta.miniters]*self.meta.nwalkers,
                               plot_y_fracs,
                               c='k',
                               alpha=self.a_fracs,
                               linewidth=0.1,
                               s=self.meta.nbins,
                               rasterized=True)
        timesaver(self.meta,'fracs-done',key)


    def finish(self):

        self.sps_times.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f_times.savefig(os.path.join(self.meta.topdir,'times.png'),dpi=100)
        timesaver(self.meta,'times',self.meta.topdir)

        self.sps_fracs.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f_fracs.savefig(os.path.join(self.meta.topdir,'fracs.png'),dpi=100)
        timesaver(self.meta,'fracs',self.meta.topdir)

# plot log probabilities of samples of full posterior
class plotter_probs(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_probs = float(self.ncolors)/self.meta.nwalkers
        self.f = plt.figure(figsize=(5,5))
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title('Probability Evolution for ' + str(meta.nwalkers) + ' walkers')
        self.sps.set_ylabel('log probability of walker')
        self.sps.set_xlabel('iteration number')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1).T

        for w in xrange(self.meta.nwalkers):
            self.sps.plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,
                          plot_y[w],
                          c=self.meta.colors[w%self.ncolors],
                          alpha=self.a_probs,
                          rasterized=True)
        timesaver(self.meta,'probs',key)

    def finish(self):

        self.sps.set_xlim(-1*self.meta.miniters,(self.last_key.r+2)*self.meta.miniters)
        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)
        timesaver(self.meta,'probs-done',key)

# plot full posterior samples
class plotter_samps(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_samp = 1./self.meta.nbins
        self.f_samps = plt.figure(figsize=(5, 10))
        self.sps_samps = [self.f_samps.add_subplot(2,1,l+1) for l in xrange(0,2)]

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

    def plot(self,key):

        if key.burnin == False:

            data = key.load_state(self.meta.topdir)['chains']

            plot_y_ls = np.swapaxes(data,0,1)
            plot_y_s = np.exp(plot_y_ls)

            randsteps = random.sample(xrange(self.meta.ntimes),self.meta.nwalkers)
            randwalks = random.sample(xrange(self.meta.nwalkers),len(self.meta.colors))
            #self.a_samp = (key.r+1)/self.meta.nbins

            for w in randwalks:
                for x in randsteps:
                    self.sps_samps[0].hlines(plot_y_ls[x][w],
                                             self.meta.binlos,
                                             self.meta.binhis,
                                             color=self.meta.colors[w%self.ncolors],
                                             alpha=self.a_samp,
                                             rasterized=True)
                    self.sps_samps[1].hlines(plot_y_s[x][w],
                                             self.meta.binlos,
                                             self.meta.binhis,
                                             color=self.meta.colors[w%self.ncolors],
                                             alpha=self.a_samp,
                                             rasterized=True)
            timesaver(self.meta,'samps',key)

    def plotone(self,subplot,plot_y,style,ylabel):
        subplot.hlines(plot_y,
                       self.meta.binlos,
                       self.meta.binhis,
                       color='k',
                       linewidth=2,
                       linestyle=style,
                       label=ylabel)
        subplot.vlines(self.meta.binends[1:-1],
                       plot_y[:-1],
                       plot_y[1:],
                       color='k',
                       linewidth=2,
                       linestyle=style)
        return

    def finish(self):
        timesaver(self.meta,'samps-start',key)
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
            variances = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]
        #print(type(variances))
        #print('plot variances'+str(variances))
        if self.meta.logtrueNz is not None:
            #print(type(variances['vslogstack']))
            #print(variances['vslogstack'])
            logstacklabel = r' $\sigma^{2}=$'+str(int(variances['vslogstack']))
            stacklabel = r' $\sigma^{2}=$'+str(int(variances['vsstack']))
            logmaplabel = r' $\sigma^{2}=$'+str(int(variances['vslogmapNz']))
            maplabel = r' $\sigma^{2}=$'+str(int(variances['vsmapNz']))
            logexplabel = r' $\sigma^{2}=$'+str(int(variances['vslogexpNz']))
            explabel = r' $\sigma^{2}=$'+str(int(variances['vsexpNz']))
            logsampprep = min(variances['var_ls'])#/(self.last_key.r+1.)
            print('var_ls '+str(variances['var_ls']))
            logsamplabel = r' $\sigma^{2}=$'+str(int(logsampprep))#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
            print('plot var_ls='+str(logsampprep))
            sampprep = min(variances['var_s'])#/(self.last_key.r+1.)
            print('var_s '+str(variances['var_s']))
            samplabel = r' $\sigma^{2}=$'+str(int(sampprep))#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
            print('plot var_s='+str(sampprep))
        else:
            logstacklabel = ' '
            stacklabel = ' '
            logmaplabel = ' '
            maplabel = ' '
            logexplabel = ' '
            explabel = ' '
            logsamplabel = r' $\sigma^{2}=$'+str(int(variances['tot_ls']/(self.last_key.r+1.)))
            samplabel = r' $\sigma^{2}=$'+str(int(variances['tot_s']/(self.last_key.r+1.)))
            self.meta.logtrueNz = [-1.]*self.meta.nbins
            self.meta.trueNz = [-1.]*self.meta.nbins

        self.plotone(sps_samp_log,self.meta.logstack,'--',r'Stacked $\ln N(z)$'+logstacklabel)
        self.plotone(sps_samp,self.meta.stack,'--',r'Stacked $N(z)$'+stacklabel)
        self.plotone(sps_samp_log,self.meta.logmapNz,'-.',r'MAP $\ln N(z)$'+logmaplabel)
        self.plotone(sps_samp,self.meta.mapNz,'-.',r'MAP $N(z)$'+maplabel)
        self.plotone(sps_samp_log,self.meta.logexpNz,':',r'$\ln N(E(z))$'+logexplabel)
        self.plotone(sps_samp,self.meta.expNz,':',r'$N(E(z))$'+explabel)
        self.plotone(sps_samp_log,self.meta.logtrueNz,'-',r'True $\ln N(z)$'+logsamplabel)
        self.plotone(sps_samp,self.meta.trueNz,'-',r'True $N(z)$'+samplabel)

        sps_samp_log.legend(fontsize='xx-small', loc='upper left')
        sps_samp.legend(fontsize='xx-small', loc='upper left')

        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)
        timesaver(self.meta,'samps-done',key)

#plot full posterior chain evolution
class plotter_chains(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_chain = 1./ self.meta.nbins
        self.f_chains = plt.figure(figsize=(5*self.meta.nbins, 5))
        self.sps_chains = [self.f_chains.add_subplot(1,self.meta.nbins,k+1) for k in xrange(self.meta.nbins)]
        self.randwalks = random.sample(xrange(self.meta.nwalkers),self.ncolors)

        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.plot([0],[0],color = 'k',label = 'Mean Sample Value',rasterized = True)
            sps_chain.set_ylim(-m.log(self.meta.ngals), m.log(self.meta.ngals / self.meta.bindif)+1)
            sps_chain.set_xlabel('iteration number')
            sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
            sps_chain.set_title(r'$\ln N(z)$ Parameter {} of {}'.format(k, self.meta.nbins))

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_c = np.swapaxes(data,0,1).T

        randsteps = random.sample(xrange(self.meta.ntimes),self.meta.nwalkers)

        for k in xrange(self.meta.nbins):
            mean = np.sum(plot_y_c[k])/(self.meta.ntimes*self.meta.nwalkers)
            self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                    [mean]*self.meta.ntimes,
                                    color = 'k',
                                    rasterized = True)
            for x in xrange(self.meta.ntimes):
                for w in self.randwalks:
                    self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                            plot_y_c[k][w],
                                            color = self.meta.colors[w%self.ncolors],
                                            alpha = self.a_chain,
                                            rasterized = True)
        timesaver(self.meta,'chains',key)

    def plotone(self,subplot,plot_x,plot_y,style,ylabel):
        subplot.plot(plot_x,
                     plot_y,
                     color='k',
                     linewidth=2,
                     linestyle=style,
                     label=ylabel)
        return

    def finish(self):
        timesaver(self.meta,'chains-start',key)

        maxsteps = self.last_key.r+1
        maxiternos = np.arange(0,maxsteps)
        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.set_xlim(-1*self.meta.miniters,(maxsteps+1)*self.meta.miniters)
            sps_chain = self.sps_chains[k]
#             sps_chain.step(maxiternos*self.meta.miniters,
#                                        [self.meta.logflatNz[k]]*maxsteps,
#                                         color='k',
#                                         label='Flat value',
#                                     linewidth=2,
#                                         linestyle=':')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logstack[k]]*maxsteps,'--','Stacked value')
#             sps_chain.plot(maxiternos*self.meta.miniters,
#                                        [self.meta.logstack[k]]*maxsteps,
#                                        color='k',
#                                        #alpha=0.5,
#                                        linewidth=2,
#                                       linestyle='--',
#                                        label='Stacked value')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logmapNz[k]]*maxsteps,'-.','MAP value')
#             sps_chain.plot(maxiternos*self.meta.miniters,
#                                        [self.meta.logmapNz[k]]*maxsteps,
#                                        color='k',
#                                        #alpha=0.5,
#                                        linewidth=2,
#                                     linestyle='-.',
#                                        label='MAP value')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logexpNz[k]]*maxsteps,':',r'$E(z)$ value')
#             sps_chain.plot(maxiternos*self.meta.miniters,
#                                        [self.meta.logexpNz[k]]*maxsteps,
#                                        color='k',
#                                        #alpha=0.25,
#                                        linewidth=2,
#                                     linestyle=':',
#                                        label=r'$E(z)$ value')
            if self.meta.logtrueNz is not None:
                self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logtrueNz[k]]*maxsteps,'-','True value')
#                 sps_chain.plot(maxiternos*self.meta.miniters,
#                                        [self.meta.logtrueNz[k]]*maxsteps,
#                                        color='k',
#                                        #alpha=0.5,
#                                        linewidth=2,
#                                       linestyle='-',
#                                        label='True value')

            sps_chain.legend(fontsize='xx-small', loc='lower right')
            sps_chain.set_xlim(0,(self.last_key.r+1)*self.meta.miniters)

        self.f_chains.savefig(os.path.join(self.meta.topdir,'chains.png'),dpi=100)

        timesaver(self.meta,'chains-done',key)

# initialize all plotters
all_plotters = [plotter_chains
                ,plotter_samps
                ,plotter_probs
                ,plotter_timefrac
                ]
