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
import sklearn
from sklearn import neighbors
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
        self.miny = 0.

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1).T
        self.miny = np.min(np.array([np.min(plot_y),self.miny]))

        randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)
        for w in randwalks:
            self.sps.plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,
                          plot_y[w],
                          c=self.meta.colors[w%self.ncolors],
                          alpha=self.a_probs,
                          rasterized=True)

        timesaver(self.meta,'probs',key)

    def finish(self):

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as statprobs:
            probs = cpkl.load(statprobs)

        self.sps.plot(self.meta.miniters*np.arange(0,self.last_key.r+2),
                      [int(probs['lp_true'])]*(self.last_key.r+2),
                      label=r'True $p(\{\vec{d}_{j}\}_{J}|\tilde{\vec{\theta}})$',
                      color='k',linewidth=2,linestyle='-')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_stack'])]*(self.last_key.r+2),
                      label=r'Stacked $p(\{\vec{d}_{j}\}_{J}|\vec{\theta})$',
                      color='k',linewidth=2,linestyle='--')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_mapNz'])]*(self.last_key.r+2),
                      label=r'MAP $p(\{\vec{d}_{j}\}_{J}|\vec{\theta})$',
                      color='k',linewidth=2,linestyle='-.')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_expNz'])]*(self.last_key.r+2),
                      label=r'$E(z) p(\{\vec{d}_{j}\}_{J}|\vec{\theta})$',
                      color='k',linewidth=2,linestyle=':')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_true'])]*(self.last_key.r+2),
                      label=r'Sampled $p(\{\vec{d}_{j}\}_{J}|\vec{\theta})$',
                      color='k',linewidth=1,linestyle='-')

        self.sps.legend(fontsize='xx-small', loc='lower right')
        self.sps.set_xlim(0,(self.last_key.r+1)*self.meta.miniters)
        self.sps.set_ylim(self.miny,0.)
        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)

#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
#             both = cpkl.load(statboth)

#         f = plt.figure(figsize=(5,5))
#         sps = f.add_subplot(1,1,1)
#         sps.set_title('Samples vs. Point Estimate Competitors')
#         sps.hist(both['llr_stack'],bins=self.meta.nwalkers,alpha=1./3.,label=r'Stacked $A$')
#         sps.hist(both['llr_mapNz'],bins=self.meta.nwalkers,alpha=1./3.,label=r'MAP $N(z)$ $A$')
#         sps.hist(both['llr_expNz'],bins=self.meta.nwalkers,alpha=1./3.,label=r'N(E[z]) $A$')
#         sps.semilogy()
#         sps.legend(fontsize='xx-small', loc='upper left')
#         f.savefig(os.path.join(self.meta.topdir,'llr.png'),dpi=100)

        timesaver(self.meta,'probs-done',key)

# plot full posterior samples
class plotter_samps(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_samp = 1./self.meta.factor
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

        self.ll_samp = []

    def plot(self,key):

#         if key.burnin == False:

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)

        randsteps = random.sample(xrange(self.meta.ntimes),1)#self.ncolors)
        randwalks = random.sample(xrange(self.meta.nwalkers),1)#self.ncolors)
            #self.a_samp = (key.r+1)/self.meta.nbins

        for w in randwalks:
            for x in randsteps:
                self.sps_samps[0].step(self.meta.binlos,plot_y_ls[x][w],color=self.meta.colors[key.r%self.ncolors],where='post',alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
                self.sps_samps[1].step(self.meta.binlos,plot_y_s[x][w],color=self.meta.colors[key.r%self.ncolors],where='post',alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
#                     self.sps_samps[0].hlines(plot_y_ls[x][w],
#                                              self.meta.binlos,
#                                              self.meta.binhis,
#                                              color=self.meta.colors[key.r%self.ncolors],
#                                              alpha=self.a_samp,
#                                              rasterized=True)
#                     self.sps_samps[1].hlines(plot_y_s[x][w],
#                                              self.meta.binlos,
#                                              self.meta.binhis,
#                                              color=self.meta.colors[key.r%self.ncolors],
#                                              alpha=self.a_samp,
#                                              rasterized=True)
        timesaver(self.meta,'samps',key)

    def plotone(self,subplot,plot_y,style,lw,ylabel):
        subplot.hlines(plot_y,
                       self.meta.binlos,
                       self.meta.binhis,
                       color='k',
                       linewidth=lw,
                       linestyle=style,
                       label=ylabel)
        subplot.vlines(self.meta.binends[1:-1],
                       plot_y[:-1],
                       plot_y[1:],
                       color='k',
                       linewidth=lw,
                       linestyle=style)
        return

    def finish(self):
        timesaver(self.meta,'samps-start',key)
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
            gofs = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]

        #with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as statprobs:
        #    gof = cpkl.load(statprobs)
        #self.ll_samp = np.array(self.ll_samp)

        if self.meta.logtrueNz is not None:
            #llr_stackprep = str(int(gof['llr_stack']))
            logstackprep_v = str(int(gofs['vslogstack']))
            logstackprep_c = str(int(gofs['cslogstack']))
            #lr_stackprep = str(int(np.log10(np.exp(gof['llr_stack']))))
            stackprep_v = str(int(gofs['vsstack']))
            stackprep_c = str(int(gofs['csstack']))
            logstacklabel = r'; $\sigma^{2}=$'+logstackprep_v+r'; $\chi^{2}=$'+logstackprep_c#+r'; $\ln(r)=$'+llr_stackprep
            stacklabel = r'; $\sigma^{2}=$'+stackprep_v+r'; $\chi^{2}=$'+stackprep_c#+r'; $\log(r)=$'+lr_stackprep

            #llr_mapNzprep = str(int(gof['llr_mapNz']))
            logmapNzprep_v = str(int(gofs['vslogmapNz']))
            logmapNzprep_c = str(int(gofs['cslogmapNz']))
            #lr_mapNzprep = str(int(np.log10(np.exp(gof['llr_mapNz']))))
            mapNzprep_v = str(int(gofs['vsmapNz']))
            mapNzprep_c = str(int(gofs['csmapNz']))
            logmaplabel = r'; $\sigma^{2}=$'+logmapNzprep_v+r'; $\chi^{2}=$'+logmapNzprep_c#+r'; $\ln(r)=$'+llr_mapNzprep
            maplabel = r'; $\sigma^{2}=$'+mapNzprep_v+r'; $\chi^{2}=$'+mapNzprep_c#+r'; $\log(r)=$'+lr_mapNzprep

            #llr_expNzprep = str(int(gof['llr_expNz']))
            logexpNzprep_v = str(int(gofs['vslogexpNz']))
            logexpNzprep_c = str(int(gofs['cslogexpNz']))
            #lr_expNzprep = str(int(np.log10(np.exp(gof['llr_expNz']))))
            expNzprep_v = str(int(gofs['vsexpNz']))
            expNzprep_c = str(int(gofs['csexpNz']))
            logexplabel = r'; $\sigma^{2}=$'+logexpNzprep_v+r'; $\chi^{2}=$'+logexpNzprep_c#+r'; $\ln(r)=$'+llr_expNzprep
            explabel = r'; $\sigma^{2}=$'+expNzprep_v+r'; $\chi^{2}=$'+expNzprep_c#+r'; $\log(r)=$'+lr_expNzprep

            #llr_sampprep = str(int(np.average(self.ll_samp)))
            logsampprep_v = str(int(min(gofs['var_ls'])))#/(self.last_key.r+1.)
            logsampprep_c = str(int(min(gofs['chi_ls'])))#/(self.last_key.r+1.)
            #lr_sampprep = str(int(np.log10(np.exp(np.average(self.ll_samp)))))
            sampprep_v = str(int(min(gofs['var_s'])))#/(self.last_key.r+1.)
            sampprep_c = str(int(min(gofs['chi_s'])))#/(self.last_key.r+1.)
            logsamplabel = r'; $\sigma^{2}=$'+logsampprep_v+r'; $\chi^{2}=$'+logsampprep_c#+r'; $\ln(r)=$'+llr_sampprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
            samplabel = r'; $\sigma^{2}=$'+sampprep_v+r'; $\chi^{2}=$'+sampprep_c#+r'; $\log(r)=$'+lr_sampprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
        else:
            logstacklabel = ' '
            stacklabel = ' '
            logmaplabel = ' '
            maplabel = ' '
            logexplabel = ' '
            explabel = ' '
            logsamplabel = r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_ls']/(self.last_key.r+1.)))
            samplabel = r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_s']/(self.last_key.r+1.)))
            self.meta.logtrueNz = [-1.]*self.meta.nbins
            self.meta.trueNz = [-1.]*self.meta.nbins

        self.plotone(sps_samp_log,self.meta.logtrueNz,'-',2,r'True $\ln N(z)$')
        self.plotone(sps_samp,self.meta.trueNz,'-',2,r'True $N(z)$')
        self.plotone(sps_samp_log,self.meta.logstack,'--',2,r'Stacked $\ln N(z)$'+logstacklabel)
        self.plotone(sps_samp,self.meta.stack,'--',2,r'Stacked $N(z)$'+stacklabel)
        self.plotone(sps_samp_log,self.meta.logmapNz,'-.',2,r'MAP $\ln N(z)$'+logmaplabel)
        self.plotone(sps_samp,self.meta.mapNz,'-.',2,r'MAP $N(z)$'+maplabel)
        self.plotone(sps_samp_log,self.meta.logexpNz,':',2,r'$\ln N(E(z))$'+logexplabel)
        self.plotone(sps_samp,self.meta.expNz,':',2,r'$N(E(z))$'+explabel)
        self.plotone(sps_samp_log,self.meta.logtrueNz,'-',0.5,r'Sampled $\ln N(z)$'+logsamplabel)
        self.plotone(sps_samp,self.meta.trueNz,'-',0.5,r'Sampled $N(z)$'+samplabel)

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
        self.f_chains = plt.figure(figsize=(5,5*self.meta.nbins))
        self.sps_chains = [self.f_chains.add_subplot(self.meta.nbins,1,k+1) for k in xrange(self.meta.nbins)]
        self.randwalks = random.sample(xrange(self.meta.nwalkers),1)#self.ncolors)

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

        #randsteps = random.sample(xrange(self.meta.ntimes),self.meta.ncolors)

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

        maxsteps = self.last_key.r+2
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

# make all plots not needing MCMC
def final_plots(runs):
    for run in runs.keys():
        meta = runs[run]
        plot_llr(meta)
        timesaver(meta,'fplot',meta.key)
        #print('final plots completed')

def plot_llr(meta):
    with open(os.path.join(meta.topdir,'stat_both.p'),'rb') as statboth:
        both = cpkl.load(statboth)

    alldata = np.concatenate((both['llr_stack'],both['llr_mapNz'],both['llr_expNz']))
    min_llr = min(alldata)
    max_llr = max(alldata)
    datarange = np.linspace(min_llr,max_llr,meta.nwalkers)[:, np.newaxis]

    kde_stack = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=1).fit(both['llr_stack'][:, np.newaxis])
    print('constructed kde_stack')
    plot_stack = kde_stack.score_samples(datarange)
    print('scored kde_stack')
    kde_mapNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=1).fit(both['llr_mapNz'][:, np.newaxis])
    plot_mapNz = kde_mapNz.score_samples(datarange)
    kde_expNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=1).fit(both['llr_expNz'][:, np.newaxis])
    plot_expNz = kde_expNz.score_samples(datarange)

    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title('Samples vs. Point Estimate Competitors')
#     sps.hist(both['llr_stack'],bins=meta.nwalkers,alpha=1./3.,histtype='bar',log=True,
#              label=r'Stacked $A$, $\max(A)='+str(np.max(both['llr_stack']))+r'$')
#     sps.hist(both['llr_mapNz'],bins=meta.nwalkers,alpha=1./3.,histtype='bar',log=True,
#              label=r'MAP $N(z)$ $A$, $\max(A)='+str(np.max(both['llr_mapNz']))+r'$')
#     sps.hist(both['llr_expNz'],bins=meta.nwalkers,alpha=1./3.,histtype='bar',log=True,
#              label=r'N(E[z]) $A$, $\max(A)='+str(np.max(both['llr_expNz']))+r'$')
    sps.plot(datarange[:,0],np.exp(plot_stack),label=r'Stacked $A$, $\max(A)='+str(np.max(both['llr_stack']))+r'$')
    print('plotted kde_stack')
    sps.plot(datarange[:,0],np.exp(plot_mapNz),label=r'MAP $A$, $\max(A)='+str(np.max(both['llr_mapNz']))+r'$')
    sps.plot(datarange[:,0],np.exp(plot_expNz),label=r'$N(E[z])$ $A$, $\max(A)='+str(np.max(both['llr_expNz']))+r'$')

    sps.legend(fontsize='xx-small', loc='upper left')
    f.savefig(os.path.join(meta.topdir,'llr.png'),dpi=100)

    return
