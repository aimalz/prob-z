"""
plot-mcmc module makes all plots including multiprocessed info
"""

# TO DO: split up datagen and pre-run plots
import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import timeit
import random
import psutil
import cPickle as cpkl
import sklearn as skl
from sklearn import neighbors
import scipy as sp

import distribute
from utilmcmc import *
from keymcmc import key

# set up for better looking plots
title = 15
label = 15
mpl.rcParams['axes.titlesize'] = title
mpl.rcParams['axes.labelsize'] = label
mpl.rcParams['figure.subplot.left'] = 0.2
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.bottom'] = 0.2
mpl.rcParams['figure.subplot.top'] = 0.9
mpl.rcParams['figure.subplot.wspace'] = 0.5
mpl.rcParams['figure.subplot.hspace'] = 0.5

global lnz,nz,tv,t
lnz,nz,tv,t = r'$\ln[N(z)]$',r'$N(z)$',r'$\vec{\theta}$',r'$\theta$'
# setting up unified appearance parameters
global s_tru,w_tru,a_tru,c_tru,l_tru
s_tru,w_tru,a_tru,c_tru,l_tru = '-',2.,1.,'k','True '
global s_int,w_int,a_int,c_int,l_int
s_int,w_int,a_int,c_int,l_int = ':',1.5,0.5,'k','Interim '
global s_stk,w_stk,a_stk,c_stk,l_stk
s_stk,w_stk,a_stk,c_stk,l_stk = '-.',1.5,1.,'k','Stacked '
# global s_map,w_map,a_map,c_map,l_map
# s_map,w_map,a_map,c_map,l_map = '-.',1.5,0.75,'k','MMAP '
# global s_exp,w_exp,a_exp,c_exp,l_exp
# s_exp,w_exp,a_exp,c_exp,l_exp = '-.',1.5,0.25,'k','MExp '
global s_mml,w_mml,a_mml,c_mml,l_mml
s_mml,w_mml,a_mml,c_mml,l_mml = '-.',1.5,0.5,'k','MMLE '
global s_smp,w_smp,a_smp,c_smp,l_smp
s_smp,w_smp,a_smp,c_smp,l_smp = '-',1.,1.,'k','Sampled '
global s_bfe,w_bfe,a_bfe,c_bfe,l_bfe
s_bfe,w_bfe,a_bfe,c_bfe,l_bfe = '--',1.5,1.,'k','Best Fit '

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,s='-',c='k',a=1,w=1,l=None):
    ploth(subplot,binends,plot,s,c,a,w,l)
    plotv(subplot,binends,plot,s,c,a,w)

def ploth(subplot,binends,plot,s='-',c='k',a=1,w=1,l=None):
    subplot.hlines(plot,
                   binends[:-1],
                   binends[1:],
                   linewidth=w,
                   linestyle=s,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=True)
def plotv(subplot,binends,plot,s='-',c='k',a=1,w=1,l=None):
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=w,
                   linestyle=s,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=True)

def footer(subplot):
    subplot.annotate('Malz, et al. (2015)',(0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

def timesaver(meta,name,key):
    with open(meta.plottime,'a') as plottimer:
        process = psutil.Process(os.getpid())
        plottimer.write(name+' '+str(timeit.default_timer())+' '+str(key)+'\n')#' mem:'+str(process.get_memory_info())+'\n')
    return

# make all plots not needing MCMC
def initial_plots(runs):
    for run in runs.keys():
        meta = runs[run]
        plot_pdfs(meta)
        plot_priorsamps(meta)
        plot_ivals(meta)
        if meta.truNz is not None:
            plot_true(meta)
        timesaver(meta,'iplot',meta.key)
        #print('initial plots completed')

# plot some individual posteriors
def plot_pdfs(meta):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Observed galaxy posteriors for '+meta.name)
    #sps.set_title('shape='+str(meta.shape)+', noise='+str(meta.noise))
    randos = random.sample(xrange(meta.ngals),len(meta.colors))
    for r in lrange(randos):
        plotstep(sps,meta.binends,meta.pdfs[randos[r]],c=meta.colors[r%len(meta.colors)])
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
#     sps.set_ylim(0.,1./meta.bindif)
    f.savefig(os.path.join(meta.topdir,'samplepzs.png'))
    return

# plot some samples from prior for one instantiation of survey
def plot_priorsamps(meta):
    priorsamps = np.array(meta.priordist.sample_ps(len(meta.colors))[0])
    f = plt.figure(figsize=(5,5))
    f.suptitle(r'Prior samples for '+meta.name+':')
    sps = f.add_subplot(1,1,1)
    sps.set_title(r'$q='+str(meta.q)+r'$, $e='+str(meta.e)+r'$, $t='+str(meta.t)+r'$')
    sps.set_xlabel(r'$z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlim(meta.binends[0]-meta.bindif,meta.binends[-1]+meta.bindif)#,s_run.seed)#max(n_run.full_logfltNz)+m.log(s_run.seed/meta.zdif)))
    plotstep(sps,meta.binends,meta.logfltNz,l=r'flat $\ln N(z)$')
    for c in lrange(meta.colors):
        plotstep(sps,meta.binends,priorsamps[c],c=meta.colors[c])
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.topdir, 'priorsamps.png'))
    return

# plot initial values for all initialization procedures
def plot_ivals(meta):
    f = plt.figure(figsize=(5, 5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Initialization of '+str(meta.nwalkers)+' walkers for '+meta.name)
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
    sps.set_title(meta.init_names)
    plotstep(sps,meta.binends,meta.mean)
    for i in lrange(meta.ivals):
        plotstep(sps,meta.binends,meta.ivals[i],a=1./meta.factor,c=meta.colors[i%len(meta.colors)])
    f.savefig(os.path.join(meta.topdir,'initializations.png'),dpi=100)
    return

def plot_true(meta):
    f = plt.figure(figsize=(5,10))
    f.suptitle(str(meta.nbins)+r' Parameter '+meta.name+' for '+str(meta.ngals)+' galaxies')
    sps = f.add_subplot(2,1,1)
    sps.set_title('True $\ln N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(np.log(1./min(meta.bindifs)),np.log(meta.ngals/min(meta.bindifs)))#(-1.,np.log(test.ngals/min(test.meta.zdifs)))
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
#     plotstep(sps,test.binends,test.logtruNz,s=s_smp,w=w_smp,a=a_smp,c=c_smp,l=l_smp+lnz)
    plotstep(sps,meta.zrange,meta.lNz_range,s=s_tru,w=w_tru,a=a_tru,c=c_tru,l=l_tru+lnz)
    plotstep(sps,meta.binends,meta.logstkNz,s=s_stk,w=w_stk,a=a_tru,c=c_tru,l=l_stk+lnz)
#     plotstep(sps,meta.binends,meta.logmapNz,s=s_map,w=w_map,a=a_tru,c=c_tru,l=l_map+lnz)
#    plotstep(sps,meta.binends,meta.logexpNz,s=s_exp,w=w_exp,a=a_tru,c=c_tru,l=l_exp+lnz)
    plotstep(sps,meta.binends,meta.logmmlNz,s=s_mml,w=w_mml,a=a_tru,c=c_tru,l=l_mml+lnz)
    plotstep(sps,meta.binends,meta.logintNz,s=s_int,w=w_int,a=a_tru,c=c_tru,l=l_int+lnz)
    sps.legend(loc='lower right',fontsize='x-small')
    sps = f.add_subplot(2,1,2)
    sps.set_title('True $N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$N(z)$')
#     sps.set_ylim(0.,test.ngals/min(test.bindifs))#(0.,test.ngals/min(test.meta.zdifs))
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
#     plotstep(sps,test.binends,test.truNz,s=s_smp,w=w_smp,a=a_smp,c=c_smp,l=l_smp+nz)
    plotstep(sps,meta.zrange,meta.Nz_range,s=s_tru,w=w_tru,a=a_tru,c=c_tru,l=l_tru+nz)
    plotstep(sps,meta.binends,meta.stkNz,s=s_stk,w=w_stk,a=a_tru,c=c_tru,l=l_stk+nz)# with $\sigma^{2}=$'+str(int(test.vsstack)))
#     plotstep(sps,meta.binends,meta.mapNz,s=s_map,w=w_map,a=a_tru,c=c_tru,l=l_map+nz)# with $\sigma^{2}=$'+str(int(test.vsmapNz)))
#     plotstep(sps,meta.binends,meta.expNz,s=s_exp,w=w_exp,a=a_tru,c=c_tru,l=l_exp+nz)# with $\sigma^{2}=$'+str(int(test.vsexpNz)))
    plotstep(sps,meta.binends,meta.mmlNz,s=s_mml,w=w_mml,a=a_tru,c=c_tru,l=l_mml+nz)
    plotstep(sps,meta.binends,meta.intNz,s=s_int,w=w_int,a=a_tru,c=c_tru,l=l_int+nz)# with $\sigma^{2}=$'+str(int(test.vsinterim)))
    sps.legend(loc='upper left',fontsize='x-small')

    footer(sps)
    f.savefig(os.path.join(meta.topdir,'trueNz.png'))
    return

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        print(self.meta.name+' last key is '+str(self.last_key))
        self.plot(key)

# plot autocorrelation times and acceptance fractions
class plotter_times(plotter):

    def __init__(self, meta):
        self.meta = meta

        self.f_times = plt.figure(figsize=(5,5))
        self.f_times.suptitle(self.meta.name)
        self.sps_times = self.f_times.add_subplot(1,1,1)
        self.a_times = float(len(self.meta.colors))/self.meta.nwalkers
        if self.meta.mode == 'bins':
            self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nbins) + ' bins')
        if self.meta.mode == 'walkers':
            self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nwalkers) + ' walkers')
        self.sps_times.set_ylabel('autocorrelation time')
        self.sps_times.set_xlabel('number of iterations')
        self.sps_times.set_ylim(0, 100)

#         self.f_fracs = plt.figure(figsize=(5,5))
#         self.f_fracs.suptitle(self.meta.name)
#         self.a_fracs = float(len(self.meta.colors))/self.meta.nwalkers
#         self.sps_fracs = self.f_fracs.add_subplot(1,1,1)
#         self.sps_fracs.set_title('Acceptance Fractions for ' + str(self.meta.nwalkers) + ' walkers')
#         self.sps_fracs.set_ylim(0,1)
#         self.sps_fracs.set_ylabel('acceptance fraction')
#         self.sps_fracs.set_xlabel('number of iterations')

        plot_pdfs(self.meta)
        plot_priorsamps(self.meta)
        plot_ivals(self.meta)
        if self.meta.truNz is not None:
            plot_true(self.meta)
        timesaver(self.meta,'iplot',self.meta.key)

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

        self.f_times.savefig(os.path.join(self.meta.topdir,'times.png'),dpi=100)

        timesaver(self.meta,'times-done',key)

#         frac_data = key.load_state(self.meta.topdir)['fracs']
#         plot_y_fracs = frac_data.T

#         self.sps_fracs.scatter([(key.r+1)*self.meta.miniters]*self.meta.nwalkers,
#                                plot_y_fracs,
#                                c='k',
#                                alpha=self.a_fracs,
#                                linewidth=0.1,
#                                s=self.meta.nbins,
#                                rasterized=True)

#         self.f_fracs.savefig(os.path.join(self.meta.topdir,'fracs.png'),dpi=100)

#         timesaver(self.meta,'fracs-done',key)


    def finish(self):

        self.sps_times.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
        self.f_times.savefig(os.path.join(self.meta.topdir,'times.png'),dpi=100)
        timesaver(self.meta,'times',self.meta.topdir)

#         self.sps_fracs.set_xlim(0,(self.last_key.r+2)*self.meta.miniters)
#         self.f_fracs.savefig(os.path.join(self.meta.topdir,'fracs.png'),dpi=100)
#         timesaver(self.meta,'fracs',self.meta.topdir)

# plot log probabilities of samples of full posterior
class plotter_probs(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_probs = 1.#/self.meta.factor#self.meta.nwalkers
        self.f = plt.figure(figsize=(5,5))
        self.f.suptitle(self.meta.name)
        sps = self.f.add_subplot(1,1,1)
        self.sps = sps
        self.sps.set_title('Posterior Probability Evolution for ' + str(meta.nwalkers) + ' walkers')
        self.sps.set_ylabel('log probability of walker')
        self.sps.set_xlabel('iteration number')
        self.medy = []

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['probs']
        plot_y = np.swapaxes(data,0,1)#ntimes*nwalkers#.T#nwalkers*ntimes
#         if key.r != 0:
#         self.medy.append(np.median([np.median(plot_y[w]) for w in xrange(self.meta.nwalkers)]))

        locs,scales = [],[]
        for x in xrange(self.meta.ntimes):
            loc,scale = sp.stats.norm.fit_loc_scale(plot_y[x])
            locs.append(loc)
            scales.append(scale)
        locs = np.array(locs)
        scales = np.array(scales)
        x_all = np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto
        self.sps.plot(x_all,locs,color='k',rasterized=True)
#         self.sps.vlines(xrange(self.meta.ntimes+1),(locs-scales),(locs+scales),color='k',alpha=0.5,linewidth=2.,rasterized=True)
        self.sps.fill_between(x_all,locs-scales,locs+scales,color='k',alpha=0.1,linewidth=0.)
#         randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)
#         for w in randwalks:
#             self.sps.plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,
#                           plot_y[w],
#                           c=self.meta.colors[w%self.ncolors],
#                           alpha=self.a_probs,
#                           rasterized=True)

        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)

        timesaver(self.meta,'probs',key)

    def finish(self):

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as statprobs:
            probs = cpkl.load(statprobs)

#         yrange = self.medy#np.array(self.medy+[probs['lp_truNz'],probs['lp_stkNz'],probs['lp_mapNz'],probs['lp_expNz']])
#         miny = np.min(yrange)-np.log(self.meta.ngals)
#         maxy = np.max(yrange)+np.log(self.meta.ngals)

        if self.meta.logtruNz is not None:
#             self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_truNz'])]*(self.last_key.r+2),
#                       label=r'True $\vec{\theta}$',
#                       color='k',linewidth=2,linestyle='-')
            self.plotone(probs['lp_truNz'],w=w_tru,s=s_tru,a=a_tru,c=c_tru,l=l_tru+tv)
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_stkNz'])]*(self.last_key.r+2),
#                      label=r'Stacked $\vec{\theta}$',
#                       color='k',linewidth=2,linestyle='--')
            self.plotone(probs['lp_stkNz'],w=w_stk,s=s_stk,a=a_stk,c=c_stk,l=l_stk+tv)
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_mapNz'])]*(self.last_key.r+2),
#                       label=r'MAP $\vec{\theta}$',
#                       color='k',linewidth=2,linestyle='-.')
#             self.plotone(probs['lp_mapNz'],w=w_map,s=s_map,a=a_map,c=c_map,l=l_map+tv)
# #         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
# #                       [int(probs['lp_expNz'])]*(self.last_key.r+2),
# #                       label=r'$E[z]$ $\vec{\theta}$',
# #                       color='k',linewidth=2,linestyle=':')
# #           self.plotone(probs['lp_expNz'],w=w_exp,s=s_exp,a=a_exp,c=c_exp,l=l_exp+tv)
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_mmlNz'])]*(self.last_key.r+2),
#                       label=r'MMLE $\vec{\theta}$',
#                       color='k',linewidth=2,linestyle=':')
            self.plotone(probs['lp_mmlNz'],w=w_mml,s=s_mml,a=a_mml,c=c_mml,l=l_mml+tv)#r'MMLE $\vec{\theta}$',w=2.,s=':')
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_truNz'])]*(self.last_key.r+2),
#                       label=r'Sampled $\vec{\theta}$',
#                       color='k',linewidth=1,linestyle='-')
#           self.plotone(probs['lp_intNz'],w=w_int,s=s_int,a=a_int,c=c_int,l=l_int+tv

        self.sps.legend(fontsize='xx-small', loc='upper right')
        self.sps.set_xlim(-1*self.meta.miniters,(self.last_key.r+2)*self.meta.miniters)
#         self.sps.set_ylim(miny,maxy)
        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)

#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
#             both = cpkl.load(statboth)

#         f = plt.figure(figsize=(5,5))
#         sps = f.add_subplot(1,1,1)
#         sps.set_title('Samples vs. Point Estimate Competitors')
#         sps.hist(both['llr_stkNz'],bins=self.meta.nwalkers,alpha=1./3.,label=r'Stacked $A$')
#         sps.hist(both['llr_mapNz'],bins=self.meta.nwalkers,alpha=1./3.,label=r'MAP $N(z)$ $A$')
#         sps.hist(both['llr_expNz'],bins=self.meta.nwalkers,alpha=1./3.,label=r'N(E[z]) $A$')
#         sps.semilogy()
#         sps.legend(fontsize='xx-small', loc='upper left')
#         f.savefig(os.path.join(self.meta.topdir,'llr.png'),dpi=100)

#         self.plot_llr()

        timesaver(self.meta,'probs-done',key)

#     def plot_llr(self):
#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
#             both = cpkl.load(statboth)
#         with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
#             gofs = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]

#         f = plt.figure(figsize=(5,5))
#         f.suptitle(self.meta.name)

#         sps = f.add_subplot(1,1,1)#[f.add_subplot(1,2,l+1) for l in xrange(0,2)]

# #         alldata = np.concatenate((both['llr_stkNz'],both['llr_mapNz'],both['llr_mmlNz']))
# #         min_llr = np.min(alldata)
# #         max_llr = np.max(alldata)
# #         datarange = np.linspace(min_llr,max_llr,10*self.meta.nwalkers)[:, np.newaxis]
# #         kde_stkNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_stkNz'][:, np.newaxis])
# #         plot_stkNz = kde_stkNz.score_samples(datarange)
# #         kde_mapNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_mapNz'][:, np.newaxis])
# #         plot_mapNz = kde_mapNz.score_samples(datarange)
# # #         kde_expNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_expNz'][:, np.newaxis])
# # #         plot_expNz = kde_expNz.score_samples(datarange)
# #         kde_mmlNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_mmlNz'][:, np.newaxis])
# #         plot_mmlNz = kde_mmlNz.score_samples(datarange)

# #         sps[0].set_title('Log Likelihood Ratio')
# #         sps[0].semilogy()
# #         sps[0].plot(datarange[:,0],np.exp(plot_stkNz),color='k',linestyle='--',label=r'Stacked $R$, $\max(R)='+str(round(np.max(both['llr_stkNz']),3))+r'$')
# #         sps[0].plot(datarange[:,0],np.exp(plot_mapNz),color='k',linestyle='-.',label=r'MAP $R$, $\max(R)='+str(round(np.max(both['llr_mapNz']),3))+r'$')
# # #         sps[0].plot(datarange[:,0],np.exp(plot_expNz),color='k',linestyle=':',label=r'$N(E[z])$ $R$, $\max(R)='+str(round(np.max(both['llr_expNz']),3))+r'$')
# #         sps[0].plot(datarange[:,0],np.exp(plot_mmlNz),color='k',linestyle=':',label=r'MMLE $R$, $\max(R)='+str(round(np.max(both['llr_mmlNz']),3))+r'$')
# #         sps[0].legend(fontsize='xx-small', loc='upper left')
# #         sps[0].set_ylim(1e-10,1)
# #         sps[0].set_xlabel('log likelihood ratio')
# #         sps[0].set_ylabel('kernel density estimate')

#         alldata = np.concatenate((gofs['kl_smpNzvtruNz'],gofs['kl_truNzvsmpNz']))
#         min_kl = np.min(alldata)
#         max_kl = np.max(alldata)
#         datarange = np.linspace(min_kl,max_kl,10*self.meta.nwalkers)[:, np.newaxis]
#         kde_smpNzvtruNz = skl.neighbors.KernelDensity(kernel='gaussian',bandwidth=1.).fit(gofs['kl_smpNzvtruNz'][:, np.newaxis])
#         plot_smpNzvtruNz = kde_smpNzvtruNz.score_samples(datarange)
#         kde_truNzvsmpNz = skl.neighbors.KernelDensity(kernel='gaussian',bandwidth=1.).fit(gofs['kl_truNzvsmpNz'][:, np.newaxis])
#         plot_truNzvsmpNz = kde_truNzvsmpNz.score_samples(datarange)

#         sps.set_title('Kullback-Leibler Divergence')
#         yrange = np.concatenate((np.exp(plot_smpNzvtruNz),np.exp(plot_truNzvsmpNz)))

#         sps.vlines(gofs['kl_stkNzvtruNz'],0.,1.,color='k',linestyle='--',label=r'Stacked $KLD=('+str(gofs['kl_stkNzvtruNz'])+','+str(gofs['kl_truNzvstkNz'])+r')$')
#         sps.vlines(gofs['kl_truNzvstkNz'],0.,1.,color='k',linestyle='--')
#         sps.vlines(gofs['kl_mapNzvtruNz'],0.,1.,color='k',linestyle='-.',label=r'MAP $KLD=('+str(gofs['kl_mapNzvtruNz'])+','+str(gofs['kl_truNzvmapNz'])+r')$')
#         sps.vlines(gofs['kl_truNzvmapNz'],0.,1.,color='k',linestyle='-.')
# #         sps[1].vlines(gofs['kl_expNzvtruNz'],0.,1.,color='k',linestyle=':',label=r'$E[z]$ $KLD=('+str(gofs['kl_expNzvtruNz'])+','+str(gofs['kl_truNzvexpNz'])+r')$')
# #         sps[1].vlines(gofs['kl_truNzvexpNz'],0.,1.,color='k',linestyle=':')
#         sps.vlines(gofs['kl_mmlNzvtruNz'],0.,1.,color='k',linestyle=':',label=r'MMLE $KLD=('+str(gofs['kl_mmlNzvtruNz'])+','+str(gofs['kl_truNzvmmlNz'])+r')$')
#         sps.vlines(gofs['kl_truNzvmmlNz'],0.,1.,color='k',linestyle=':')
#         sps.vlines(gofs['kl_intNzvtruNz'],0.,1.,alpha=0.5,color='k',linestyle='-',label=r'Interim $KLD=('+str(gofs['kl_intNzvtruNz'])+','+str(gofs['kl_truNzvintNz'])+r')$')
#         sps.vlines(gofs['kl_truNzvintNz'],0.,1.,alpha=0.5,color='k',linestyle='-')
#         sps.plot(datarange[:,0],np.exp(plot_smpNzvtruNz),color='k',linewidth=2,label=r'Sampled $\min KLD=('+str(np.min(gofs['kl_smpNzvtruNz'].flatten()))+','+str(np.min(gofs['kl_truNzvsmpNz'].flatten()))+r')$')
#         sps.plot(datarange[:,0],np.exp(plot_truNzvsmpNz),color='k',linewidth=2)
#         sps.legend(fontsize='xx-small', loc='upper left')
#         sps.set_xlabel('Kullback-Leibler divergence')
#         sps.set_ylabel('kernel density estimate')
#         sps.semilogx()

#         f.savefig(os.path.join(self.meta.topdir,'kld.png'),dpi=100)

    def plotone(self,plot_y,l=None,a=1.,c='k',w=1.,s='-'):
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                      [int(plot_y)]*(self.last_key.r+2),
#                      color=c,
#                      linewidth=w,
#                      linestyle=s,
#                      alpha=a,
#                      label=l)
        ploth(self.sps,self.meta.miniters*np.arange(0.,self.last_key.r+2),plot_y,s,c,a,w,l)

# plot full posterior samples
class plotter_samps(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_samp = 1.#/self.meta.ntimes#self.ncolors/self.meta.nwalkers
        self.f_samps = plt.figure(figsize=(5, 10))
        self.f_samps.suptitle(self.meta.name)
        self.sps_samps = [self.f_samps.add_subplot(2,1,l+1) for l in xrange(0,2)]

        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]
        sps_samp_log.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
#         sps_samp_log.set_ylim(-1.,m.log(self.meta.ngals/self.meta.bindif)+1.)
        sps_samp_log.set_xlabel(r'$z$')
        sps_samp_log.set_ylabel(r'$\ln N(z)$')
        sps_samp_log.set_title(r'Samples of $\ln N(z)$')
        sps_samp.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
#         sps_samp.set_ylim(0.,self.meta.ngals/self.meta.bindif+self.meta.ngals)
        sps_samp.set_xlabel(r'$z$')
        sps_samp.set_ylabel(r'$N(z)$')
        sps_samp.set_title(r'Samples of $N(z)$')

        self.ll_smpNz = []

#         plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=2,l=r'True $\ln N(z)$')
#         plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=2,l=r'True $N(z)$')
#         self.plotone(self.meta.logtruNz,self.meta.truNz,w=w_tru,s=s_tru,a=a_tru,c=c_tru,l=l_tru)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logstkNz,s='--',w=2)
#         plotstep(sps_samp,self.meta.binends,self.meta.stkNz,s='--',w=2)
        self.plotone(self.meta.logstkNz,self.meta.stkNz,w=w_stk,s=s_stk,a=a_stk,c=c_stk,l=l_stk)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logmapNz,s='-.',w=2)
#         plotstep(sps_samp,self.meta.binends,self.meta.mapNz,s='-.',w=2)
#         self.plotone(self.meta.logmapNz,self.meta.mapNz,w=w_map,s=s_map,a=a_map,c=c_map,l=l_map)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logexpNz,s=':',w=2,l=r'$\ln N(E(z))$'+logexplabel)
#         plotstep(sps_samp,self.meta.binends,self.meta.expNz,s=':',w=2,l=r'$N(E(z))$'+explabel)
#         self.plotone(self.meta.logexpNz,self.meta.expNz,w=w_exp,s=s_exp,a=a_exp,c=c_exp,l=l_exp)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logmmlNz,s=':',w=2)
#         plotstep(sps_samp,self.meta.binends,self.meta.mmlNz,s=':',w=2)
        self.plotone(self.meta.logmmlNz,self.meta.mmlNz,w=w_mml,s=s_mml,a=a_mml,c=c_mml,l=l_mml)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logintNz,s='-',a=0.5,w=1)
#         plotstep(sps_samp,self.meta.binends,self.meta.intNz,s='-',a=0.5,w=1)
        self.plotone(self.meta.logintNz,self.meta.intNz,w=w_int,s=s_int,a=a_int,c=c_int,l=l_int)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=0.5)
#         plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=0.5)
        if self.meta.logtruNz is not None:
            plotstep(sps_samp_log,self.meta.zrange,self.meta.lNz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,l=l_tru+lnz)
            plotstep(sps_samp,self.meta.zrange,self.meta.Nz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,l=l_tru+nz)
#             plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=2)
#             plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=2)

    def plot(self,key):

      if key.burnin == False:

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)

        randsteps = random.sample(xrange(self.meta.ntimes),1)#xrange(self.meta.ntimes)#self.ncolors)
        randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)#self.ncolors)
            #self.a_samp = (key.r+1)/self.meta.nbins

        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        for w in randwalks:
            for x in randsteps:
                plotstep(sps_samp_log,self.meta.binends,plot_y_ls[x][w],s=s_smp,w=w_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])
                plotstep(sps_samp,self.meta.binends,plot_y_s[x][w],s=s_smp,w=w_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])
#                 self.plotone(plot_y_ls[x][w],plot_y_s[x][w],w=w_smp,s=s_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])
#                 self.sps_samps[0].step(self.meta.binlos,plot_y_ls[x][w],color=self.meta.colors[key.r%self.ncolors],alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
#                 self.sps_samps[1].step(self.meta.binlos,plot_y_s[x][w],color=self.meta.colors[key.r%self.ncolors],alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
#                 self.sps_samps[0].hlines(plot_y_ls[x][w],
#                                              self.meta.binends[:-1],
#                                              self.meta.binends[1:],
#                                              color=self.meta.colors[key.r%self.ncolors],
#                                              alpha=self.a_samp,
#                                              rasterized=True)
#                 self.sps_samps[1].hlines(plot_y_s[x][w],
#                                              self.meta.binends[:-1],
#                                              self.meta.binends[1:],
#                                              color=self.meta.colors[key.r%self.ncolors],
#                                              alpha=self.a_samp,
#                                              rasterized=True)

        for k in xrange(self.meta.nbins):
            y_all = plot_y_ls.T[k].flatten()
            loc,scale = sp.stats.norm.fit_loc_scale(y_all)
#             sps_samp_log.fill_between([self.meta.binends[k],self.meta.binends[k+1]],loc-scale,loc+scale,color=self.meta.colors[key.r%self.ncolors],alpha=0.1)
#             sps_samp.fill_between([self.meta.binends[k],self.meta.binends[k+1]],np.exp(loc-scale),np.exp(loc+scale),color=self.meta.colors[key.r%self.ncolors],alpha=0.1)

        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)

        timesaver(self.meta,'samps',key)

    def finish(self):
        timesaver(self.meta,'samps-start',key)
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
            gofs = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
            both = cpkl.load(statboth)

#         for v in lrange(both['mapvals']):
#             plotstep(sps_samp_log,self.meta.binends,both['mapvals'][v],w=2,c=self.meta.colors[v%self.ncolors])
#             plotstep(sps_samp,self.meta.binends,np.exp(both['mapvals'][v]),w=2,c=self.meta.colors[v%self.ncolors])

#         if self.meta.logtruNz is not None:
# # #             #llr_stkNzprep = str(int(gof['llr_stkNz']))
# # #             logstkNzprep_v = str(int(gofs['vslogstkNz']))
# # #             logstkNzprep_c = str(int(gofs['cslogstkNz']))
# # #             #lr_stkNzprep = str(int(np.log10(np.exp(gof['llr_stkNz']))))
# # #             stkNzprep_v = str(int(gofs['vsstkNz']))
# # #             stkNzprep_c = str(int(gofs['csstkNz']))
# #             logstklabel = r'; $\ln \mathcal{L}='+str(round(both['ll_stkNz']))+r'$'#r'; $\sigma^{2}=$'+logstkNzprep_v+r'; $\chi^{2}=$'+logstkNzprep_c#+r'; $\ln(r)=$'+llr_stkNzprep
# #             stklabel = r'; $KLD=('+str(gofs['kl_stkNzvtruNz'])+','+str(gofs['kl_truNzvstkNz'])+r')$'#r'; $\sigma^{2}=$'+stkNzprep_v+r'; $\chi^{2}=$'+stkNzprep_c#+r'; $\log(r)=$'+lr_stkNzprep

# # #             #llr_mapNzprep = str(int(gof['llr_mapNz']))
# # #             logmapNzprep_v = str(int(gofs['vslogmapNz']))
# # #             logmapNzprep_c = str(int(gofs['cslogmapNz']))
# # #             #lr_mapNzprep = str(int(np.log10(np.exp(gof['llr_mapNz']))))
# # #             mapNzprep_v = str(int(gofs['vsmapNz']))
# # #             mapNzprep_c = str(int(gofs['csmapNz']))
# #             logmaplabel = r'; $\ln \mathcal{L}='+str(round(both['ll_mapNz']))+r'$'#r'; $\sigma^{2}=$'+logmapNzprep_v+r'; $\chi^{2}=$'+logmapNzprep_c#+r'; $\ln(r)=$'+llr_mapNzprep
# #             maplabel = r'; $KLD=('+str(gofs['kl_mapNzvtruNz'])+','+str(gofs['kl_truNzvmapNz'])+r')$'#r'; $\sigma^{2}=$'+mapNzprep_v+r'; $\chi^{2}=$'+mapNzprep_c#+r'; $\log(r)=$'+lr_mapNzprep

# # # #             #llr_expNzprep = str(int(gof['llr_expNz']))
# # # #             logexpNzprep_v = str(int(gofs['vslogexpNz']))
# # # #             logexpNzprep_c = str(int(gofs['cslogexpNz']))
# # # #             #lr_expNzprep = str(int(np.log10(np.exp(gof['llr_expNz']))))
# # # #             expNzprep_v = str(int(gofs['vsexpNz']))
# # # #             expNzprep_c = str(int(gofs['csexpNz']))
# # #             logexplabel = r'; $\ln \mathcal{L}='+str(both['ll_expNz'])+r'$'#r'; $\sigma^{2}=$'+logexpNzprep_v+r'; $\chi^{2}=$'+logexpNzprep_c#+r'; $\ln(r)=$'+llr_expNzprep
# # #             explabel = r'; $KLD=('+str(gofs['kl_expNzvtruNz'])+','+str(gofs['kl_truNzvexpNz'])+r')$'#r'; $\sigma^{2}=$'+expNzprep_v+r'; $\chi^{2}=$'+expNzprep_c#+r'; $\log(r)=$'+lr_expNzprep

# #             #llr_smpNzprep = str(int(np.average(self.ll_smpNz)))
# #             logsmpNzprep_v = str(int(min(gofs['var_ls'])))#/(self.last_key.r+1.)
# #             logsmpNzprep_c = str(int(min(gofs['chi_ls'])))#/(self.last_key.r+1.)
# #             #lr_smpNzprep = str(int(np.log10(np.exp(np.average(self.ll_smpNz)))))
# #             smpNzprep_v = str(int(min(gofs['var_s'])))#/(self.last_key.r+1.)
# #             smpNzprep_c = str(int(min(gofs['chi_s'])))#/(self.last_key.r+1.)
#             logsmplabel = r'; $\max\ln \mathcal{L}='+str(round(max(both['ll_smpNz'])))+r'$'#r'; $\sigma^{2}=$'+logsmpNzprep_v+r'; $\chi^{2}=$'+logsmpNzprep_c#+r'; $\ln(r)=$'+llr_smpNzprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
#             smplabel = r'; $\min KLD=('+str(min(gofs['kl_smpNzvtruNz']))+','+str(min(gofs['kl_truNzvsmpNz']))+r')$'#r'; $\sigma^{2}=$'+smpNzprep_v+r'; $\chi^{2}=$'+smpNzprep_c#+r'; $\log(r)=$'+lr_smpNzprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))

#             logintlabel = r'; $\ln \mathcal{L}='+str(round(both['ll_intNz']))+r'$'
#             intlabel = r'; $KLD=('+str(gofs['kl_intNzvtruNz'])+','+str(gofs['kl_truNzvintNz'])+r')$'

#             logmmllabel = r'; $\ln \mathcal{L}='+str(round(both['ll_mmlNz']))+r'$'
#             mmllabel = r'; $KLD=('+str(gofs['kl_mmlNzvtruNz'])+','+str(gofs['kl_truNzvmmlNz'])+r')$'

#         else:
# #             logstklabel = ' '
# #             stklabel = ' '
# #             logmaplabel = ' '
# #             maplabel = ' '
# # #             logexplabel = ' '
# # #             explabel = ' '
#             logintlabel = ' '
#             intlabel = ' '
#             logmmllabel = ' '
#             mmllabel = ' '
#             logsmplabel = ' '#r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_ls']/(self.last_key.r+1.)))
#             smplabel = ' '#r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_s']/(self.last_key.r+1.)))
#             self.meta.logtruNz = [-1.]*self.meta.nbins
#             self.meta.truNz = [-1.]*self.meta.nbins

#         if self.meta.logtruNz is not None:
#             plotstep(sps_samp_log,self.meta.zrange,self.meta.lNz_range,w=2.,s='-',l=r'True $\ln N(z)$')
#             plotstep(sps_samp,self.meta.zrange,self.meta.Nz_range,w=2.,s='-',l=r'True $N(z)$')
# #             plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=2)
# #             plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=2)
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=2,l=r'True $\ln N(z)$')
# #         plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=2,l=r'True $N(z)$')
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logstkNz,s='--',w=2,l=r'Stacked $\ln N(z)$'+logstklabel)
# #         plotstep(sps_samp,self.meta.binends,self.meta.stkNz,s='--',w=2,l=r'Stacked $N(z)$'+stklabel)
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logmapNz,s='-.',w=2,l=r'MAP $\ln N(z)$'+logmaplabel)
# #         plotstep(sps_samp,self.meta.binends,self.meta.mapNz,s='-.',w=2,l=r'MAP $N(z)$'+maplabel)
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logexpNz,s=':',w=2,l=r'$\ln N(E(z))$'+logexplabel)
# #         plotstep(sps_samp,self.meta.binends,self.meta.expNz,s=':',w=2,l=r'$N(E(z))$'+explabel)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logmmlNz,s=':',w=2,l=r'MMLE $\ln N(z)$'+logmmllabel)
#         plotstep(sps_samp,self.meta.binends,self.meta.mmlNz,s=':',w=2,l=r'MMLE $N(z)$'+mmllabel)
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logintNz,s='-',a=0.5,w=1,l=r'Interim $\ln N(z)$'+logintlabel)
# #         plotstep(sps_samp,self.meta.binends,self.meta.intNz,s='-',a=0.5,w=1,l=r'Interim $N(z)$'+intlabel)
# #         plotstep(sps_samp_log,self.meta.binends,self.meta.logtruNz,s='-',w=0.5,l=r'Sampled $\ln N(z)$'+logsmplabel)
# #         plotstep(sps_samp,self.meta.binends,self.meta.truNz,s='-',w=0.5,l=r'Sampled $N(z)$'+smplabel)

#         self.plotone(sps_samp_log,self.meta.logmml,'-',3,r'MMLE $\ln N(z)$')
#         self.plotone(sps_samp,self.meta.mml,'-',3,r'MMLE $N(z)$')

        with open(os.path.join(self.meta.topdir,'samples.csv'),'rb') as csvfile:
            tuples = (line.split(None) for line in csvfile)
            alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
            alldata = np.array(alldata).T
        locs,scales = [],[]
        for k in xrange(self.meta.nbins):
            y_all = alldata[k].flatten()
            loc,scale = sp.stats.norm.fit_loc_scale(y_all)
            locs.append(loc)
            scales.append(scale)
#             sps_samp_log.hlines(loc,self.meta.binends[k],self.meta.binends[k+1],color='k',linestyle='--',linewidth=2.)
#             sps_samp.hlines(np.exp(loc),self.meta.binends[k],self.meta.binends[k+1],color='k',linestyle='--',linewidth=2.)
            sps_samp_log.fill_between([self.meta.binends[k],self.meta.binends[k+1]],loc-scale,loc+scale,color='k',alpha=0.1,linewidth=0.)
            sps_samp.fill_between([self.meta.binends[k],self.meta.binends[k+1]],np.exp(loc-scale),np.exp(loc+scale),color='k',alpha=0.1,linewidth=0.)
        loc = np.array(locs)
        self.plotone(locs,np.exp(locs),w=w_bfe,s=s_bfe,a=a_bfe,c=c_bfe,l=l_bfe)
#         plotstep(sps_samp_log,self.meta.binends,locs,s='--',w=2,l=r'Best Fit $\ln N(z)$')
#         plotstep(sps_samp,self.meta.binends,np.exp(locs),s='--',w=2,l=r'Best Fit $N(z)$')

        sps_samp_log.legend(fontsize='xx-small', loc='lower right')
        sps_samp.legend(fontsize='xx-small', loc='upper left')

        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)

        timesaver(self.meta,'samps-done',key)

    def plotone(self,logy,y,w=1.,s='-',a=1.,c='k',l=' '):
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]
        plotstep(sps_samp_log,self.meta.binends,logy,w=w,s=s,a=a,c=c,l=l+lnz)
        plotstep(sps_samp,self.meta.binends,y,w=w,s=s,a=a,c=c,l=l+nz)

#plot full posterior chain evolution
class plotter_chains(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_chain = 1.#/self.meta.nsteps#self.ncolors/ self.meta.nwalkers
        self.f_chains = plt.figure(figsize=(10,5*self.meta.nbins))
        self.f_chains.suptitle(self.meta.name)
        self.sps_chains = [self.f_chains.add_subplot(self.meta.nbins,2,2*k+1) for k in xrange(self.meta.nbins)]
        self.sps_pdfs = [self.f_chains.add_subplot(self.meta.nbins,2,2*(k+1)) for k in xrange(self.meta.nbins)]
        self.randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)#self.ncolors)

        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.plot([0],[0],color = 'k',label = 'Mean Sample Value',rasterized = True)
            sps_chain.set_ylim(-np.log(self.meta.ngals), np.log(self.meta.ngals / self.meta.bindif)+1)
            sps_chain.set_xlabel('iteration number')
            sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
            sps_chain.set_title(r'$\ln N(z)$ Parameter {} of {}'.format(k+1, self.meta.nbins))
            self.sps_pdfs[k].set_ylim(0.,1.)
            #self.sps_pdfs[k].semilogy()
            sps_pdf = self.sps_pdfs[k]
            sps_pdf.set_xlabel(r'$\theta_{'+str(k+1)+r'}$')
            sps_pdf.set_ylabel('kernel density estimate')
            sps_pdf.set_title(r'Distribution of $\theta_{'+str(k+1)+r'}$ Values')
#             self.yrange = [-1.,0.,1.,2.]
            sps_pdf.vlines(self.meta.logstkNz[k],0.,1.,linestyle=s_stk,linewidth=w_stk,color=c_stk,alpha=a_stk,label=l_stk+t+r'$_{k}$')
#             plotv(sps_pdf,self.meta.logstkNz[k],yrange,s=s_stk,w=w_stk,c=c_stk,a=a_stk,l=l_stk+t+r'$_{k}$')
#             sps_pdf.vlines(self.meta.logmapNz[k],0.,1.,linestyle='-.',w=2.)
#           plotv(sps_pdf,self.meta.logmapNz[k],yrange,s=s_map,w=w_map,c=c_map,a=a_map,l=l_map+t+r'$_{k}$')
# #             sps_pdf.vlines(self.meta.logexpNz[k],0.,1.,linestyle=':',w=2.,label=r'$E(z)$ value')
# #             plotv(sps_pdf,self.meta.logexpNz[k],yrange,s=s_exp,w=w_exp,c=c_exp,a=a_exp,l=l_exp+t+r'$_{k}$')
            sps_pdf.vlines(self.meta.logmmlNz[k],0.,1.,linestyle=s_mml,linewidth=w_mml,color=c_mml,alpha=a_mml,label=l_mml+t+r'$_{k}$')
#             plotv(sps_pdf,self.meta.logmmlNz[k],yrange,s=s_mml,w=w_mml,c=c_mml,a=a_mml,l=l_mml+t+r'$_{k}$')
            sps_pdf.vlines(self.meta.logintNz[k],0.,1.,linestyle=s_int,linewidth=w_int,color=c_int,alpha=a_int,label=l_int+t+r'$_{k}$')
#             plotv(sps_pdf,self.meta.logintNz[k],yrange,s=s_int,w=w_int,c=c_int,a=a_int,l=l_int+t+r'$_{k}$')
            if self.meta.logtruNz is not None:
                sps_pdf.vlines(self.meta.logtruNz[k],0.,1.,linestyle=s_tru,linewidth=w_tru,color=c_tru,alpha=a_tru,label=l_tru+t+r'$_{k}$')
#                 plotv(sps_pdf,self.meta.logtruNz[k],yrange,s=s_tru,w=w_tru,c=c_tru,a=a_tru,l=l_tru+t+r'$_{k}$')

    def plot(self,key):

      if key.burnin == False:

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_c = np.swapaxes(data,0,1).T

        randsteps = xrange(self.meta.ntimes)#random.sample(xrange(self.meta.ntimes),self.meta.ncolors)

        for k in xrange(self.meta.nbins):
            sps_pdf = self.sps_pdfs[k]
            mean = np.sum(plot_y_c[k])/(self.meta.ntimes*self.meta.nwalkers)
            self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                    [mean]*self.meta.ntimes,
                                    color = 'k',
                                    rasterized = True)

            x_all = plot_y_c[k].flatten()
            x_kde = x_all[:, np.newaxis]
            kde = skl.neighbors.KernelDensity(kernel='gaussian', bandwidth=1.0).fit(x_kde)
            x_plot = np.arange(np.min(plot_y_c[k]),np.max(plot_y_c[k]),0.1)[:, np.newaxis]
            log_dens = kde.score_samples(x_plot)
            sps_pdf.plot(x_plot[:, 0],np.exp(log_dens),color=self.meta.colors[key.r%self.ncolors],rasterized=True)
            for x in randsteps:
                for w in self.randwalks:
                    self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                            plot_y_c[k][w],
                                            color = self.meta.colors[w%self.ncolors],
                                            alpha = self.a_chain,
                                            rasterized = True)

            loc,scale = sp.stats.norm.fit_loc_scale(x_all)
            sps_pdf.vlines(loc,0.,1.,color=self.meta.colors[key.r%self.ncolors],linestyle=s_smp,linewidth=w_smp,alpha=a_smp)
#             plotv(sps_pdf,yrange,loc,s=s_smp,w=w_smp,c=self.meta.colors[key.r%self.ncolors],a=a_chain)
#             sps_pdf.axvspan(loc-scale,loc+scale,color=self.meta.colors[key.r%self.ncolors],alpha=0.1)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
              both = cpkl.load(statboth)

#         for k in xrange(self.meta.nbins):
#             self.sps_pdfs[k].vlines(both['mapvals'][-1][k],0.,1.,linewidth=2,color=self.meta.colors[key.r%self.ncolors])

        self.f_chains.savefig(os.path.join(self.meta.topdir,'chains.png'),dpi=100)

        timesaver(self.meta,'chains',key)

    def plotone(self,subplot,plot_x,plot_y,s,ylabel,a=1.):
        subplot.plot(plot_x,
                     plot_y,
                     color='k',
                     linewidth=2,
                     linestyle=s,
                     alpha=a,
                     label=ylabel)
        return

    def finish(self):
        timesaver(self.meta,'chains-start',key)

        with open(os.path.join(self.meta.topdir,'samples.csv'),'rb') as csvfile:
            tuples = (line.split(None) for line in csvfile)
            alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
            alldata = np.array(alldata).T

        maxsteps = self.last_key.r+2
        maxiternos = np.arange(0,maxsteps)
        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.set_xlim(-1*self.meta.miniters,(maxsteps+1)*self.meta.miniters)
            sps_chain = self.sps_chains[k]
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logstkNz[k]]*maxsteps,'--','Stacked value')
#             self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logmapNz[k]]*maxsteps,'-.','MAP value')
# #             self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logexpNz[k]]*maxsteps,':',r'$E(z)$ value')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logmmlNz[k]]*maxsteps,':','MMLE value')

#             if self.meta.logtruNz is not None:
#                 self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logtruNz[k]]*maxsteps,'-','True value')

            sps_pdf = self.sps_pdfs[k]
#             yrange = [-1.,0.,1.,2.]
#             if self.meta.logtruNz is not None:
#                 plotv(sps_pdf,self.meta.logtruNz[k],yrange,s=s_tru,w=w_tru,c=c_tru,a=a_tru,l=l_tru+t+r'$_{k}$')
# #                 sps_pdf.vlines(self.meta.logtruNz[k],0.,1.,linestyle='-',w=2.,label='True value')
# #             sps_pdf.vlines(self.meta.logstkNz[k],0.,1.,linestyle='--',w=2.,label='Stacked value')
# #             sps_pdf.vlines(self.meta.logmapNz[k],0.,1.,linestyle='-.',w=2.,label='MAP value')
# # #             sps_pdf.vlines(self.meta.logexpNz[k],0.,1.,linestyle=':',w=2.,label=r'$E(z)$ value')
#             sps_pdf.vlines(self.meta.logmmlNz[k],0.,1.,linestyle=':',w=2.,label='MMLE value')

            y_all = alldata[k]
            loc,scale = sp.stats.norm.fit_loc_scale(y_all)
            yrange = [-1.,0.,1.,2.]
#             plotv(sps_pdf,loc,yrange,s=s_bfe,w=w_bfe,c=c_bfe,a=a_bfe,l=l_bfe+t+r'$_{k}$')
            sps_pdf.vlines(loc,0.,1.,linestyle=s_bfe,linewidth=w_bfe,color=c_bfe,alpha=a_bfe,label=l_bfe+t+r'$_{k}$')
            sps_pdf.axvspan(loc-scale,loc+scale,color='k',alpha=0.1)

            sps_pdf.legend(fontsize='xx-small',loc='upper left')
            sps_chain.legend(fontsize='xx-small', loc='lower right')
            sps_chain.set_xlim(0,(self.last_key.r+1)*self.meta.miniters)

        self.f_chains.savefig(os.path.join(self.meta.topdir,'chains.png'),dpi=100)

        timesaver(self.meta,'chains-done',key)
