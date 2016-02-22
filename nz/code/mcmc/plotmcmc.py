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
import statmcmc as sm

# set up for better looking plots
title = 15
label = 15
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.titlesize'] = title
mpl.rcParams['axes.labelsize'] = label
mpl.rcParams['figure.subplot.left'] = 0.2
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.bottom'] = 0.2
mpl.rcParams['figure.subplot.top'] = 0.9
mpl.rcParams['figure.subplot.wspace'] = 0.5
mpl.rcParams['figure.subplot.hspace'] = 0.5

global lnz,nz,tv,t
lnz,nz,tv,t,kld = r'$\ln[N(z)]$',r'$N(z)$',r'$\vec{\theta}$',r'$\theta$','\n KLD='
# setting up unified appearance parameters
global s_tru,w_tru,a_tru,c_tru,d_tru,l_tru
s_tru,w_tru,a_tru,c_tru,d_tru,l_tru = '--',1.,1.,'k',[(0,(1,0.0001))],'True '
global s_int,w_int,a_int,c_int,d_int,l_int
s_int,w_int,a_int,c_int,d_int,l_int = '--',1.,0.5,'k',[(0,(1,0.0001))],'Interim '
global s_stk,w_stk,a_stk,c_stk,d_stk,l_stk
s_stk,w_stk,a_stk,c_stk,d_stk,l_stk = '--',1.,1.,'k',[(0,(2,2))],'Stacked '
# global s_map,w_map,a_map,c_map,d_map,l_map
# s_map,w_map,a_map,c_map,d_map,l_map = '--',1.5,0.75,'k',[(0,(4,4,2,4))],'MMAP '
# global s_exp,w_exp,a_exp,c_exp,d_exp,l_exp
# s_exp,w_exp,a_exp,c_exp,d_exp,l_exp = '--',1.5,0.25,'k',[(0,(4,4,2,4))],'MExp '
global s_mml,w_mml,a_mml,c_mml,d_mml,l_mml
s_mml,w_mml,a_mml,c_mml,d_mml,l_mml = '--',1.,1.,'k',[(0,(4,2))],'MMLE '
global s_smp,w_smp,a_smp,c_smp,d,smp,l_smp
s_smp,w_smp,a_smp,c_smp,d_smp,l_smp = '--',1.,1.,'k',[(0,(1,0.0001))],'Sampled '
global s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe
s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe = '--',2.,1.,'k',[(0,(1,0.0001))],'Mean of\n Samples '

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))],l=None):
    ploth(subplot,binends,plot,s,c,a,w,d,l)
    plotv(subplot,binends,plot,s,c,a,w,d)

def ploth(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))],l=None):
    subplot.hlines(plot,
                   binends[:-1],
                   binends[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=True)
def plotv(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))]):
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   rasterized=True)

def footer(subplot):
    subplot.annotate('Malz+2016',(0,0), (190,200), xycoords='axes fraction', textcoords='offset points', va='top')
    return

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
#         if meta.truNz is not None:
#             plot_true(meta)
        timesaver(meta,'iplot',meta.key)

# plot some individual posteriors
def plot_pdfs(meta):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    randos = random.sample(xrange(meta.ngals),len(meta.colors))
    for r in lrange(randos):
        plotstep(sps,meta.binends,meta.pdfs[randos[r]],c=meta.colors[r%len(meta.colors)])
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
    f.savefig(os.path.join(meta.topdir,'samplepzs.png'))
    return

# plot some samples from prior for one instantiation of survey
def plot_priorsamps(meta):
    priorsamps = np.array(meta.priordist.sample_ps(len(meta.colors))[0])
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
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
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
    plotstep(sps,meta.binends,meta.mean)
    for i in lrange(meta.ivals):
        plotstep(sps,meta.binends,meta.ivals[i],a=1./meta.factor,c=meta.colors[i%len(meta.colors)])
    f.savefig(os.path.join(meta.topdir,'initializations.png'),dpi=100)
    return

def plot_true(meta):
    f = plt.figure(figsize=(5,10))
    sps = f.add_subplot(2,1,1)
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(np.log(1./min(meta.bindifs)),np.log(meta.ngals/min(meta.bindifs)))#(-1.,np.log(test.ngals/min(test.meta.zdifs)))
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
#     plotstep(sps,test.binends,test.logtruNz,s=s_smp,w=w_smp,a=a_smp,c=c_smp,l=l_smp+lnz)
    plotstep(sps,meta.zrange,meta.lNz_range,s=s_tru,w=w_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru+lnz)
    plotstep(sps,meta.binends,meta.logstkNz,s=s_stk,w=w_stk,a=a_stk,c=c_stk,d=d_stk,l=l_stk+lnz)
#     plotstep(sps,meta.binends,meta.logmapNz,s=s_map,w=w_map,a=a_map,c=c_map,d=d_map,l=l_map+lnz)
#     plotstep(sps,meta.binends,meta.logexpNz,s=s_exp,w=w_exp,a=a_exp,c=c_exp,d=d_exp,l=l_exp+lnz)
    plotstep(sps,meta.binends,meta.logmmlNz,s=s_mml,w=w_mml,a=a_mml,c=c_mml,d=d_mml,l=l_mml+lnz)
    plotstep(sps,meta.binends,meta.logintNz,s=s_int,w=w_int,a=a_int,c=c_int,d=d_int,l=l_int+lnz)
    sps.legend(loc='lower right',fontsize='x-small')
    sps = f.add_subplot(2,1,2)
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$N(z)$')
#     sps.set_ylim(0.,test.ngals/min(test.bindifs))#(0.,test.ngals/min(test.meta.zdifs))
    sps.set_xlim(meta.binlos[0]-meta.bindif,meta.binhis[-1]+meta.bindif)
#     plotstep(sps,test.binends,test.truNz,s=s_smp,w=w_smp,a=a_smp,c=c_smp,l=l_smp+nz)
    plotstep(sps,meta.zrange,meta.Nz_range,s=s_tru,w=w_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru+nz)
    plotstep(sps,meta.binends,meta.stkNz,s=s_stk,w=w_stk,a=a_stk,c=c_stk,d=d_stk,l=l_stk+nz)# with $\sigma^{2}=$'+str(int(test.vsstack)))
#     plotstep(sps,meta.binends,meta.mapNz,s=s_map,w=w_map,a=a_msp,c=c_map,d=d_map,l=l_map+nz)# with $\sigma^{2}=$'+str(int(test.vsmapNz)))
#     plotstep(sps,meta.binends,meta.expNz,s=s_exp,w=w_exp,a=a_exp,c=c_exp,d=d_exp,l=l_exp+nz)# with $\sigma^{2}=$'+str(int(test.vsexpNz)))
    plotstep(sps,meta.binends,meta.mmlNz,s=s_mml,w=w_mml,a=a_mml,c=c_mml,d=d_mml,l=l_mml+nz)
    plotstep(sps,meta.binends,meta.intNz,s=s_int,w=w_int,a=a_int,c=c_int,d=d_int,l=l_int+nz)# with $\sigma^{2}=$'+str(int(test.vsinterim)))
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
        self.sps_times = self.f_times.add_subplot(1,1,1)
        self.a_times = float(len(self.meta.colors))/self.meta.nwalkers
#         if self.meta.mode == 'bins':
#             self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nbins) + ' bins')
#         if self.meta.mode == 'walkers':
#             self.sps_times.set_title('Autocorrelation Times for ' + str(self.meta.nwalkers) + ' walkers')
        self.sps_times.set_ylabel('autocorrelation time')
        self.sps_times.set_xlabel('number of iterations')
        self.sps_times.set_ylim(0, 100)

#         self.f_fracs = plt.figure(figsize=(5,5))
#         self.a_fracs = float(len(self.meta.colors))/self.meta.nwalkers
#         self.sps_fracs = self.f_fracs.add_subplot(1,1,1)
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
        self.f_times.savefig(os.path.join(self.meta.topdir,'times.pdf'),dpi=100)
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
        self.sps = self.f.add_subplot(1,1,1)
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
#         self.sps.fill_between(x_all,locs-scales,locs+scales,color='k',alpha=0.1,linewidth=0.)
        x_cor = [x_all[:-1],x_all[:-1],x_all[1:],x_all[1:]]
        y_cor = np.array([locs-scales,locs+scales,locs+scales,locs-scales])
        self.sps.fill(x_cor,y_cor,color='k',alpha=0.1,linewidth=0.)
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

#         yrange = self.medy#np.array(self.medy+[probs['lp_tru'],probs['lp_stkNz'],probs['lp_mapNz'],probs['lp_expNz']])
#         miny = np.min(yrange)-np.log(self.meta.ngals)
#         maxy = np.max(yrange)+np.log(self.meta.ngals)

        if self.meta.logtruNz is not None:
            self.plotone(probs['lp_tru'],w=w_tru,s=s_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru+tv)
            self.plotone(probs['lp_stk'],w=w_stk,s=s_stk,a=a_stk,c=c_stk,d=d_stk,l=l_stk+tv)
#             self.plotone(probs['lp_map'],w=w_map,s=s_map,a=a_map,c=c_map,d=d_map,l=l_map+tv)
#             self.plotone(probs['lp_exp'],w=w_exp,s=s_exp,a=a_exp,c=c_exp,d=d_exp,l=l_exp+tv)
            self.plotone(probs['lp_mml'],w=w_mml,s=s_mml,a=a_mml,c=c_mml,d=d_mml,l=l_mml+tv)#r'MMLE $\vec{\theta}$',w=2.,s=':')
#             self.plotone(probs['lp_int'],w=w_int,s=s_int,a=a_int,c=c_int,d=d_intl=l_int+tv

        self.sps.legend(fontsize='xx-small', loc='upper right')
        self.sps.set_xlim(-1*self.meta.miniters,(self.last_key.r+2)*self.meta.miniters)
#         self.sps.set_ylim(miny,maxy)
        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)

#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
#             both = cpkl.load(statboth)

        timesaver(self.meta,'probs-done',key)

    def plotone(self,plot_y,l=None,a=1.,c='k',w=1.,s='--',d=[(0,(1,0.0001))]):
        ploth(self.sps,self.meta.miniters*np.arange(0.,self.last_key.r+1),plot_y,s=s,c=c,a=a,w=w,d=d,l=l)

# plot full posterior samples
class plotter_samps(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_samp = 1.#/self.meta.ntimes#self.ncolors/self.meta.nwalkers
        self.f_samps = plt.figure(figsize=(5, 10))
        self.sps_samps = [self.f_samps.add_subplot(2,1,l+1) for l in xrange(0,2)]

#         self.f_comps = plt.figure(figsize=(5, 10))
#         self.sps_comps = [self.f_comps.add_subplot(2,1,l+1) for l in xrange(0,2)]

        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        sps_samp_log.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
#         sps_samp_log.set_ylim(-1.,m.log(self.meta.ngals/self.meta.bindif)+1.)
        sps_samp_log.set_xlabel(r'$z$')
        sps_samp_log.set_ylabel(r'$\ln N(z)$')
#         sps_samp_log.set_ylim(-1.,m.log(self.meta.ngals/self.meta.bindif)+1.)
        sps_samp.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
#         sps_samp.set_ylim(0.,self.meta.ngals/self.meta.bindif+self.meta.ngals)
        sps_samp.set_xlabel(r'$z$')
        sps_samp.set_ylabel(r'$N(z)$')

        if self.meta.logtruNz is not None:
            plotstep(sps_samp_log,self.meta.zrange,self.meta.lNz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru)#+lnz)
            plotstep(sps_samp,self.meta.zrange,self.meta.Nz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru)#+nz)
            sps_samp_log.set_ylim(np.min(self.meta.lNz_range)-1.,np.max(self.meta.lNz_range)+1.)
            sps_samp.set_ylim(0,max(self.meta.Nz_range)+self.meta.ngals)
        self.plotsamp(self.meta.logintNz,self.meta.intNz,w=w_int,s=s_int,a=a_int,c=c_int,d=d_int,l=l_int)

    def plot(self,key):

        with open(os.path.join(self.meta.topdir,'iterno.p')) as where:
            iterno = cpkl.load(where)

        if (self.meta.plotonly == 0 and key.burnin == False) or (self.meta.plotonly == 1 and key.r >= iterno-self.meta.factor):

            print('about to plot a sample')
            data = key.load_state(self.meta.topdir)['chains']

            plot_y_ls = np.swapaxes(data,0,1)
#             plot_y_s = np.exp(plot_y_ls)

            randsteps = random.sample(xrange(self.meta.ntimes),1)#xrange(self.meta.ntimes)#self.ncolors)
            randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)#self.ncolors)

            sps_samp_log = self.sps_samps[0]
            sps_samp = self.sps_samps[1]
            for w in randwalks:
                for x in randsteps:
                    plotstep(sps_samp_log,self.meta.binends,plot_y_ls[x][w],s=s_smp,d=d_smp,w=w_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])
                    plotstep(sps_samp,self.meta.binends,np.exp(plot_y_ls[x][w]),s=s_smp,d=d_smp,w=w_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])
#                 self.plotone(plot_y_ls[x][w],plot_y_s[x][w],w=w_smp,s=s_smp,a=self.a_samp,c=self.meta.colors[key.r%self.ncolors])

            print('plotted a sample')
            self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)

        timesaver(self.meta,'samps',key)

    def finish(self):
        timesaver(self.meta,'samps-start',key)
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]

        self.calcbfe()
#         self.ploterr(sps_samp_log,sps_samp)
#         if self.meta.logtruNz is not None:
#             sps_samp_log.set_ylim(np.min(self.meta.lNz_range)-1.,np.max(self.meta.lNz_range)+1.)
#             sps_samp.set_ylim(0,max(self.meta.Nz_range)+self.meta.ngals)
        sps_samp_log.legend(fontsize='xx-small', loc='upper left')
        sps_samp.legend(fontsize='xx-small', loc='upper left')
        footer(sps_samp_log)
        footer(sps_samp)

        self.summary()

        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)
        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.pdf'),dpi=100)

        timesaver(self.meta,'samps-done',key)

    def calcbfe(self):

        with open(os.path.join(self.meta.topdir,'samples.csv'),'rb') as csvfile:
            tuples = (line.split(None) for line in csvfile)
            alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples]
            alldata = np.array(alldata).T
#             print(str(self.last_key.r)+' alldata shape '+str(np.shape(alldata)))

        locs,scales = [],[]
        x_cors,y_cors,y_cors2 = [],[],[]
        for k in xrange(self.meta.nbins):
            y_all = alldata[k].flatten()
            loc,scale = sp.stats.norm.fit_loc_scale(y_all)
            x_cor = [self.meta.binends[k],self.meta.binends[k],self.meta.binends[k+1],self.meta.binends[k+1]]
            y_cor = np.array([loc-scale,loc+scale,loc+scale,loc-scale])
            y_cor2 = np.array([loc-2*scale,loc+2*scale,loc+2*scale,loc-2*scale])
            locs.append(loc)
            scales.append(scale)
            x_cors.append(x_cor)
            y_cors.append(y_cor)
            y_cors2.append(y_cor2)
        self.locs = np.array(locs)
        self.scales = np.array(scales)
        self.x_cors = np.array(x_cors)
        self.y_cors = np.array(y_cors)
        self.y_cors2 = np.array(y_cors2)

    def ploterr(self,logplot,plot):

        for k in xrange(self.meta.nbins):
            logplot.fill(self.x_cors[k],self.y_cors2[k],color='k',alpha=0.2,linewidth=0.)
            plot.fill(self.x_cors[k],np.exp(self.y_cors2[k]),color='k',alpha=0.2,linewidth=0.)
            logplot.fill(self.x_cors[k],self.y_cors[k],color='k',alpha=0.3,linewidth=0.)
            plot.fill(self.x_cors[k],np.exp(self.y_cors[k]),color='k',alpha=0.3,linewidth=0.)

        self.plotsamp(self.locs,np.exp(self.locs),w=w_bfe,s=s_bfe,a=a_bfe,c=c_bfe,d=d_bfe,l=l_bfe)

    def plotsamp(self,logy,y,w=1.,s='--',a=1.,c='k',d=[(0,(1,0.0001))],l=' '):
        sps_samp_log = self.sps_samps[0]
        sps_samp = self.sps_samps[1]
        plotstep(sps_samp_log,self.meta.binends,logy,w=w,s=s,d=d,a=a,c=c,l=l)#+lnz)
        plotstep(sps_samp,self.meta.binends,y,w=w,s=s,d=d,a=a,c=c,l=l)#+nz)

    def plotcomp(self,logy,y,w=1.,s='--',a=1.,c='k',d=[(0,(1,0.0001))],l=' '):
        sps_comp_log = self.sps_comps[0]
        sps_comp = self.sps_comps[1]
        plotstep(sps_comp_log,self.meta.binends,logy,w=w,s=s,d=d,a=a,c=c,l=l)
        plotstep(sps_comp,self.meta.binends,y,w=w,s=s,d=d,a=a,c=c,l=l)

    def prepstats(self):

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
            kls = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]
#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
#             both = cpkl.load(statboth)
#         for v in lrange(both['mapvals']):
#             plotstep(sps_samp_log,self.meta.binends,both['mapvals'][v],w=2,c=self.meta.colors[v%self.ncolors])
#             plotstep(sps_samp,self.meta.binends,np.exp(both['mapvals'][v]),w=2,c=self.meta.colors[v%self.ncolors])
        self.kl_mml = min(kls['kl_mmlvtru'],kls['kl_truvmml'])
        self.kl_stk = min(kls['kl_stkvtru'],kls['kl_truvstk'])
#         self.kl_map = min(kls['kl_mapvtru'],kls['kl_truvmap'])
#         self.kl_exp = min(kls['kl_expvtru'],kls['kl_truvexp'])
        self.kl_smp = min(sm.calckl(self.meta.bindifs,self.locs,self.meta.logtruNz))

    def summary(self):

        self.f_comps = plt.figure(figsize=(5, 10))
        self.sps_comps = [self.f_comps.add_subplot(2,1,l+1) for l in xrange(0,2)]
        sps_comp_log = self.sps_comps[0]
        sps_comp = self.sps_comps[1]

        sps_comp_log.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
        sps_comp_log.set_xlabel(r'$z$')
        sps_comp_log.set_ylabel(r'$\ln N(z)$')
        sps_comp.set_xlim(self.meta.binends[0]-self.meta.bindif,self.meta.binends[-1]+self.meta.bindif)
        sps_comp.set_xlabel(r'$z$')
        sps_comp.set_ylabel(r'$N(z)$')

        if self.meta.logtruNz is not None:
            self.prepstats()
            plotstep(sps_comp_log,self.meta.zrange,self.meta.lNz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru)#+lnz)
            plotstep(sps_comp,self.meta.zrange,self.meta.Nz_range,w=w_tru,s=s_tru,a=a_tru,c=c_tru,d=d_tru,l=l_tru)#+nz)
            sps_comp_log.set_ylim(np.min(self.meta.lNz_range)-1.,np.max(self.meta.lNz_range)+1.)
            sps_comp.set_ylim(0,max(self.meta.Nz_range)+self.meta.ngals)
        else:
            self.kl_stk,self.kl_mml,self.kl_smp = None,None,None

        self.prepstats()

#         plotstep(sps_comp_log,self.meta.binends,self.meta.logintNz,w=w_int,s=s_int,a=a_int,c=c_int,l=l_int+lnz)
#         plotstep(sps_comp,self.meta.binends,self.meta.intNz,w=w_int,s=s_int,a=a_int,c=c_int,l=l_int+nz)
        self.plotcomp(self.meta.logintNz,self.meta.intNz,w=w_int,s=s_int,a=a_int,c=c_int,d=d_int,l=l_int)#+kld+str(self.kl_stk))
        self.plotcomp(self.meta.logstkNz,self.meta.stkNz,w=w_stk,s=s_stk,a=a_stk,c=c_stk,d=d_stk,l=l_stk+kld+str(self.kl_stk))
#         self.plotcomp(self.meta.logmapNz,self.meta.mapNz,w=w_map,s=s_map,a=a_map,c=c_map,d=d_map,l=l_map+kld+str(self.kl_map))
#         self.plotcomp(self.meta.logexpNz,self.meta.expNz,w=w_exp,s=s_exp,a=a_exp,c=c_exp,d=d_exp,l=l_exp+kld+str(self.kl_exp))
        self.plotcomp(self.meta.logmmlNz,self.meta.mmlNz,w=w_mml,s=s_mml,a=a_mml,c=c_mml,d=d_mml,l=l_mml+kld+str(self.kl_mml))

#         self.ploterr(sps_comp_log,sps_comp)
        self.plotcomp(self.locs,np.exp(self.locs),w=w_bfe,s=s_bfe,a=a_bfe,c=c_bfe,d=d_bfe,l=l_bfe+kld+str(self.kl_smp))

        sps_comp_log.legend(fontsize='xx-small', loc='upper left')
        sps_comp.legend(fontsize='xx-small', loc='upper left')
        footer(sps_comp_log)
        footer(sps_comp)

        self.f_comps.savefig(os.path.join(self.meta.topdir,'comps.png'),dpi=100)
        self.f_comps.savefig(os.path.join(self.meta.topdir,'comps.pdf'),dpi=100)

#plot full posterior chain evolution
class plotter_chains(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_chain = 1.#/self.meta.nsteps#self.ncolors/ self.meta.nwalkers
        self.f_chains = plt.figure(figsize=(10,5*self.meta.nbins))
        self.sps_chains = [self.f_chains.add_subplot(self.meta.nbins,2,2*k+1) for k in xrange(self.meta.nbins)]
        self.sps_pdfs = [self.f_chains.add_subplot(self.meta.nbins,2,2*(k+1)) for k in xrange(self.meta.nbins)]
        self.randwalks = random.sample(xrange(self.meta.nwalkers),1)#xrange(self.meta.nwalkers)#self.ncolors)

        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.plot([0],[0],color=c_smp,label=l_smp+r'$\theta_{'+str(k)+r'}$',alpha=a_smp,linewidth=w_smp,linestyle=s_smp,rasterized = True)
            sps_chain.set_ylim(-np.log(self.meta.ngals), np.log(self.meta.ngals / self.meta.bindif)+1)
            sps_chain.set_xlabel('iteration number')
            sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
            self.sps_pdfs[k].set_ylim(0.,1.)
            #self.sps_pdfs[k].semilogy()
            sps_pdf = self.sps_pdfs[k]
            sps_pdf.set_xlabel(r'$\theta_{'+str(k+1)+r'}$')
            sps_pdf.set_ylabel('kernel density estimate')
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

    def plotone(self,subplot,plot_x,plot_y,w=2,c='k',s='--',d=[(0,(1,0.0001))],l=' ',a=1.):
        subplot.plot(plot_x,
                     plot_y,
                     color=c,
                     linewidth=w,
                     linestyle=s,
                     dashes=d,
                     alpha=a,
                     label=l)
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

            sps_pdf = self.sps_pdfs[k]

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
