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

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,style='-',col='k',a=1,lw=1,lab=None):
    subplot.hlines(plot,
                   binends[:-1],
                   binends[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col,
                   alpha=a,
                   label=lab,
                   rasterized=True)
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col,
                   alpha=a,
                   rasterized=True)

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
    f.suptitle('Observed galaxy posteriors for '+meta.name)
    #sps.set_title('shape='+str(meta.shape)+', noise='+str(meta.noise))
    randos = random.sample(xrange(meta.ngals),len(meta.colors))
    for r in lrange(randos):
        plotstep(sps,meta.binends,meta.pobs[randos[r]],col=meta.colors[r%len(meta.colors)])
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
    f.suptitle(r'Prior samples for '+meta.name+':')
    sps = f.add_subplot(1,1,1)
    sps.set_title(r'$q='+str(meta.q)+r'$, $e='+str(meta.e)+r'$, $t='+str(meta.t)+r'$')
    sps.set_xlabel(r'$z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlim(meta.binends[0]-meta.bindif,meta.binends[-1]+meta.bindif)#,s_run.seed)#max(n_run.full_logflatNz)+m.log(s_run.seed/meta.zdif)))
    plotstep(sps,meta.binends,meta.logflatNz,lab=r'flat $\ln N(z)$')
    for c in lrange(meta.colors):
        plotstep(sps,meta.binends,priorsamps[c],col=meta.colors[c])
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
        plotstep(sps,meta.binends,meta.ivals[i],a=1./meta.factor,col=meta.colors[i%len(meta.colors)])
    f.savefig(os.path.join(meta.topdir,'initializations.png'),dpi=100)
    return

# most generic plotter, specific plotters below inherit from this to get handle
class plotter(distribute.consumer):
    def handle(self, key):
        self.last_key = key
        print(self.meta.name+' last key is '+str(self.last_key))
        self.plot(key)

# plot autocorrelation times and acceptance fractions
class plotter_timefrac(plotter):

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

        self.f_fracs = plt.figure(figsize=(5,5))
        self.f_fracs.suptitle(self.meta.name)
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

        self.f_times.savefig(os.path.join(self.meta.topdir,'times.png'),dpi=100)

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

        self.f_fracs.savefig(os.path.join(self.meta.topdir,'fracs.png'),dpi=100)

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
        self.a_probs = 1./self.meta.factor#self.meta.nwalkers
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
        plot_y = np.swapaxes(data,0,1).T
#         if key.r != 0:
        self.medy.append(np.median([np.median(plot_y[w]) for w in xrange(self.meta.nwalkers)]))

        randwalks = xrange(self.meta.nwalkers)#random.sample(xrange(self.meta.nwalkers),1)
        for w in randwalks:
            self.sps.plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,
                          plot_y[w],
                          c=self.meta.colors[w%self.ncolors],
                          alpha=self.a_probs,
                          rasterized=True)

        self.f.savefig(os.path.join(self.meta.topdir,'probs.png'),dpi=100)

        timesaver(self.meta,'probs',key)

    def finish(self):

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as statprobs:
            probs = cpkl.load(statprobs)

        yrange = self.medy#np.array(self.medy+[probs['lp_true'],probs['lp_stack'],probs['lp_mapNz'],probs['lp_expNz']])
        miny = np.min(yrange)-np.log(self.meta.ngals)
        maxy = np.max(yrange)+np.log(self.meta.ngals)

        self.sps.plot(self.meta.miniters*np.arange(0,self.last_key.r+2),
                      [int(probs['lp_true'])]*(self.last_key.r+2),
                      label=r'True $\vec{\theta}$',
                      color='k',linewidth=2,linestyle='-')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_stack'])]*(self.last_key.r+2),
                      label=r'Stacked $\vec{\theta}$',
                      color='k',linewidth=2,linestyle='--')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_mapNz'])]*(self.last_key.r+2),
                      label=r'MAP $\vec{\theta}$',
                      color='k',linewidth=2,linestyle='-.')
#         self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
#                       [int(probs['lp_expNz'])]*(self.last_key.r+2),
#                       label=r'$E[z]$ $\vec{\theta}$',
#                       color='k',linewidth=2,linestyle=':')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_mleNz'])]*(self.last_key.r+2),
                      label=r'MLE $\vec{\theta}$',
                      color='k',linewidth=2,linestyle=':')
        self.sps.plot(self.meta.miniters*np.arange(0.,self.last_key.r+2),
                      [int(probs['lp_true'])]*(self.last_key.r+2),
                      label=r'Sampled $\vec{\theta}$',
                      color='k',linewidth=1,linestyle='-')

        self.sps.legend(fontsize='xx-small', loc='upper right')
        self.sps.set_xlim(-1*self.meta.miniters,(self.last_key.r+2)*self.meta.miniters)
        self.sps.set_ylim(miny,maxy)
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

        self.plot_llr()

        timesaver(self.meta,'probs-done',key)

    def plot_llr(self):
        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
            both = cpkl.load(statboth)
        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as statchains:
            gofs = cpkl.load(statchains)#self.meta.key.load_stats(self.meta.topdir,'chains',self.last_key.r+1)[0]

        f = plt.figure(figsize=(10,5))
        f.suptitle(self.meta.name)

        sps = [f.add_subplot(1,2,l+1) for l in xrange(0,2)]

        alldata = np.concatenate((both['llr_stack'],both['llr_mapNz'],both['llr_mleNz']))
        min_llr = np.min(alldata)
        max_llr = np.max(alldata)
        datarange = np.linspace(min_llr,max_llr,10*self.meta.nwalkers)[:, np.newaxis]
        kde_stack = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_stack'][:, np.newaxis])
        plot_stack = kde_stack.score_samples(datarange)
        kde_mapNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_mapNz'][:, np.newaxis])
        plot_mapNz = kde_mapNz.score_samples(datarange)
#         kde_expNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_expNz'][:, np.newaxis])
#         plot_expNz = kde_expNz.score_samples(datarange)
        kde_mleNz = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=30.).fit(both['llr_mleNz'][:, np.newaxis])
        plot_mleNz = kde_mleNz.score_samples(datarange)

        sps[0].set_title('Log Likelihood Ratio')
        sps[0].semilogy()
        sps[0].plot(datarange[:,0],np.exp(plot_stack),color='k',linestyle='--',label=r'Stacked $R$, $\max(R)='+str(round(np.max(both['llr_stack']),3))+r'$')
        sps[0].plot(datarange[:,0],np.exp(plot_mapNz),color='k',linestyle='-.',label=r'MAP $R$, $\max(R)='+str(round(np.max(both['llr_mapNz']),3))+r'$')
#         sps[0].plot(datarange[:,0],np.exp(plot_expNz),color='k',linestyle=':',label=r'$N(E[z])$ $R$, $\max(R)='+str(round(np.max(both['llr_expNz']),3))+r'$')
        sps[0].plot(datarange[:,0],np.exp(plot_mleNz),color='k',linestyle=':',label=r'MLE $R$, $\max(R)='+str(round(np.max(both['llr_mleNz']),3))+r'$')
        sps[0].legend(fontsize='xx-small', loc='upper left')
        sps[0].set_ylim(1e-10,1)
        sps[0].set_xlabel('log likelihood ratio')
        sps[0].set_ylabel('kernel density estimate')

        alldata = np.concatenate((gofs['kl_sampvtrue'],gofs['kl_truevsamp']))
        min_kl = np.min(alldata)
        max_kl = np.max(alldata)
        datarange = np.linspace(min_kl,max_kl,10*self.meta.nwalkers)[:, np.newaxis]
        kde_sampvtrue = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=1.).fit(gofs['kl_sampvtrue'][:, np.newaxis])
        plot_sampvtrue = kde_sampvtrue.score_samples(datarange)
        kde_truevsamp = sklearn.neighbors.KernelDensity(kernel='gaussian',bandwidth=1.).fit(gofs['kl_truevsamp'][:, np.newaxis])
        plot_truevsamp = kde_truevsamp.score_samples(datarange)

        sps[1].set_title('Kullback-Leibler Divergence')
        yrange = np.concatenate((np.exp(plot_sampvtrue),np.exp(plot_truevsamp)))

        sps[1].vlines(gofs['kl_stackvtrue'],0.,1.,color='k',linestyle='--',label=r'Stacked $KLD=('+str(gofs['kl_stackvtrue'])+','+str(gofs['kl_truevstack'])+r')$')
        sps[1].vlines(gofs['kl_truevstack'],0.,1.,color='k',linestyle='--')
        sps[1].vlines(gofs['kl_mapNzvtrue'],0.,1.,color='k',linestyle='-.',label=r'MAP $KLD=('+str(gofs['kl_mapNzvtrue'])+','+str(gofs['kl_truevmapNz'])+r')$')
        sps[1].vlines(gofs['kl_truevmapNz'],0.,1.,color='k',linestyle='-.')
#         sps[1].vlines(gofs['kl_expNzvtrue'],0.,1.,color='k',linestyle=':',label=r'$E[z]$ $KLD=('+str(gofs['kl_expNzvtrue'])+','+str(gofs['kl_truevexpNz'])+r')$')
#         sps[1].vlines(gofs['kl_truevexpNz'],0.,1.,color='k',linestyle=':')
        sps[1].vlines(gofs['kl_mleNzvtrue'],0.,1.,color='k',linestyle=':',label=r'MLE $KLD=('+str(gofs['kl_mleNzvtrue'])+','+str(gofs['kl_truevmleNz'])+r')$')
        sps[1].vlines(gofs['kl_truevmleNz'],0.,1.,color='k',linestyle=':')
        sps[1].vlines(gofs['kl_intNzvtrue'],0.,1.,alpha=0.5,color='k',linestyle='-',label=r'Interim $KLD=('+str(gofs['kl_intNzvtrue'])+','+str(gofs['kl_truevintNz'])+r')$')
        sps[1].vlines(gofs['kl_truevintNz'],0.,1.,alpha=0.5,color='k',linestyle='-')
        sps[1].plot(datarange[:,0],np.exp(plot_sampvtrue),color='k',linewidth=2,label=r'Sampled $\min KLD=('+str(np.min(gofs['kl_sampvtrue'].flatten()))+','+str(np.min(gofs['kl_truevsamp'].flatten()))+r')$')
        sps[1].plot(datarange[:,0],np.exp(plot_truevsamp),color='k',linewidth=2)
        sps[1].legend(fontsize='xx-small', loc='upper left')
        sps[1].set_xlabel('Kullback-Leibler divergence')
        sps[1].set_ylabel('kernel density estimate')
        sps[1].semilogx()

        f.savefig(os.path.join(self.meta.topdir,'llr.png'),dpi=100)


# plot full posterior samples
class plotter_samps(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_samp = 1./self.meta.ntimes#self.ncolors/self.meta.nwalkers
        self.f_samps = plt.figure(figsize=(5, 10))
        self.f_samps.suptitle(self.meta.name)
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

        if self.meta.logtrueNz is not None:
            plotstep(sps_samp_log,self.meta.binends,self.meta.logtrueNz,style='-',lw=2,lab=r'True $\ln N(z)$')
            plotstep(sps_samp,self.meta.binends,self.meta.trueNz,style='-',lw=2,lab=r'True $N(z)$')

    def plot(self,key):

#         if key.burnin == False:

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_ls = np.swapaxes(data,0,1)
        plot_y_s = np.exp(plot_y_ls)

        randsteps = xrange(self.meta.ntimes)#random.sample(xrange(self.meta.ntimes),1)#self.ncolors)
        randwalks = xrange(self.meta.nwalkers)#random.sample(xrange(self.meta.nwalkers),1)#self.ncolors)
            #self.a_samp = (key.r+1)/self.meta.nbins

        for w in randwalks:
            for x in randsteps:
#                 self.sps_samps[0].step(self.meta.binlos,plot_y_ls[x][w],color=self.meta.colors[key.r%self.ncolors],where='post',alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
#                 self.sps_samps[1].step(self.meta.binlos,plot_y_s[x][w],color=self.meta.colors[key.r%self.ncolors],where='post',alpha=self.a_samp,rasterized=True)#,label=str(self.meta.miniters*(key.r+1)))
                    self.sps_samps[0].hlines(plot_y_ls[x][w],
                                             self.meta.binends[:-1],
                                             self.meta.binends[1:],
                                             color=self.meta.colors[key.r%self.ncolors],
                                             alpha=self.a_samp,
                                             rasterized=True)
                    self.sps_samps[1].hlines(plot_y_s[x][w],
                                             self.meta.binends[:-1],
                                             self.meta.binends[1:],
                                             color=self.meta.colors[key.r%self.ncolors],
                                             alpha=self.a_samp,
                                             rasterized=True)

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

        for v in lrange(both['mapvals']):
            plotstep(sps_samp_log,self.meta.binends,both['mapvals'][v],lw=2,col=self.meta.colors[v%self.ncolors])
            plotstep(sps_samp,self.meta.binends,np.exp(both['mapvals'][v]),lw=2,col=self.meta.colors[v%self.ncolors])

        if self.meta.logtrueNz is not None:
#             #llr_stackprep = str(int(gof['llr_stack']))
#             logstackprep_v = str(int(gofs['vslogstack']))
#             logstackprep_c = str(int(gofs['cslogstack']))
#             #lr_stackprep = str(int(np.log10(np.exp(gof['llr_stack']))))
#             stackprep_v = str(int(gofs['vsstack']))
#             stackprep_c = str(int(gofs['csstack']))
            logstacklabel = r'; $\ln \mathcal{L}='+str(round(both['ll_stack']))+r'$'#r'; $\sigma^{2}=$'+logstackprep_v+r'; $\chi^{2}=$'+logstackprep_c#+r'; $\ln(r)=$'+llr_stackprep
            stacklabel = r'; $KLD=('+str(gofs['kl_stackvtrue'])+','+str(gofs['kl_truevstack'])+r')$'#r'; $\sigma^{2}=$'+stackprep_v+r'; $\chi^{2}=$'+stackprep_c#+r'; $\log(r)=$'+lr_stackprep

#             #llr_mapNzprep = str(int(gof['llr_mapNz']))
#             logmapNzprep_v = str(int(gofs['vslogmapNz']))
#             logmapNzprep_c = str(int(gofs['cslogmapNz']))
#             #lr_mapNzprep = str(int(np.log10(np.exp(gof['llr_mapNz']))))
#             mapNzprep_v = str(int(gofs['vsmapNz']))
#             mapNzprep_c = str(int(gofs['csmapNz']))
            logmaplabel = r'; $\ln \mathcal{L}='+str(round(both['ll_mapNz']))+r'$'#r'; $\sigma^{2}=$'+logmapNzprep_v+r'; $\chi^{2}=$'+logmapNzprep_c#+r'; $\ln(r)=$'+llr_mapNzprep
            maplabel = r'; $KLD=('+str(gofs['kl_mapNzvtrue'])+','+str(gofs['kl_truevmapNz'])+r')$'#r'; $\sigma^{2}=$'+mapNzprep_v+r'; $\chi^{2}=$'+mapNzprep_c#+r'; $\log(r)=$'+lr_mapNzprep

# #             #llr_expNzprep = str(int(gof['llr_expNz']))
# #             logexpNzprep_v = str(int(gofs['vslogexpNz']))
# #             logexpNzprep_c = str(int(gofs['cslogexpNz']))
# #             #lr_expNzprep = str(int(np.log10(np.exp(gof['llr_expNz']))))
# #             expNzprep_v = str(int(gofs['vsexpNz']))
# #             expNzprep_c = str(int(gofs['csexpNz']))
#             logexplabel = r'; $\ln \mathcal{L}='+str(both['ll_expNz'])+r'$'#r'; $\sigma^{2}=$'+logexpNzprep_v+r'; $\chi^{2}=$'+logexpNzprep_c#+r'; $\ln(r)=$'+llr_expNzprep
#             explabel = r'; $KLD=('+str(gofs['kl_expNzvtrue'])+','+str(gofs['kl_truevexpNz'])+r')$'#r'; $\sigma^{2}=$'+expNzprep_v+r'; $\chi^{2}=$'+expNzprep_c#+r'; $\log(r)=$'+lr_expNzprep

#             #llr_sampprep = str(int(np.average(self.ll_samp)))
#             logsampprep_v = str(int(min(gofs['var_ls'])))#/(self.last_key.r+1.)
#             logsampprep_c = str(int(min(gofs['chi_ls'])))#/(self.last_key.r+1.)
#             #lr_sampprep = str(int(np.log10(np.exp(np.average(self.ll_samp)))))
#             sampprep_v = str(int(min(gofs['var_s'])))#/(self.last_key.r+1.)
#             sampprep_c = str(int(min(gofs['chi_s'])))#/(self.last_key.r+1.)
            logsamplabel = r'; $\max\ln \mathcal{L}='+str(round(max(both['ll_samp'])))+r'$'#r'; $\sigma^{2}=$'+logsampprep_v+r'; $\chi^{2}=$'+logsampprep_c#+r'; $\ln(r)=$'+llr_sampprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))
            samplabel = r'; $\min KLD=('+str(min(gofs['kl_sampvtrue']))+','+str(min(gofs['kl_truevsamp']))+r')$'#r'; $\sigma^{2}=$'+sampprep_v+r'; $\chi^{2}=$'+sampprep_c#+r'; $\log(r)=$'+lr_sampprep#[(self.last_key.r+1.)/2:])/(self.last_key.r+1.)))

            logintlabel = r'; $\ln \mathcal{L}='+str(round(both['ll_intNz']))+r'$'
            intlabel = r'; $KLD=('+str(gofs['kl_intNzvtrue'])+','+str(gofs['kl_truevintNz'])+r')$'

            logmlelabel = r'; $\ln \mathcal{L}='+str(round(both['ll_mleNz']))+r'$'
            mlelabel = r'; $KLD=('+str(gofs['kl_mleNzvtrue'])+','+str(gofs['kl_truevmleNz'])+r')$'

        else:
            logstacklabel = ' '
            stacklabel = ' '
            logmaplabel = ' '
            maplabel = ' '
#             logexplabel = ' '
#             explabel = ' '
            logintlabel = ' '
            intlabel = ' '
            logmlelabel = ' '
            mlelabel = ' '
            logsamplabel = ' '#r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_ls']/(self.last_key.r+1.)))
            samplabel = ' '#r' $\sigma^{2}=$'+str(int(min(gofs['var_ls'])))#gofs['tot_var_s']/(self.last_key.r+1.)))
            self.meta.logtrueNz = [-1.]*self.meta.nbins
            self.meta.trueNz = [-1.]*self.meta.nbins

#         plotstep(sps_samp_log,self.meta.binends,self.meta.logtrueNz,style='-',lw=2,lab=r'True $\ln N(z)$')
#         plotstep(sps_samp,self.meta.binends,self.meta.trueNz,style='-',lw=2,lab=r'True $N(z)$')
        plotstep(sps_samp_log,self.meta.binends,self.meta.logstack,style='--',lw=2,lab=r'Stacked $\ln N(z)$'+logstacklabel)
        plotstep(sps_samp,self.meta.binends,self.meta.stack,style='--',lw=2,lab=r'Stacked $N(z)$'+stacklabel)
        plotstep(sps_samp_log,self.meta.binends,self.meta.logmapNz,style='-.',lw=2,lab=r'MAP $\ln N(z)$'+logmaplabel)
        plotstep(sps_samp,self.meta.binends,self.meta.mapNz,style='-.',lw=2,lab=r'MAP $N(z)$'+maplabel)
#         plotstep(sps_samp_log,self.meta.binends,self.meta.logexpNz,style=':',lw=2,lab=r'$\ln N(E(z))$'+logexplabel)
#         plotstep(sps_samp,self.meta.binends,self.meta.expNz,style=':',lw=2,lab=r'$N(E(z))$'+explabel)
        plotstep(sps_samp_log,self.meta.binends,self.meta.logmleNz,style=':',lw=2,lab=r'MLE $\ln N(z)$'+logmlelabel)
        plotstep(sps_samp,self.meta.binends,self.meta.mleNz,style=':',lw=2,lab=r'MLE $N(z)$'+mlelabel)
        plotstep(sps_samp_log,self.meta.binends,self.meta.logintNz,style='-',a=0.5,lw=1,lab=r'Interim $\ln N(z)$'+logintlabel)
        plotstep(sps_samp,self.meta.binends,self.meta.intNz,style='-',a=0.5,lw=1,lab=r'Interim $N(z)$'+intlabel)
        plotstep(sps_samp_log,self.meta.binends,self.meta.logtrueNz,style='-',lw=0.5,lab=r'Sampled $\ln N(z)$'+logsamplabel)
        plotstep(sps_samp,self.meta.binends,self.meta.trueNz,style='-',lw=0.5,lab=r'Sampled $N(z)$'+samplabel)

#         self.plotone(sps_samp_log,self.meta.logmle,'-',3,r'MLE $\ln N(z)$')
#         self.plotone(sps_samp,self.meta.mle,'-',3,r'MLE $N(z)$')

        sps_samp_log.legend(fontsize='xx-small', loc='lower right')
        sps_samp.legend(fontsize='xx-small', loc='upper left')

        self.f_samps.savefig(os.path.join(self.meta.topdir,'samps.png'),dpi=100)

        timesaver(self.meta,'samps-done',key)

#plot full posterior chain evolution
class plotter_chains(plotter):

    def __init__(self, meta):
        self.meta = meta
        self.ncolors = len(self.meta.colors)
        self.a_chain = 1./self.meta.factor#self.ncolors/ self.meta.nwalkers
        self.f_chains = plt.figure(figsize=(10,5*self.meta.nbins))
        self.f_chains.suptitle(self.meta.name)
        self.sps_chains = [self.f_chains.add_subplot(self.meta.nbins,2,2*k+1) for k in xrange(self.meta.nbins)]
        self.sps_pdfs = [self.f_chains.add_subplot(self.meta.nbins,2,2*(k+1)) for k in xrange(self.meta.nbins)]
        self.randwalks = xrange(self.meta.nwalkers)#random.sample(xrange(self.meta.nwalkers),1)#self.ncolors)

        for k in xrange(self.meta.nbins):
            sps_chain = self.sps_chains[k]
            sps_chain.plot([0],[0],color = 'k',label = 'Mean Sample Value',rasterized = True)
            sps_chain.set_ylim(-m.log(self.meta.ngals), m.log(self.meta.ngals / self.meta.bindif)+1)
            sps_chain.set_xlabel('iteration number')
            sps_chain.set_ylabel(r'$\ln N_{'+str(k+1)+r'}(z)$')
            sps_chain.set_title(r'$\ln N(z)$ Parameter {} of {}'.format(k, self.meta.nbins))
            self.sps_pdfs[k].set_ylim(0.,1.)
            #self.sps_pdfs[k].semilogy()
            sps_pdf = self.sps_pdfs[k]
            sps_pdf.set_xlabel(r'$\theta_{'+str(k+1)+r'}$')
            sps_pdf.set_ylabel('kernel density estimate')
            sps_pdf.set_title(r'Distribution of $\theta_{'+str(k+1)+r'}$ Values')
            sps_pdf.vlines(self.meta.logstack[k],0.,1.,linestyle='--',label='Stacked value')
            sps_pdf.vlines(self.meta.logmapNz[k],0.,1.,linestyle='-.',label='MAP value')
#             sps_pdf.vlines(self.meta.logexpNz[k],0.,1.,linestyle=':',label=r'$E(z)$ value')
            sps_pdf.vlines(self.meta.logmleNz[k],0.,1.,linestyle=':',label='MLE value')
            if self.meta.logtrueNz is not None:
                sps_pdf.vlines(self.meta.logtrueNz[k],0.,1.,linestyle='-',label='True value')

    def plot(self,key):

        data = key.load_state(self.meta.topdir)['chains']

        plot_y_c = np.swapaxes(data,0,1).T

        randsteps = xrange(self.meta.ntimes)#random.sample(xrange(self.meta.ntimes),self.meta.ncolors)

        for k in xrange(self.meta.nbins):
            mean = np.sum(plot_y_c[k])/(self.meta.ntimes*self.meta.nwalkers)
            self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                    [mean]*self.meta.ntimes,
                                    color = 'k',
                                    rasterized = True)

            x_kde = plot_y_c[k].flatten()[:, np.newaxis]
            kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=1.0).fit(x_kde)
            x_plot = np.arange(np.min(plot_y_c[k]),np.max(plot_y_c[k]),0.1)[:, np.newaxis]
            log_dens = kde.score_samples(x_plot)
            #print(sum(np.exp(log_dens)))
            self.sps_pdfs[k].plot(x_plot[:, 0],np.exp(log_dens),color=self.meta.colors[key.r%self.ncolors],rasterized=True)
            for x in randsteps:
                for w in self.randwalks:
                    self.sps_chains[k].plot(np.arange(key.r*self.meta.ntimes,(key.r+1)*self.meta.ntimes)*self.meta.thinto,#i_run.eachtimenos[r],
                                            plot_y_c[k][w],
                                            color = self.meta.colors[w%self.ncolors],
                                            alpha = self.a_chain,
                                            rasterized = True)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as statboth:
              both = cpkl.load(statboth)

        for k in xrange(self.meta.nbins):
            self.sps_pdfs[k].vlines(both['mapvals'][-1][k],0.,1.,linewidth=2,color=self.meta.colors[key.r%self.ncolors])

        self.f_chains.savefig(os.path.join(self.meta.topdir,'chains.png'),dpi=100)

        timesaver(self.meta,'chains',key)

    def plotone(self,subplot,plot_x,plot_y,style,ylabel,a=1.):
        subplot.plot(plot_x,
                     plot_y,
                     color='k',
                     linewidth=2,
                     linestyle=style,
                     alpha=a,
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
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logstack[k]]*maxsteps,'--','Stacked value')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logmapNz[k]]*maxsteps,'-.','MAP value')
#             self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logexpNz[k]]*maxsteps,':',r'$E(z)$ value')
            self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logmleNz[k]]*maxsteps,':','MLE value')

            if self.meta.logtrueNz is not None:
                self.plotone(sps_chain,maxiternos*self.meta.miniters,[self.meta.logtrueNz[k]]*maxsteps,'-','True value')

            self.sps_pdfs[k].legend(fontsize='xx-small',loc='upper left')
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
