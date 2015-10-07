"""
iplots module makes quick plots that don't need multiprocessing before MCMC
"""

# TO DO: split up datagen and pre-run plots

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import timeit
import random
import math as m
from util import *
from key import key

# make all the plots
def initial_plots(meta, runs):
    plot_priorgen(meta)
    print ("runs = " + str(runs))
    for p in lrange(meta.params):
        pkey = key(p=p)
        p_runs=runs['p_runs']
        print ('p_runs = ' + str(p_runs))
        p_run = p_runs[pkey]
        for s in lrange(meta.survs):
            skey = pkey.add(s=s)
            s_run = runs['s_runs'][skey]
            survinfo = (meta, p_run, s_run)
            plot_true_tup = plot_true_setup(*survinfo)
            plot_truevmap_tup = plot_truevmap_setup(*survinfo)
            plot_pdfs_tup = plot_pdfs_setup(*survinfo)
            for n in xrange(meta.samps):
                nkey = skey.add(n=n)
                n_run = runs['n_runs'][nkey]
                sampinfo = survinfo + (n_run,)
                plot_true_tup = plot_true(plot_true_tup, *sampinfo)
                plot_truevmap_tup = plot_truevmap(plot_truevmap_tup, *sampinfo)
                ivals = plot_ivals_setup(*sampinfo)
                plot_priorsamps(*sampinfo)
                plot_pdfs_tup = plot_pdfs(plot_pdfs_tup,*sampinfo)
                for i in lrange(meta.inits):
                    ikey = nkey.add(i=i)
                    i_run = runs['i_runs'][ikey]
                    initinfo = sampinfo + (i_run,)
                    ivals = plot_ivals(ivals, *initinfo)
                plot_ivals_wrapup(ivals, *sampinfo)
            plot_true_wrapup(plot_true_tup, *survinfo)
            plot_truevmap_wrapup(plot_truevmap_tup, *survinfo)
            plot_pdfs_wrapup(plot_pdfs_tup,*survinfo)
    print('initial plots completed')
    with open(meta.plottime,'w') as plottimer:
            plottimer.write(str(timeit.default_timer())+' iplots \n')
            plottimer.close()

# plot the underlying P(z) and its components
def plot_priorgen(metainfo):

    meta = metainfo

    realsum = sum(meta.realistic)
    realistic_pdf = meta.realistic/meta.zdifs/realsum
    plotrealistic = np.array([sum(r) for r in meta.realistic_comps])
    plotrealisticsum = sum(plotrealistic)
    plotrealistic_comps = np.transpose(meta.realistic_comps/plotrealisticsum)
    plotrealistic_pdf = plotrealistic/plotrealisticsum

    f = plt.figure(figsize=(5,5))
    sys.stdout.flush()
    sps = f.add_subplot(1,1,1)
    f.suptitle('True p(z)')
    sps.step(meta.allzmids,plotrealistic_pdf,c='k',label='True p(z)')
    for k in range(0,len(meta.real)):
        sps.step(meta.allzmids,plotrealistic_comps[k],c=meta.colors[k],label='component '+str(meta.real[k][2])+'N('+str(meta.real[k][0])+','+str(meta.real[k][1])+')')
    sps.set_ylabel('p(z)')
    sps.set_xlabel('z')
    sps.legend(fontsize='x-small',loc='upper left')
    f.savefig(os.path.join(meta.topdir,'physPz.png'))
    return

#plot all samples of true N(z) for one set of parameters and one survey size
def plot_true_setup(meta, p_run, s_run):

    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title(r''+str(meta.params[p_run.p])+' Parameter True $N(z)$ for '+str(s_run.seed)+' galaxies')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(-1.,m.log(s_run.seed/meta.zdif)+1.)
    sps.set_xlim(meta.allzlos[0]-meta.zdif,meta.allzhis[p_run.ndims-1]+meta.zdif)
    sps.step(p_run.zmids,s_run.logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
    return((f,sps))

def plot_true((f,sps),meta,p_run,s_run,n_run):

    sps.step(p_run.zmids,n_run.logsampNz,
             color=meta.colors[n_run.n%6],
             label=r'true $\ln N(z)$ '+str(n_run.n+1),
             where='mid')#,alpha=0.1)
    return(f,sps)

def plot_true_wrapup((f,sps),meta,p_run,s_run):

    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(s_run.get_dir(),'trueNz.png'))
    return

# plot some individual posteriors
def plot_pdfs_setup(meta,p_run,s_run):
    f = plt.figure(figsize=(5*meta.samps,5))
    sps = [f.add_subplot(1, meta.samps, x+1) for x in xrange(meta.samps)]
    f.suptitle('Observed galaxy posteriors')#\n multimodal='+str(meta.shape[n])+', noise='+str(meta.noise[n]))
    return(f,sps)

def plot_pdfs((f,sps),meta,p_run,s_run,n_run):
    sps[n_run.n].set_title('multimodal='+str(meta.shape[n_run.n])+', noise='+str(meta.noise[n_run.n]))
    randos = random.sample(xrange(n_run.ngals),len(meta.colors))#n_run.ngals
    for r in lrange(randos):
        sps[n_run.n].step(n_run.binmids,n_run.pobs[randos[r]],where='mid',color=meta.colors[r])#,alpha=a)
        sps[n_run.n].vlines(n_run.trueZs[randos[r]],0.,max(n_run.pobs[randos[r]]),color=meta.colors[r],linestyle='--')
    sps[n_run.n].set_ylabel(r'$p(z|\vec{d})$')
    sps[n_run.n].set_xlabel(r'$z$')
    sps[n_run.n].set_xlim(n_run.binlos[0]-meta.zdif,n_run.binhis[-1]+meta.zdif)
    sps[n_run.n].set_ylim(0.,1./meta.zdif)
    return(f,sps)

def plot_pdfs_wrapup((f,sps),meta,p_run,s_run):
    f.savefig(os.path.join(s_run.get_dir(),'samplepzs.png'))

# plot true vs. MAP redshifts
def plot_truevmap_setup(meta, p_run, s_run):

    global a_tvm
    a_tvm = 1./meta.samps
    f = plt.figure(figsize=(5*len(meta.survs),5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('True Redshifts vs. MAP Redshifts')
    #print('plot_lfs')
    #randos = random.sample(pobs[-1][0],ncolors)
    sps.set_ylabel(r'Observed $z$')
    sps.set_xlabel(r'True $z$')
    sps.set_xlim(meta.allzlos[0]-meta.zdif,meta.allzhis[p_run.ndims-1]+meta.zdif)
    sps.set_ylim(meta.allzlos[0]-meta.zdif,meta.allzhis[p_run.ndims-1]+meta.zdif)
    return((f,sps))

def plot_truevmap((f,sps),meta,p_run,s_run,n_run):
    sps.scatter(n_run.trueZs, n_run.mapzs,
                alpha=a_tvm,
                c=meta.colors[n_run.n],
                label='shape='+str(meta.shape[n_run.n])+', noise='+str(meta.noise[n_run.n]),
                linewidth=0.1)
    return((f,sps))

def plot_truevmap_wrapup((f,sps),meta,p_run,s_run):
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(s_run.get_dir(),'truevmap.png'))
    return

# plot some samples from prior for one instantiation of survey
def plot_priorsamps(meta,p_run,s_run,n_run):

    priorsamps = np.array(n_run.priordist.sample_ps(len(meta.colors))[0])

    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)

    sps.set_title(r'Prior samples for $J_{0}='+str(s_run.seed)+r'$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_xlim(n_run.binends[0]-meta.zdif,n_run.binends[-1]+meta.zdif)#,s_run.seed)#max(n_run.full_logflatNz)+m.log(s_run.seed/meta.zdif)))
    sps.step(n_run.binmids,n_run.full_logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
    for c in lrange(meta.colors):
        sps.step(n_run.binmids,priorsamps[c],color=meta.colors[c],where='mid')
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(n_run.get_dir(), 'priorsamps.png'))
    return

# plot initial values for all initialization procedures
def plot_ivals_setup(meta,p_run,s_run,n_run):

    f = plt.figure(figsize=(5*len(meta.inits), 5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
    sps = [f.add_subplot(1, len(meta.inits), x+1) for x in lrange(meta.inits)]

    f.suptitle('Initialization of '+str(n_run.nwalkers)+' walkers '+str(n_run.n+1))
    return((f,sps))

def plot_ivals((f,sps),meta,p_run,s_run,n_run,i_run):

    sps[i_run.i].set_ylabel(r'$\ln N(z)$ with $J_{0}='+str(s_run.seed)+'$')
    sps[i_run.i].set_xlabel(r'$z$')
    sps[i_run.i].set_title(meta.init_names[i_run.i])

    for iguess in i_run.iguesses:
        sps[i_run.i].step(n_run.binlos,iguess,
                          where='post',
                          alpha=0.5,)
    sps[i_run.i].step(n_run.binmids,i_run.mean,color='k',linewidth=2,where='mid')
    return((f,sps))

def plot_ivals_wrapup((f,sps),meta,p_run,s_run,n_run):

    f.savefig(os.path.join(n_run.get_dir(),'initializations.png'),dpi=100)
    return
