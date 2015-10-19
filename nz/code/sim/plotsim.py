"""
plot-sim module makes plots of data generation
"""

# TO DO: split up datagen and pre-run plots

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import random
import math as m
from utilsim import *

# make all the plots
def initial_plots(meta, test):
    plot_priorgen(meta)
    plot_true(meta,test)
    plot_pdfs(meta,test)
    plot_truevmap(meta,test)
    print('plotted setup')

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
    sps.set_ylabel(r'$p(z)$')
    sps.set_xlabel(r'$z$')
    sps.legend(fontsize='x-small',loc='upper left')
    f.savefig(os.path.join(meta.simdir,'physPz.png'))
    return

# plot sample of true N(z) for one set of parameters and one survey size
def plot_true(meta, test):

    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title(str(meta.params)+r' Parameter True $N(z)$ for '+str(test.seed)+' galaxies')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(-1.,m.log(test.seed/meta.zdif)+1.)
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    sps.step(test.zmids,test.logsampNz,
             #color=meta.colors[n_run.n%6],
             label=r'true $\ln N(z)$',#+str(n_run.n+1),
             where='mid')#,alpha=0.1)
    sps.step(test.binmids,test.logstack,label=r'Stacked $\ln N(z)$',where='mid')
    sps.step(test.binmids,test.logmapNz,label=r'MAP $\ln N(z)$',where='mid')
    sps.step(test.binmids,test.logexpNz,label=r'$E(z)\ \ln N(z)$',where='mid')
    sps.step(test.zmids,test.logflatNz,label=r'flat $\ln N(z)$',where='mid')
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.simdir,'trueNz.png'))
    return

# plot some individual posteriors
def plot_pdfs(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Observed galaxy posteriors')
    sps.set_title('shape='+str(meta.shape)+', noise='+str(meta.noise))
    randos = random.sample(xrange(test.ngals),len(meta.colors))#n_run.ngals
    for r in randos:
        sps.step(test.binmids,test.pobs[r],where='mid',color=meta.colors[r%len(meta.colors)])#,alpha=a)
        sps.vlines(test.trueZs[r],0.,max(test.pobs[r]),color=meta.colors[r%len(meta.colors)],linestyle='--')
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(test.binlos[0]-meta.zdif,test.binhis[-1]+meta.zdif)
    sps.set_ylim(0.,1./meta.zdif)
    f.savefig(os.path.join(meta.simdir,'samplepzs.png'))
    return

# plot true vs. MAP redshifts
def plot_truevmap(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('True Redshifts vs. Point Estimates')
    #print('plot_lfs')
    #randos = random.sample(pobs[-1][0],ncolors)
    sps.set_ylabel('Point Estimate')
    sps.set_xlabel(r'True $z$')
    sps.set_xlim(meta.allzlos[0]-meta.zdif,meta.allzhis[test.ndims-1]+meta.zdif)
    sps.set_ylim(meta.allzlos[0]-meta.zdif,meta.allzhis[test.ndims-1]+meta.zdif)
    sps.plot(meta.allzmids,meta.allzmids,c='k')
    sps.scatter(test.trueZs, test.mapzs,
                c=meta.colors[0],
                alpha = 0.5,
                label=r'MAP $z$',
                linewidth=0.1)
    sps.scatter(test.trueZs, test.expzs,
                c=meta.colors[1],
                alpha = 0.5,
                label=r'$E(z)$',
                linewidth=0.1)
    plotpobs = test.pobs*meta.zdif
    for x in lrange(test.zmids):
        for y in xrange(test.ngals):
            sps.scatter(test.trueZs[y], test.binmids[x],
                        c=meta.colors[2],
                        alpha=plotpobs[y][x],
                        linewidth=0.1)
    sps.scatter([-1],[-1],c=meta.colors[2],alpha=0.5,linewidth=0.1,label=r'$p(z|\vec{d})$')
    for x in lrange(test.zmids):
        for y in xrange(test.ngals):
            sps.scatter(test.trueZs[y], test.binmids[x],
                        c=meta.colors[2],
                        alpha=plotpobs[y][x],
                        linewidth=0.1)
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.simdir,'truevmap.png'))
    return
