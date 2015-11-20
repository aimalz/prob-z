"""
plot-sim module makes plots of data generation
"""

# TO DO: split up datagen and pre-run plots

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math as m

from utilsim import *

np.random.seed(seed=0)

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,style='-',col='k',lw=1,lab=' '):
    subplot.hlines(plot,binends[:-1],
                   binends[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col,
                   label=lab)
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col)

# make all the plots
def initial_plots(meta, test):
    plot_priorgen(meta)
    plot_true(meta,test)
    plot_pdfs(meta,test)
    plot_truevmap(meta,test)
    print(meta.name+' plotted setup')

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
    f.suptitle(meta.name+' True p(z)')
    sps.semilogy()
    sps.set_ylim(0.01,1.0)
    plotstep(sps,meta.allzs,plotrealistic_pdf,lab=r'True $p(z)$')
    for k in range(0,len(meta.real)):
        plotstep(sps,meta.allzs,plotrealistic_comps[k],col=meta.colors[k],lab='component '+str(meta.real[k][2])+'N('+str(meta.real[k][0])+','+str(meta.real[k][1])+')')
    sps.set_ylabel(r'$p(z)$')
    sps.set_xlabel(r'$z$')
    sps.legend(fontsize='x-small',loc='upper left')
    f.savefig(os.path.join(meta.simdir,'physPz.png'))
    return

# plot sample of true N(z) for one set of parameters and one survey size
def plot_true(meta, test):

    plotrealistic_Nz = test.ngals*meta.realistic/sum(meta.realistic*meta.zdifs)
    plotrealistic_logNz = np.log(plotrealistic_Nz)

    f = plt.figure(figsize=(5,10))
    f.suptitle(str(test.nbins)+r' Parameter '+meta.name+' for '+str(test.ngals)+' galaxies')
    sps = f.add_subplot(2,1,1)
    sps.set_title('True $\ln N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(-1.,m.log(test.seed/meta.zdif))
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    plotstep(sps,test.allzs,plotrealistic_logNz,col='b',lab=r'underlying $\ln N(z)$')
    plotstep(sps,test.allzs,test.logsampNz,style='-',lab=r'true $\ln N(z)$')
    plotstep(sps,test.binends,test.logstack,style='--',lab=r'Stacked $\ln N(z)$ with $\sigma^{2}=$'+str(int(test.vslogstack)))
    plotstep(sps,test.binends,test.logmapNz,style='-.',lab=r'MAP $\ln N(z)$ with $\sigma^{2}=$'+str(int(test.vslogmapNz)))
    plotstep(sps,test.binends,test.logexpNz,style=':',lab=r'$\ln N(E[z])$ with $\sigma^{2}=$'+str(int(test.vslogexpNz)))
    plotstep(sps,test.binends,test.full_loginterim,col='r',lab=r'interim $\ln N(z)$ with $\sigma^{2}=$'+str(int(test.vsloginterim)))
    sps.legend(loc='lower right',fontsize='xx-small')
    sps = f.add_subplot(2,1,2)
    sps.set_title('True $N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$N(z)$')
    sps.set_ylim(0.,test.seed/meta.zdif)
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    plotstep(sps,test.allzs,plotrealistic_Nz,col='b',lab=r'underlying $N(z)$')
    plotstep(sps,test.allzs,test.sampNz,style='-',lab=r'true $N(z)$')
    plotstep(sps,test.binends,test.stack,style='--',lab=r'Stacked $N(z)$ with $\sigma^{2}=$'+str(int(test.vsstack)))
    plotstep(sps,test.binends,test.mapNz,style='-.',lab=r'MAP $N(z)$ with $\sigma^{2}=$'+str(int(test.vsmapNz)))
    plotstep(sps,test.binends,test.expNz,style=':',lab=r'$N(E[z])$ with $\sigma^{2}=$'+str(int(test.vsexpNz)))
    plotstep(sps,test.binends,test.full_interim,col='r',lab=r'interim $N(z)$ with $\sigma^{2}=$'+str(int(test.vsinterim)))
#     sps.step(test.zhis,plotrealistic_Nz,label=r'underlying $N(z)$')
#     sps.step(test.zhis,test.sampNz,label=r'true $N(z)$')
#     sps.step(test.binhis,test.stack,label=r'Stacked $N(z)$ with $\sigma^{2}=$'+str(int(test.vsstack)),linestyle='--')
#     sps.step(test.binhis,test.mapNz,label=r'MAP $N(z)$ with $\sigma^{2}=$'+str(int(test.vsmapNz)),linestyle='-.')
#     sps.step(test.binhis,test.expNz,label=r'$N(E[z])$ with $\sigma^{2}=$'+str(int(test.vsexpNz)),linestyle=':')
#     sps.step(test.binhis,test.full_interim,label=r'interim $N(z)$ with $\sigma^{2}=$'+str(int(test.vsinterim)))
    sps.legend(loc='upper left',fontsize='xx-small')
    f.savefig(os.path.join(meta.simdir,'trueNz.png'))
    return

# plot some individual posteriors
def plot_pdfs(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Observed galaxy posteriors for '+meta.name)
    #sps.set_title('multimodal='+str(meta.shape)+', noisy='+str(meta.noise))
    randos = np.random.randint(0,test.ngals,len(meta.colors))
    for r in lrange(randos):
        plotstep(sps,test.binends,test.pobs[randos[r]],col=meta.colors[r])#,alpha=a)
        sps.vlines(test.trueZs[randos[r]],0.,max(test.pobs[randos[r]]),color=meta.colors[r],linestyle='--')
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(test.binlos[0]-meta.zdif,test.binhis[-1]+meta.zdif)
    sps.set_ylim(0.,1./meta.zdif)
    f.savefig(os.path.join(meta.simdir,'samplepzs.png'))
    return

# plot true vs. MAP vs E(z) redshifts
def plot_truevmap(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle(meta.name+' True Redshifts vs. Point Estimates')
    a = 1./np.log10(test.ngals)**2
    #randos = random.sample(pobs[-1][0],ncolors)
    sps.set_ylabel('Point Estimate')
    sps.set_xlabel(r'True $z$')
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    sps.set_ylim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    sps.plot(test.binmids,test.binmids,c='k')
    plotpobs = (test.pobs*meta.zdif)
    plotpobs = plotpobs/np.max(plotpobs)
    for k in xrange(test.nbins):
        for j in xrange(test.ngals):
            sps.vlines(test.trueZs[j], test.binlos[k], test.binhis[k],
                        color=meta.colors[2],
                        alpha=plotpobs[j][k]*a,
                        linewidth=2)
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
    sps.plot([-1],[-1],c=meta.colors[2],alpha=0.5,linewidth=2,label=r'$p(z|\vec{d})$')
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.simdir,'truevmap.png'))
    return
