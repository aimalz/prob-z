"""
plot-sim module makes plots of data generation
"""

# TO DO: split up datagen and pre-run plots

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math as m

import utilsim as us

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,style='-',col='k',lw=1,lab=' ',a=1.):
    subplot.hlines(plot,binends[:-1],
                   binends[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col,
                   alpha=a,
                   label=lab)
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=lw,
                   linestyle=style,
                   color=col,
                   alpha=a)

# make all the plots
def initial_plots(meta, test):
    global zrange
    zrange = np.arange(test.allzs[0],test.allzs[-1],1./meta.surv)
    global pranges
    pranges = test.real.pdfs(zrange)
    global prange
    prange = np.sum(pranges,axis=0)
    plot_physgen(meta,test)
    plot_true(meta,test)
    plot_liktest(meta,test)
    plot_pdfs(meta,test)
#     plot_truevmap(meta,test)
    print(meta.name+' plotted setup')

# plot the underlying P(z) and its components
def plot_physgen(meta,test):

    f = plt.figure(figsize=(5,5))
    sys.stdout.flush()
    sps = f.add_subplot(1,1,1)
    f.suptitle(meta.name+' True p(z)')
    sps.plot(zrange,prange,label=r'True $p(z)$',color='k')
    for k in us.lrange(meta.real):
        sps.plot(zrange,pranges[k],color=meta.colors[k],label='component '+str(meta.real[k][2])+'N('+str(meta.real[k][0])+','+str(meta.real[k][1])+')')
#     plotstep(sps,test.z_cont,test.phsPz,lw=3.,col='k',a=1./3.)
    #sps.semilogy()
    sps.set_ylabel(r'$p(z)$')
    sps.set_xlabel(r'$z$')
    sps.legend(fontsize='xx-small',loc='upper left')
    f.savefig(os.path.join(meta.simdir,'physPz.png'))
    return

# plot trule of true N(z) for one set of parameters and one survey size
def plot_true(meta, test):

    nrange = test.ngals*prange
    lnrange = us.safelog(nrange)
    f = plt.figure(figsize=(7.5,10))
    f.suptitle(str(test.nbins)+r' Parameter '+meta.name+' for '+str(test.ngals)+' galaxies')
    sps = f.add_subplot(2,1,1)
    sps.set_title('True $\ln N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim(np.log(1./min(test.bindifs)),np.log(test.ngals/min(test.bindifs)))#(-1.,np.log(test.ngals/min(test.meta.zdifs)))
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     plotstep(sps,test.binends,test.logtruNz,style='-',lab=r'Sampled $\ln N(z)$')
    plotstep(sps,zrange,lnrange,lw=2.,lab=r'True $\ln N(z)$')
    plotstep(sps,test.binends,test.logstkNz,style='--',lab=r'Stacked $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_stkNz)+r'$')
    plotstep(sps,test.binends,test.logmapNz,style='-.',lab=r'MAP $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_mapNz)+r'$')
#    plotstep(sps,test.binends,test.full_logexpNz,style=':',lab=r'$\ln N(E[z])$; $\ln\mathcal{L}='+str(round(test.lik_expNz))+r'$')
    plotstep(sps,test.binends,test.logmmlNz,style=':',lab=r'MMLE $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_mmlNz)+r'$')
    plotstep(sps,test.binends,test.logintNz,a=0.5,lab=r'Interim $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_intNz)+r'$')
    sps.legend(loc='lower right',fontsize='xx-small')
    sps = f.add_subplot(2,1,2)
    sps.set_title('True $N(z)$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$N(z)$')
#     sps.set_ylim(0.,test.ngals/min(test.bindifs))#(0.,test.ngals/min(test.meta.zdifs))
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     plotstep(sps,test.binends,test.truNz,style='-',lab=r'Sampled $N(z)$')
    plotstep(sps,zrange,nrange,lw=2.,lab=r'True $N(z)$')
    plotstep(sps,test.binends,test.stkNz,style='--',lab=r'Stacked $N(z)$; $KLD='+str(test.kl_stkNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsstack)))
    plotstep(sps,test.binends,test.mapNz,style='-.',lab=r'MAP $N(z)$; $KLD='+str(test.kl_mapNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsmapNz)))
#     plotstep(sps,test.binends,test.expNz,style=':',lab=r'$N(E[z])$; $KLD='+str(test.kl_expNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsexpNz)))
    plotstep(sps,test.binends,test.mmlNz,style=':',lab=r'MMLE $N(z)$; $KLD='+str(test.kl_mmlNz)+r'$')
    plotstep(sps,test.binends,test.intNz,a=0.5,lab=r'Interim $N(z)$; $KLD='+str(test.kl_intNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsinterim)))
    sps.legend(loc='upper left',fontsize='xx-small')
    f.savefig(os.path.join(meta.simdir,'trueNz.png'))
    return

# plot some individual posteriors
def plot_pdfs(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    f.suptitle('Observed galaxy posteriors for '+meta.name)
    sps.plot([-1.],[-1.],color='k',linestyle='-.',label=r'True $z$')
    sps.plot([-1.],[-1.],color='k',linestyle=':',label=r'$E(z)$')
    sps.legend(loc='upper left',fontsize='xx-small')
    #sps.set_title('multimodal='+str(meta.shape)+', noisy='+str(meta.noise))
    for r in us.lrange(test.randos):
        plotstep(sps,test.binends,test.pdfs[test.randos[r]],col=meta.colors[r])#,alpha=a)
        sps.vlines(test.truZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle='-.')
        sps.vlines(test.obsZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle=':')
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
    a_point = 10./test.ngals
    a_bar = float(len(meta.colors))*a_point
    #randos = random.sample(pdfs[-1][0],ncolors)
    sps.set_ylabel('Point Estimate')
    sps.set_xlabel(r'True $z$')
    sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    sps.set_ylim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
    sps.plot(test.binmids,test.binmids,c='k')
    plotpdfs = (test.pdfs*meta.zdif)
    plotpdfs = plotpdfs/np.max(plotpdfs)
    for k in xrange(test.nbins):
        for j in xrange(test.ngals):
            sps.vlines(test.truZs[j], test.binlos[k], test.binhis[k],
                        color=meta.colors[2],
                        alpha=plotpdfs[j][k]*a_bar,
                        linewidth=2)
    sps.scatter(test.truZs, test.mapZs,
                c=meta.colors[0],
                alpha = a_point,
                label=r'MAP $z$',
                linewidth=0.1)
#     sps.scatter(test.truZs, test.expZs,
#                 c=meta.colors[1],
#                 alpha = a_bar,
#                 label=r'$E(z)$',
#                 linewidth=0.1)
    sps.plot([-1],[-1],c=meta.colors[2],alpha=0.5,linewidth=2,label=r'$p(z|\vec{d})$')
    sps.legend(loc='upper left',fontsize='x-small')
    f.savefig(os.path.join(meta.simdir,'truevmap.png'))
    return

def plot_liktest(meta,test):
    f = plt.figure(figsize=(15,7.5))
    sps = [f.add_subplot(1,2,x+1) for x in xrange(0,2)]
    f.suptitle(meta.name+' Likelihood Test')
    sps[0].set_ylabel('Log Likelihood')
    sps[1].set_ylabel(r'$\ln N(z)$')
    sps[0].set_xlabel(r'Fraction $\ln\tilde{\vec{\theta}}; (1 -$ Fraction $\ln\vec{\theta}^{0})$')
    sps[1].set_xlabel(r'$z$')

    plotstep(sps[1],test.binends,test.mmlNz,lw=3,style=':',lab=r'MMLE $N(z)$')
    plotstep(sps[1],test.binends,test.truNz,lw=3,style='--',lab=r'True $N(z)$')
    plotstep(sps[1],test.binends,test.intNz,lw=3,style='-.',lab=r'Interim Prior $N(z)$')

    frac_t = np.arange(0.,2.+0.2,0.2)
    frac_i = 1.1*(1.-frac_t)

    for i in xrange(0,len(frac_t)):
        logmix = test.logtruNz*frac_t[i]+test.logintNz*frac_i[i]
        mix = np.exp(logmix)
#         mix = test.truNz*frac_t[i]+test.intNz*frac_i[i]
#         logmix = us.safelog(mix)
        index = frac_t[i]/(frac_i[i]+frac_t[i])
        sps[0].scatter(index,test.calclike(logmix))
        plotstep(sps[1],test.binends,mix,lab=str(frac_t[i])+r'\ True $\ln N(z)$, '+str(frac_i[i])+r'\ Interim Prior')
    avgNz = test.truNz*0.5+test.intNz*0.5
    logavgNz = us.safelog(avgNz)#test.logtruNz*0.5+test.logstkNz*0.5
    sps[0].hlines([test.lik_mmlNz]*len(frac_t),0.,2.,linewidth=1.,linestyle=':',label=r'MMLE $\ln N(z)$')
    sps[0].hlines([test.lik_truNz]*len(frac_t),0.,2.,linewidth=1.,linestyle='--',label=r'True $\ln N(z)$')
    sps[0].hlines([test.lik_intNz]*len(frac_t),0.,2.,linewidth=1.,linestyle='-.',label=r'Interim Prior $\ln N(z)$')
    sps[0].hlines([test.calclike(logavgNz)]*len(frac_t),0.,2.,linewidth=1.,linestyle='-',label=r'50-50 Mix of True and Interim Pror $\ln N(z)$')
    sps[0].legend(fontsize='xx-small',loc='lower right')
    sps[1].legend(fontsize='xx-small',loc='lower right')
    f.savefig(os.path.join(meta.simdir,'liktest.png'))
