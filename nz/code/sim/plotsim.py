"""
plot-sim module makes plots of data generation
"""

# TO DO: split up datagen and pre-run plots

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys
import os
import math as m
import random

import utilsim as us

title = 10
label = 10
mpl.rcParams['axes.titlesize'] = title
mpl.rcParams['axes.labelsize'] = label
#print('set font sizes')
mpl.rcParams['figure.subplot.left'] = 0.2
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.bottom'] = 0.2
mpl.rcParams['figure.subplot.top'] = 0.9
#print('set figure sizes')
mpl.rcParams['figure.subplot.wspace'] = 0.5
mpl.rcParams['figure.subplot.hspace'] = 0.5
#print('set spaces')

cmap = np.linspace(0.,1.,4)
colors = [cm.Greys(i) for i in cmap]#"gnuplot" works well

# global s_tru,w_tru,a_tru,c_tru,d_tru,l_tru
# s_tru,w_tru,a_tru,c_tru,d_tru,l_tru = '--',1.,1.,'k',[(0,(1,0.0001))],r' True '
# global s_int,w_int,a_int,c_int,d_int,l_int
# s_int,w_int,a_int,c_int,d_int,l_int = '--',1.,0.5,'k',[(0,(1,0.0001))],r' Interim '
# global s_stk,w_stk,a_stk,c_stk,d_stk,l_stk
# s_stk,w_stk,a_stk,c_stk,d_stk,l_stk = '--',1.,1.,'k',[(0,(1,2))],r' Stacked '
# global s_map,w_map,a_map,c_map,d_map,l_map
# s_map,w_map,a_map,c_map,d_map,l_map = '--',1.,0.75,'k',[(0,(4,4,2,4))],r' MMAP '
# global s_exp,w_exp,a_exp,c_exp,d_exp,l_exp
# s_exp,w_exp,a_exp,c_exp,d_exp,l_exp = '--',1.,0.25,'k',[(0,(4,4,2,4))],r' MExp '
# global s_mml,w_mml,a_mml,c_mml,d_mml,l_mml
# s_mml,w_mml,a_mml,c_mml,d_mml,l_mml = '--',1.,1.,'k',[(0,(3,2))],r' MMLE '
# global s_smp,w_smp,a_smp,c_smp,d,smp,l_smp
# s_smp,w_smp,a_smp,c_smp,d_smp,l_smp = '--',1.,1.,'k',[(0,(1,0.0001))],r' Sampled '
global s_tru,w_tru,a_tru,c_tru,d_tru,l_tru
s_tru,w_tru,a_tru,c_tru,d_tru,l_tru = '--',0.5,1.,'k',[(0,(1,0.0001))],'True '
global s_int,w_int,a_int,c_int,d_int,l_int
s_int,w_int,a_int,c_int,d_int,l_int = '--',0.5,0.5,'k',[(0,(1,0.0001))],'Interim '
global s_stk,w_stk,a_stk,c_stk,d_stk,l_stk
s_stk,w_stk,a_stk,c_stk,d_stk,l_stk = '--',1.5,1.,'k',[(0,(3,2))],'Stacked '#[(0,(2,1))]
global s_map,w_map,a_map,c_map,d_map,l_map
s_map,w_map,a_map,c_map,d_map,l_map = '--',1.,1.,'k',[(0,(3,2))],'MMAP '#[(0,(1,1,3,1))]
global s_exp,w_exp,a_exp,c_exp,d_exp,l_exp
s_exp,w_exp,a_exp,c_exp,d_exp,l_exp = '--',1.,1.,'k',[(0,(1,1))],'MExp '#[(0,(3,3,1,3))]
global s_mml,w_mml,a_mml,c_mml,d_mml,l_mml
s_mml,w_mml,a_mml,c_mml,d_mml,l_mml = '--',2.,1.,'k',[(0,(1,1))],'MMLE '#[(0,(3,2))]
global s_smp,w_smp,a_smp,c_smp,d,smp,l_smp
s_smp,w_smp,a_smp,c_smp,d_smp,l_smp = '--',1.,1.,'k',[(0,(1,0.0001))],'Sampled '
global s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe
s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe = '--',2.,1.,'k',[(0,(1,0.0001))],'Mean of\n Samples '

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,s='--',d=[(0,(1,0.0001))],c='k',w=1,l=None,a=1.):
    subplot.hlines(plot,
                   binends[:-1],
                   binends[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   label=l)
    subplot.vlines(binends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a)

# def plotstep(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))],l=None):
#     ploth(subplot,binends,plot,s,c,a,w,d,l)
#     plotv(subplot,binends,plot,s,c,a,w,d)

# def ploth(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))],l=None):
#     subplot.hlines(plot,
#                    binends[:-1],
#                    binends[1:],
#                    linewidth=w,
#                    linestyle=s,
#                    dashes=d,
#                    color=c,
#                    alpha=a,
#                    label=l)
# def plotv(subplot,binends,plot,s='--',c='k',a=1,w=1,d=[(0,(1,0.0001))]):
#     subplot.vlines(binends[1:-1],
#                    plot[:-1],
#                    plot[1:],
#                    linewidth=w,
#                    linestyle=s,
#                    dashes=d,
#                    color=c,
#                    alpha=a)

# make all the plots
def initial_plots(meta, test):
    global zrange
    zrange = np.arange(test.allzs[0],test.allzs[-1],1./meta.surv)
    global pranges
    pranges = test.real.pdfs(zrange)
    global prange
    prange = np.sum(pranges,axis=0)
#     print('will plots work?')
    plot_physgen(meta,test)
    plot_ests(meta,test)
#     plot_true(meta,test)
#     plot_liktest(meta,test)
    plot_pdfs(meta,test)
#     plot_pdfs_test(meta,test)
#    plot_truevmap(meta,test)
    plot_lfs(meta,test)
#    plot_mlehist(meta,test)
    print(meta.name+' plotted setup')

# plot the underlying P(z) and its components
def plot_physgen(meta,test):

    f = plt.figure(figsize=(5,5))
    sys.stdout.flush()
    sps = f.add_subplot(1,1,1)
#     f.suptitle(meta.name+' True p(z)')
    sps.set_title(meta.name+r' True $p(z)$')
    sps.plot(zrange,prange/np.sum(prange*np.average(zrange[1:]-zrange[:-1])),color='k')
#     for k in us.lrange(meta.real):
#         sps.plot(zrange,pranges[k],color=meta.colors[k],label=str(meta.real[k][2])+r'$\mathcal{N}$('+str(meta.real[k][0])+','+str(meta.real[k][1])+')')
# #     plotstep(sps,test.z_cont,test.phsPz,lw=3.,col='k',a=1./3.)
    #sps.semilogy()
    sps.set_ylabel(r'$p(z)$')
    sps.set_xlabel(r'$z$')
#     sps.legend(fontsize='small',loc='upper right')
    f.savefig(os.path.join(meta.simdir,'physPz.pdf'),bbox_inches='tight', pad_inches = 0)
    return

# plot some individual posteriors
def plot_pdfs(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title(meta.name+r' PDFs')
    plotstep(sps,test.binends,test.intPz,c=c_int,l=l_int+r'$P(z)$',s=s_int,w=w_int,d=d_int,a=a_int)
    dummy_x,dummy_y = np.array([-1,-2,-3]),np.array([-1,-2,-3])
    plotstep(sps,dummy_x,dummy_y,c=c_tru,s=s_tru,w=w_tru,l=l_tru+r'$z$',d=d_tru,a=a_tru)
    plotstep(sps,dummy_x,dummy_y,c=c_map,s=s_map,w=w_map,l=r' MAP $z$',d=d_map,a=a_map)
    plotstep(sps,dummy_x,dummy_y,c=c_exp,s=s_exp,w=w_exp,l=r' $E(z)$',d=d_exp,a=a_exp)
    sps.legend(loc='upper right',fontsize='x-small')
    #sps.set_title('multimodal='+str(meta.shape)+', noisy='+str(meta.noise))
    #if meta.shape <= 1:
    randos = test.randos
    #print(randos)
    #else:
    #randos = np.random.choice(np.where(test.npeaks > 1)[0],len(meta.colors),replace=False)
    #print(randos)
    for r in us.lrange(randos):
        plotstep(sps,test.binends,test.pdfs[randos[r]],c=meta.colors[r],s=s_smp,w=w_smp,d=d_smp,a=a_smp)
        sps.vlines(test.gals[randos[r]].truZ,0.,max(test.pdfs[randos[r]]),color=meta.colors[r],linestyle=s_tru,linewidth=w_tru,dashes=d_tru,alpha=a_tru)
        sps.vlines(test.gals[randos[r]].mapZ,0.,max(test.pdfs[randos[r]]),color=meta.colors[r],linestyle=s_map,linewidth=w_map,dashes=d_map,alpha=a_map)
        sps.vlines(test.gals[randos[r]].expZ,0.,max(test.pdfs[randos[r]]),color=meta.colors[r],linestyle=s_exp,linewidth=w_exp,dashes=d_exp,alpha=a_exp)
#         for p in xrange(test.npeaks[randos[r]]):
#             sps.vlines(test.obsZs[randos[r]][p],0.,max(test.pdfs[randos[r]]),color=meta.colors[r],linestyle=s_map,linewidth=w_map,dashes=d_map,alpha=a_map)
#     if meta.degen != 0:
#         for x in xrange(meta.degen):
#             sps.vlines(test.mudegen[x],0.,(test.zhis[-1]-test.zlos[0])/test.bindif,color='k')
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(test.binlos[0]-meta.zdif,test.binhis[-1]+meta.zdif)
    sps.set_ylim(0.,1./meta.zdif)
    f.savefig(os.path.join(meta.simdir,'samplepzs.pdf'),bbox_inches='tight', pad_inches = 0)
    return

# # plot trule of true N(z) for one set of parameters and one survey size
# def plot_true(meta, test):

#     nrange = test.ngals*prange
#     lnrange = us.safelog(nrange)
#     f = plt.figure(figsize=(5,10))
#     f.suptitle(str(test.nbins)+r' Parameter '+meta.name+' for '+str(test.ngals)+' galaxies')
#     sps = f.add_subplot(2,1,1)
#     sps.set_title('True $\ln N(z)$')
#     sps.set_xlabel(r'binned $z$')
#     sps.set_ylabel(r'$\ln N(z)$')
#     sps.set_ylim(np.log(1./min(test.bindifs)),np.log(test.ngals/min(test.bindifs)))#(-1.,np.log(test.ngals/min(test.meta.zdifs)))
#     sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     plotstep(sps,test.binends,test.logtruNz,style='-',lab=r'Sampled $\ln N(z)$')
#     plotstep(sps,zrange,lnrange,lw=2.,lab=r'True $\ln N(z)$')
# #     plotstep(sps,test.binends,test.logstkNz,style='--',lw=3.,lab=r'Stacked $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_stkNz)+r'$')
# #     plotstep(sps,test.binends,test.logmapNz,style='-.',lw=3.,lab=r'MAP $\ln N(z)$; $\ln\mathcal{L}='+str(test.lik_mapNz)+r'$')
# #    plotstep(sps,test.binends,test.full_logexpNz,style=':',lab=r'$\ln N(E[z])$; $\ln\mathcal{L}='+str(round(test.lik_expNz))+r'$')
#     plotstep(sps,test.binends,test.logmmlNz,style=':',lw=3.,lab=r'MMLE $\ln N(z)$')#; $\ln\mathcal{L}='+str(test.lik_mmlNz)+r'$')
#     plotstep(sps,test.binends,test.logintNz,col='k',a=0.5,lw=2.,lab=r'Interim $\ln N(z)$')#; $\ln\mathcal{L}='+str(test.lik_intNz)+r'$')
#     sps.legend(loc='lower right',fontsize='x-small')
#     sps = f.add_subplot(2,1,2)
#     sps.set_title('True $N(z)$')
#     sps.set_xlabel(r'binned $z$')
#     sps.set_ylabel(r'$N(z)$')
# #     sps.set_ylim(0.,test.ngals/min(test.bindifs))#(0.,test.ngals/min(test.meta.zdifs))
#     sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     plotstep(sps,test.binends,test.truNz,style='-',lab=r'Sampled $N(z)$')
#     plotstep(sps,zrange,nrange,lw=2.,lab=r'True $N(z)$')
# #     plotstep(sps,test.binends,test.stkNz,style='--',lw=3.,lab=r'Stacked $N(z)$'+'\n'+r'$KLD='+str(test.kl_stkNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsstack)))
# #     plotstep(sps,test.binends,test.mapNz,style='-.',lw=3.,lab=r'MAP $N(z)$'+'\n'+r'$KLD='+str(test.kl_mapNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsmapNz)))
# #     plotstep(sps,test.binends,test.expNz,style=':',lab=r'$N(E[z])$'+'\n'+r'$KLD='+str(test.kl_expNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsexpNz)))
#     plotstep(sps,test.binends,test.mmlNz,style=':',lw=3.,lab=r'MMLE $N(z)$')#+'\n'+r'$KLD='+str(test.kl_mmlNz)+r'$')
#     plotstep(sps,test.binends,test.intNz,col='k',a=0.5,lw=2.,lab=r'Interim $N(z)$')#+'\n'+r'$KLD='+str(test.kl_intNz)+r'$')# with $\sigma^{2}=$'+str(int(test.vsinterim)))
#     sps.legend(loc='upper left',fontsize='x-small')

#     f.savefig(os.path.join(meta.simdir,'trueNz.pdf'),bbox_inches='tight', pad_inches = 0)
#     return

# # plot true vs. MAP vs E(z) redshifts
# def plot_truevmap(meta,test):
#     f = plt.figure(figsize=(5,5))
#     sps = f.add_subplot(1,1,1)
#     f.suptitle(meta.name+r' $p(z_{obs}|z_{tru})$')
#     a_point = 10./test.ngals
#     a_bar = float(len(meta.colors))*a_point
#     #randos = random.sample(pdfs[-1][0],ncolors)
#     sps.set_ylabel(r'$z_{obs}$')
#     sps.set_xlabel(r'$z_{tru}$')
#     sps.set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     sps.set_ylim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)
#     #sps.plot(test.binmids,test.binmids,c='k')
#     plotpdfs = (test.pdfs*meta.zdif)
#     plotpdfs = plotpdfs/np.max(plotpdfs)
#     for k in xrange(test.nbins):
#         for j in xrange(test.ngals):
#             sps.vlines(test.truZs[j], test.binlos[k], test.binhis[k],
#                         color=meta.colors[2],
#                         alpha=plotpdfs[j][k]*a_bar,
#                         linewidth=2,rasterized=True)
# #     sps.scatter(test.truZs, test.mapZs,
# #                 c=meta.colors[0],
# #                 alpha = a_point,
# #                 label=r'MAP $z$',
# #                 linewidth=0.1,rasterized=True)
# #     sps.scatter(test.truZs, test.expZs,
# #                 c=meta.colors[1],
# #                 alpha = a_bar,
# #                 label=r'$E(z)$',
# #                 linewidth=0.1)
#     sps.plot([-1],[-1],c=meta.colors[2],alpha=0.5,linewidth=2,label=r'$p(z|\vec{d})$',rasterized=True)
#     sps.legend(loc='upper left',fontsize='small')
#     f.savefig(os.path.join(meta.simdir,'zobsvztru.pdf'),bbox_inches='tight', pad_inches = 0)
#     return

# def plot_liktest(meta,test):

#     frac_t = np.arange(0.,2.+0.2,0.2)
#     frac_i = 1.1*(1.-frac_t)
#     avgNz = test.truNz*0.5+test.intNz*0.5
#     logavgNz = us.safelog(avgNz)#test.logtruNz*0.5+test.logstkNz*0.5

#     f = plt.figure(figsize=(10,5))
#     sps = [f.add_subplot(1,2,x+1) for x in xrange(0,2)]
#     f.suptitle(meta.name+' Likelihood Test')
#     sps[0].set_ylabel('Log Likelihood')
#     sps[1].set_ylabel(r'$\ln N(z)$')
#     sps[0].set_xlabel(r'Fraction $\ln\tilde{\vec{\theta}}; (1 -$ Fraction $\ln\vec{\theta}^{0})$')
#     sps[1].set_xlabel(r'$z$')
#     sps[1].set_xlim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)

#     sps[0].hlines([test.lik_truNz]*len(frac_t),0.,2.,linewidth=2.,linestyle='--',label=r'True $\ln N(z)$',rasterized=True)
#     sps[0].hlines([test.lik_intNz]*len(frac_t),0.,2.,linewidth=2.,linestyle='-.',label=r'Interim Prior $\ln N(z)$',rasterized=True)
#     sps[0].hlines([test.calclike(logavgNz)]*len(frac_t),0.,2.,linewidth=2.,linestyle='-',label=r'50-50 Mix',rasterized=True)
#     sps[0].hlines([test.lik_mmlNz]*len(frac_t),0.,2.,linewidth=2.,linestyle=':',label=r'MMLE $\ln N(z)$',rasterized=True)

#     plotstep(sps[1],test.binends,test.mmlNz,lw=3,style=':',lab=r'MMLE $N(z)$',rasterized=True)
#     plotstep(sps[1],test.binends,test.truNz,lw=3,style='--',lab=r'True $N(z)$',rasterized=True)
#     plotstep(sps[1],test.binends,test.intNz,lw=3,style='-.',lab=r'Interim Prior $N(z)$',rasterized=True)

#     sps[0].legend(fontsize='small',loc='lower right')
#     sps[1].legend(fontsize='small',loc='lower right')

#     for i in xrange(0,len(frac_t)):
#         logmix = test.logtruNz*frac_t[i]+test.logintNz*frac_i[i]
#         mix = np.exp(logmix)
# #         mix = test.truNz*frac_t[i]+test.intNz*frac_i[i]
# #         logmix = us.safelog(mix)
#         index = frac_t[i]/(frac_i[i]+frac_t[i])
#         sps[0].scatter(index,test.calclike(logmix),rasterized=True)
#         plotstep(sps[1],test.binends,mix,lab=str(frac_t[i])+r'\ True $\ln N(z)$, '+str(frac_i[i])+r'\ Interim Prior')

#     f.savefig(os.path.join(meta.simdir,'liktest.png'),bbox_inches='tight', pad_inches = 0)

def makepdfs(meta,test,j,tru,zfactors,grid):
    gal = test.gals[j]
    dgen = gal.distdgen

    #shift = test.shift[j]

#     if meta.degen != 0:
#         dgen = []
#         for x in xrange(meta.degen):
#             weight = (1./test.sigdegen[x])/sum(1./test.sigdegen)
#             dgen.append([test.mudegen[x],test.sigdegen[x],weight])
#         dgen = us.gmix(dgen,(min(grid),max(grid)))
#     else:
#         dgen = None

#     if meta.outlier != 0:
#         outlier = test.peaklocs
#     else:
#         outlier = None

    #sigZ = test.varZs[j]*zfactor
    #obsZ = test.obsZs[j]
    #allelems = gal.allelems
    #print('outlier={}; obsZ={}'.format(outlier, obsZ))
    pdf = us.makepdf(grid,tru,gal,
                     intp=None,dgen=dgen,outlier=gal.outelems)#(grid,tru,shift,obsZ,sigZ,intp=None,dgen=dgen,outlier=outlier)

#     difs = grid[1:]-grid[:-1]
#     allsummed = np.zeros(len(grid)-1)

#     for pn in xrange(test.npeaks[j]):
#         func = us.tnorm(tru,test.varZs[j][pn]*zfactor,(min(grid),max(grid)))
#         cdf = np.array([func.cdf(binend) for binend in grid])
#         spread = cdf[1:]-cdf[:-1]

#         allsummed += spread/(test.npeaks[j]+meta.degen)

#     if meta.degen != 0:
#         for x in xrange(meta.degen):
#             funcs = us.tnorm(test.mudegen[x][0],test.sigdegen[x][0],(min(grid),max(grid)))
#                 #pdfs = funcs.pdf(self.truZs[j])
#             cdfs = np.array([funcs.cdf(binend) for binend in grid])
#             spreads = cdfs[1:]-cdfs[:-1]

#             allsummed += spreads/(test.npeaks[j]+meta.degen)

#     if meta.shape > 1:

#         funcs = [[tru,test.varZs[j][0]*zfactor,1.]]
#         for n in xrange(test.npeaks[j]-1):
#             if meta.outlier == 1:
#                 funcs.append([test.peaklocs[n],test.peakvars[n],1.])
#             else:
#                 funcs.append([test.obsZs[j][n],test.sigZs[j][n],1.])
#         func = us.gmix(funcs,(min(grid),max(grid)))
#         cdfs = func.cdf(grid)
#         spreads = cdfs[1:]-cdfs[:-1]

#         allsummed += spreads/(test.npeaks[j]+meta.degen)

#     # normalize probabilities to integrate (not sum)) to 1
#     pdf = allsummed/max(np.dot(allsummed,difs),sys.float_info.epsilon)

    return(pdf)

def make2dlf(meta,test,j):

    eps = 1./100.
    zrange = test.zhis[-1]-test.zlos[0]
    ztrugrid = np.arange(test.zlos[0],test.zhis[-1]+eps,eps)
    ztrugrid = ztrugrid[ztrugrid!=0.]
    zobsgrid = np.arange(test.zlos[0],test.zhis[-1]+eps,eps)
    zobsgrid = zobsgrid[zobsgrid!=0.]
    trugridmids = (ztrugrid[1:]+ztrugrid[:-1])/2.
    obsgridmids = (zobsgrid[1:]+zobsgrid[:-1])/2.
    trugriddifs = ztrugrid[1:]-ztrugrid[:-1]
    obsgriddifs = zobsgrid[1:]-zobsgrid[:-1]

    random.seed(meta.surv)
    randos = [random.choice(us.lrange(obsgridmids)) for x in us.lrange(meta.colors)]#sp.stats.uniform(loc=zobsgrid[0],scale=zrange).rvs(len(meta.colors))

    gridpdfs,zfactors = [],[]
    inttrus,samppdfs,intobss = [],[],[]
    for k in us.lrange(obsgridmids):
        if meta.sigma == True:
            #print 'making nontrivial zfactors'
            zfactor = (1.+obsgridmids[k])**meta.noisefact
        else:
            zfactor = 1.
        zfactors.append(zfactor)
        gridpdf = makepdfs(meta,test,j,obsgridmids[k],zfactors,ztrugrid)
        for x in randos:
            if k == x:
                #inttrus.append(trugridmids[k])
                samppdfs.append(gridpdf)
                intobss.append(obsgridmids[k])
        gridpdfs.append(gridpdf)
    gridpdfs = np.array(gridpdfs)
    sumx = np.sum(gridpdfs,axis=0)*trugriddifs
    sumy = np.sum(gridpdfs,axis=1)*obsgriddifs
    gridpdfs = (gridpdfs/sumx).T
    #sumx = np.sum(samppdfs,axis=0)*trugriddifs
    samppdfs = (samppdfs/sumx)

    return(gridpdfs,sumx,sumy,ztrugrid,zobsgrid,zfactors,inttrus,intobss,samppdfs)

# def makelf(meta,test,j):

#     eps = 1./100.
#     zrange = test.zhis[-1]-test.zlos[0]
#     ztrugrid = np.arange(test.zlos[0],test.zhis[-1]+eps,eps)
#     ztrugrid = ztrugrid[ztrugrid!=0.]
#     zobsgrid = np.arange(test.zlos[0]-zrange,test.zhis[-1]+zrange+eps,eps)
#     zobsgrid = zobsgrid[zobsgrid!=0.]
#     trugridmids = (ztrugrid[1:]+ztrugrid[:-1])/2.
#     obsgridmids = (zobsgrid[1:]+zobsgrid[:-1])/2.
#     trugriddifs = ztrugrid[1:]-ztrugrid[:-1]
#     obsgriddifs = zobsgrid[1:]-zobsgrid[:-1]

#     pdfs,intobss,zfactors,intshifts = [],[],[],[]
#     inttrus = sp.stats.uniform(loc=ztrugrid[0],scale=zrange).rvs(len(meta.colors))# for x in us.lrange(meta.colors)]
#     for x in us.lrange(meta.colors):
#         #inttru = sp.stats.uniform(loc=ztrugrid[0],scale=zrange).rvs(1)[0]
#         #inttrus.append(inttru)

#         if meta.sigma == True:
#             zfactor = (1.+inttrus[x])**meta.noisefact
#         else:
#             zfactor = 1.
#         zfactors.append(zfactor)

#         intshift = []
#         for p in xrange(test.npeaks[j]):
#             shift = test.shift[j][p]#-inttrus[x]#np.random.normal(loc=0.,scale=test.varZs[j][p]*zfactor)
#             intshift.append(shift)
#         intshifts.append(intshift)

#         intobs = []
#         if meta.outlier == 0:
#             outlier = None
#             for p in xrange(test.npeaks[j]):
#                 obs = inttrus[x]+intshift[p]
#                 intobs.append(obs)
#         else:
#             outlier = test.peaklocs
#             intobs = [inttrus[x]+test.shift[0]]
#             intobs.extend(outlier)

# #         for p in xrange(test.npeaks[j]):
# #             obs = inttrus[x]+intshift[p]
# #             intobs.append(obs)
#         intobss.append(intobs)
# #             obs = inttru+intshift[0]
# #             intobs.append(obs)
# #             if test.npeaks[j] != 1:
# #                 for p in xrange(test.npeaks[j]-1):
# #                     obs = test.peaklocs[p]
# #                     intobs.append(obs)
# #             intobss.append(intobs)

#         dgen = []
#         if meta.degen != 0:
#             for y in xrange(meta.degen):
#                 weight = (1./test.sigdegen[y][0])/sum(1./test.sigdegen[:,0])
#                 dgen.append(test.mudegen[y][0],test.sigdegen[y][0],weight)
#             dgen = us.gmix(dgen,(min(grid),max(grid)))
#         else:
#             dgen = None

#         intp = None
#         sigZs = test.varZs[j]*zfactor
#         obsZs = test.obsZs[j]
#         pdf = us.makepdf(zobsgrid,inttrus[x],intshift,obsZs,sigZs,intp=intp,dgen=dgen,outlier=outlier)
#         print(inttrus[x],intshift,obsZs,sigZs)
#         pdfs.append(pdf)

#     pdfs = np.array(pdfs)
#     inttrus = np.array(inttrus)
#     intobss = np.array(intobss)
#     zfactors = np.array(zfactors)

# #     inttrus = np.array([sp.stats.uniform(loc=ztrugrid[0],scale=zrange).rvs(1)[0] for x in us.lrange(meta.colors)])
# #     if meta.sigma == True:
# #         zfactors = (1.+inttrus)**meta.noisefact
# #     else:
# #         zfactors = np.array([1.]*len(inttrus))
# #     intshifts = np.array([[np.random.normal(loc=0.,scale=test.varZs[j][p]*zfactors[x]) for p in xrange(test.npeaks[j])] for x in us.lrange(meta.colors)])
# #     if meta.outlier == 0:
# #         intobss = np.array([[inttrus[x]+intshifts[x][p] for p in xrange(test.npeaks[j])] for x in us.lrange(meta.colors)])
# #     else:
# #         intobss = []
# #         for x in xrange(len(meta.colors)):
# #             obsZ = np.array([inttrus[x]+intshifts[x][0]])
# #             if test.npeaks[j] != 1:
# #                 obsZ = np.concatenate((obsZ,test.peaklocs[:test.npeaks[j]-1]))
# #             intobss.append(obsZ)
# #     intobss = np.array(intobss)

# #     if meta.degen != 0:
# #         dgen = []
# #         for x in xrange(meta.degen):
# #             weight = (1./test.sigdegen[x][0])/sum(1./test.sigdegen[:,0])
# #             dgen.append(test.mudegen[x][0],test.sigdegen[x][0],weight)
# #         dgen = us.gmix(dgen,(min(grid),max(grid)))
# #     else:
# #         dgen = None
# #     intp = None

# #     pdfs = []
# #     for x in us.lrange(meta.colors):
# #         sigZs = test.varZs[j]*zfactors[x]
# #         pdf = us.makepdf(zobsgrid,inttrus[x],intobss[x],sigZs,test.weights,intp=intp,dgen=dgen)
# #         pdfs.append(pdf)
# #     pdfs = np.array(pdfs)

# #     for x in xrange(len(meta.colors)):
# #         pdf_info = [[intobss[x][p],test.sigZs[j][p],1./float(test.npeaks[j]+meta.degen)] for p in xrange(test.npeaks[j])]
# #         pdf_info = np.array(pdf_info)
# #         pdf_infos = us.gmix(pdf_info,(zobsgrid[0],zobsgrid[-1]))
# #         plot_y = makepdfs(meta,test,intobss[x][0],j,zfactors[x],zobsgrid)#pdf_infos.pdf(np.array(obsgridmids))
# #         if meta.degen != 0:
# #             for y in xrange(meta.degen):
# #                 sps_pdfs.vlines(test.mudegen[y][0],0.,(test.zhis[-1]-test.zlos[0])/test.bindif,color='k')
# #                 func = us.tnorm(test.mudegen[y][0],test.sigdegen[y][0],(min(zobsgrid),max(zobsgrid)))
# #                 binlos = [np.argmin(np.abs(obsgridmids-intobss[x][p])) for p in xrange(test.npeaks[j])]
# #                 spread = np.sum([max(func.cdf(zobsgrid[b+1])-func.cdf(zobsgrid[b]),sys.float_info.epsilon) for b in binlos])
# #                 plot_y += spread/float(test.npeaks[j]+meta.degen)

#     return(pdfs,inttrus,intobss,zfactors)

def plot_lfs(meta,test):
    lfdir = os.path.join(meta.datadir,'lfs')
    if meta.shape == 1:
        j = test.randos[0]
    else:
        options = [gal.ind for gal in test.gals if gal.ncomps > 1]
        j = np.random.choice(options)
    f = plt.figure(figsize=(5,10))
    sps = f.add_subplot(2,1,1)
    f.suptitle(meta.name+r' $p_{'+str(j)+r'}(z_{obs}|z_{tru})$')
    sps.set_ylabel(r'$z_{obs}$')
    sps.set_xlabel(r'$z_{tru}$')

    lf = make2dlf(meta,test,j)

    ztrugrid = lf[3]
    zobsgrid = lf[4]
    zrange = test.zhis[-1]-test.zlos[0]
    trugridmids = (ztrugrid[1:]+ztrugrid[:-1])/2.
    obsgridmids = (zobsgrid[1:]+zobsgrid[:-1])/2.
    trugriddifs = ztrugrid[1:]-ztrugrid[:-1]
    obsgriddifs = zobsgrid[1:]-zobsgrid[:-1]
    extended = np.arange(test.zlos[0]-zrange,test.zhis[-1]+zrange+test.bindif,test.bindif)
    extmids = (extended[1:]+extended[:-1])/2.
    extdifs = extended[1:]-extended[:-1]

    sps.pcolorfast(ztrugrid,zobsgrid,np.transpose(lf[0]),cmap=cm.gray_r)

    sps_pdfs = f.add_subplot(2,1,2)
    sps_pdfs.set_title(r'Example PDFs')
    np.random.seed(seed=test.seed)
    #pdfs,inttrus,intobss,zfactors = makelf(meta,test,j)
    inttrus = lf[6]
    intobss = lf[7]
    samppdfs = lf[8]
#     inttrus = np.array([sp.stats.uniform(loc=ztrugrid[0],scale=zrange).rvs(1)[0] for x in xrange(len(meta.colors))])
#     zfactors = lf[5]
#     intshifts = np.array([[np.random.normal(loc=0.,scale=test.sigZs[j][p]) for p in xrange(test.npeaks[j])] for x in xrange(len(meta.colors))])
#     if meta.outlier == 0:
#         intobss = np.array([[inttrus[x]+intshifts[x][p] for p in xrange(test.npeaks[j])] for x in xrange(len(meta.colors))])
#     else:
#         intobss = []
#         for x in xrange(len(meta.colors)):
#             obsZ = np.array([inttrus[x]+intshifts[x][0]])
#             if test.npeaks[j] != 1:
#                 obsZ = np.concatenate((obsZ,test.peaklocs[:test.npeaks[j]-1]))
#             intobss.append(obsZ)
#     intobss = np.array(intobss)

    for x in xrange(len(meta.colors)):
#         pdf_info = [[intobss[x][p],test.sigZs[j][p],1./float(test.npeaks[j]+meta.degen)] for p in xrange(test.npeaks[j])]
#         pdf_info = np.array(pdf_info)
#         pdf_infos = us.gmix(pdf_info,(zobsgrid[0],zobsgrid[-1]))
#         plot_y = makepdfs(meta,test,intobss[x][0],j,zfactors[x],zobsgrid)#pdf_infos.pdf(np.array(obsgridmids))
#         if meta.degen != 0:
#             for y in xrange(meta.degen):
#                 sps_pdfs.vlines(test.mudegen[y][0],0.,(test.zhis[-1]-test.zlos[0])/test.bindif,color='k')
#                 func = us.tnorm(test.mudegen[y][0],test.sigdegen[y][0],(min(zobsgrid),max(zobsgrid)))
#                 binlos = [np.argmin(np.abs(obsgridmids-intobss[x][p])) for p in xrange(test.npeaks[j])]
#                 spread = np.sum([max(func.cdf(zobsgrid[b+1])-func.cdf(zobsgrid[b]),sys.float_info.epsilon) for b in binlos])
#                 plot_y += spread/float(test.npeaks[j]+meta.degen)
#         if meta.shape > 1:
#             for n in xrange(test.npeaks[j]-1):
#                 sps_pdfs.vlines(test.obsZs[j][p],0.,(test.zhis[-1]-test.zlos[0])/test.bindif,color='k')
#                 func = us.tnorm(test.obsZs[j][p],test.sigZs[j][n],(min(zobsgrid),max(zobsgrid)))
#                 binlos = [np.argmin(np.abs(obsgridmids-intobss[x][p])) for p in xrange(test.npeaks[j])]
#                 spread = np.sum([max(func.cdf(zobsgrid[b+1])-func.cdf(zobsgrid[b]),sys.float_info.epsilon) for b in binlos])
#                 plot_y += spread/float(test.npeaks[j]+meta.degen)

        plot_y = samppdfs[x]#pdfs[x]#/np.dot(pdfs[x],obsgriddifs)
        sps_pdfs.plot(trugridmids,plot_y,color=meta.colors[x])
        sps_pdfs.vlines(intobss[x],0.,max(plot_y),color=meta.colors[x],linestyle=s_tru,linewidth=w_tru,dashes=d_tru,alpha=a_tru)
        sps.hlines(intobss[x],ztrugrid[0],ztrugrid[-1],color=meta.colors[x],linestyle=s_tru,linewidth=w_tru,dashes=d_tru,alpha=a_tru)
        #for p in xrange(test.npeaks[j]):
            #sps.scatter(inttrus[x],intobss[x][p],color=test.meta.colors[x])
            #sps_pdfs.vlines(intobss[x][p],0.,max(plot_y),color=meta.colors[x],linestyle=s_map,linewidth=w_map,dashes=d_map,alpha=a_map)

    sps_pdfs.set_ylabel(r'$p(\vec{d}|z)$')
    sps_pdfs.set_xlabel(r'$z$')
    sps_pdfs.set_xlim(test.binends[0]-10.*test.bindif,test.binends[-1]+10.*test.bindif)
    sps.set_ylim(test.binends[0]-10.*test.bindif,test.binends[-1]+10.*test.bindif)
    #sps_pdfs.set_ylim(0.,(test.zhis[-1]-test.zlos[0])/test.bindif)

    f.savefig(os.path.join(meta.simdir,'zobsvztru.pdf'),bbox_inches='tight', pad_inches = 0)

def plot_mlehist(meta,test):
    f = plt.figure(figsize=(5,5))
    sps = f.add_subplot(1,1,1)
    sps.set_title(meta.name+r' MLE Histogram')
    sps.set_ylabel(r'$N(z)$')
    sps.set_xlabel(r'$z$')
    sps.hist(test.mapZs,test.binends)
    f.savefig(os.path.join(meta.simdir,'mlehist.pdf'),bbox_inches='tight', pad_inches = 0)
    return

# plot the underlying P(z) and its components
def plot_ests(meta,test):

    f = plt.figure(figsize=(5,5))
    sys.stdout.flush()
    sps = f.add_subplot(1,1,1)
    sps.set_title(meta.name+r' Comparisons $N(z)$')
    sps.plot(zrange,prange/np.sum(prange*np.average(zrange[1:]-zrange[:-1]))*test.ngals,color='k')
    plotstep(sps,test.binends,test.stkNz,c=c_stk,s=s_stk,w=w_stk,d=d_stk,a=a_stk,l=l_stk)
    plotstep(sps,test.binends,test.intNz,c=c_int,s=s_int,w=w_int,d=d_int,a=a_int,l=l_int)
#     plotstep(sps,test.binends,test.logmapNz,c=c_map,s=s_map,w=w_map,d=d_map,a=a_map,l=l_map)
#     plotstep(sps,test.binends,test.logmmlNz,c=c_mml,s=s_mml,w=w_mml,d=d_mml,a=a_mml,l=l_mml)
#     plotstep(sps,test.binends,test.logexpNz,c=c_exp,s=s_exp,w=w_exp,d=d_exp,a=a_exp,l=l_exp)
    sps.set_ylabel(r'$N(z)$')
    sps.set_xlabel(r'$z$')
    sps.set_ylim(0.,np.e*test.ngals)
    sps.legend(fontsize='small',loc='upper right')
    f.savefig(os.path.join(meta.simdir,'estNz.pdf'),bbox_inches='tight', pad_inches = 0)
    return
