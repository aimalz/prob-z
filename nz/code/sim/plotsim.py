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

global s_tru,w_tru,a_tru,c_tru,d_tru,l_tru
s_tru,w_tru,a_tru,c_tru,d_tru,l_tru = '--',1.,1.,'k',[(0,(1,0.0001))],r' True '
global s_int,w_int,a_int,c_int,d_int,l_int
s_int,w_int,a_int,c_int,d_int,l_int = '--',1.,0.5,'k',[(0,(1,0.0001))],r' Interim '
global s_stk,w_stk,a_stk,c_stk,d_stk,l_stk
s_stk,w_stk,a_stk,c_stk,d_stk,l_stk = '--',1.,1.,'k',[(0,(1,2))],r' Stacked '
global s_map,w_map,a_map,c_map,d_map,l_map
s_map,w_map,a_map,c_map,d_map,l_map = '--',1.,0.75,'k',[(0,(4,4,2,4))],r' MMAP '
global s_exp,w_exp,a_exp,c_exp,d_exp,l_exp
s_exp,w_exp,a_exp,c_exp,d_exp,l_exp = '--',1.,0.25,'k',[(0,(4,4,2,4))],r' MExp '
global s_mml,w_mml,a_mml,c_mml,d_mml,l_mml
s_mml,w_mml,a_mml,c_mml,d_mml,l_mml = '--',1.,1.,'k',[(0,(3,2))],r' MMLE '
global s_smp,w_smp,a_smp,c_smp,d,smp,l_smp
s_smp,w_smp,a_smp,c_smp,d_smp,l_smp = '--',1.,1.,'k',[(0,(1,0.0001))],r' Sampled '

#making a step function plotter because pyplot is stupid
def plotstep(subplot,binends,plot,s='--',d=[(0,(1,0.0001))],c='k',w=1,l=None,a=1.):
    subplot.hlines(plot,binends[:-1],
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
#     plot_true(meta,test)
#     plot_liktest(meta,test)
    plot_pdfs(meta,test)
#    plot_truevmap(meta,test)
    plot_lfs(meta,test)
    print(meta.name+' plotted setup')

# plot the underlying P(z) and its components
def plot_physgen(meta,test):

#     global pranges
#     pranges = test.real.pdfs(zrange)
#     global prange
#     prange = np.sum(pranges,axis=0)

    f = plt.figure(figsize=(5,5))
    sys.stdout.flush()
    sps = f.add_subplot(1,1,1)
#     f.suptitle(meta.name+' True p(z)')
    sps.set_title(meta.name+r' True $p(z)$')
    sps.plot(zrange,prange,color='k')
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
    plotstep(sps,dummy_x,dummy_y,c=c_exp,s=s_map,w=w_exp,l=r' MLE $z$',d=d_map,a=a_map)
    sps.legend(loc='upper right',fontsize='x-small')
    #sps.set_title('multimodal='+str(meta.shape)+', noisy='+str(meta.noise))
    for r in us.lrange(test.randos):
        plotstep(sps,test.binends,test.pdfs[test.randos[r]],c=meta.colors[r],s=s_smp,w=w_smp,d=d_smp,a=a_smp)
        sps.vlines(test.truZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle=s_tru,linewidth=w_tru,dashes=d_tru,alpha=a_tru)
        sps.vlines(test.mapZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle=s_map,linewidth=w_map,dashes=d_map,alpha=a_map)
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
    sps.set_xlim(test.binlos[0]-meta.zdif,test.binhis[-1]+meta.zdif)
    sps.set_ylim(0.,1./meta.zdif)
    f.savefig(os.path.join(meta.simdir,'samplepzs.pdf'),bbox_inches='tight', pad_inches = 0)
    return

# # plot some individual posteriors
# def plot_pdfs(meta,test):
#     f = plt.figure(figsize=(5,5))
#     sps = f.add_subplot(1,1,1)
# #     f.suptitle('Observed galaxy posteriors for '+meta.name)
#     plotstep(sps,test.binends,test.intPz,c=c_int,s=s_int,d=d_int,a=a_int,w=w_int,l=l_int+r'$P(z)$')
#     sps.plot([-1.],[-1.],color='k',linestyle=s_tru,dashes=d_tru,alpha=a_tru,linewidth=w_tru,label=l_tru+r'$z$')
#     sps.plot([-1.],[-1.],color='k',linestyle=s_map,dashes=d_map,alpha=a_map,linewidth=w_map,label=r'Central $z$')
#     sps.legend(loc='upper left',fontsize='x-small')
#     #sps.set_title('multimodal='+str(meta.shape)+', noisy='+str(meta.noise))
#     for r in us.lrange(test.randos):
#         plotstep(sps,test.binends,test.pdfs[test.randos[r]],c=meta.colors[r],s=s_stk,d=d_stk,a=a_stk,w=w_stk)
#         sps.vlines(sps,test.truZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle=s_tru,linewidth=w_tru,dashes=d_tru)
#         sps.vlines(sps,test.obsZs[test.randos[r]],0.,max(test.pdfs[test.randos[r]]),color=meta.colors[r],linestyle=s_map,linewidth=w_map,dashes=d_map)
#     sps.set_ylabel(r'$p(z|\vec{d})$')
#     sps.set_xlabel(r'$z$')
#     sps.set_xlim(test.binlos[0]-meta.zdif,test.binhis[-1]+meta.zdif)
#     sps.set_ylim(0.,1./meta.zdif)
#     f.savefig(os.path.join(meta.simdir,'samplepzs.png'),bbox_inches='tight', pad_inches = 0)
#     return


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

def makelfs(meta,test,j):

    eps = 1./100.
    zrange = test.zhis[-1]-test.zlos[0]
    ztrugrid = np.arange(test.zlos[0],test.zhis[-1]+eps,eps)
    ztrugrid = ztrugrid[ztrugrid!=0.]
    zobsgrid = np.arange(test.zlos[0]-zrange,test.zhis[-1]+zrange+eps,eps)
    zobsgrid = zobsgrid[zobsgrid!=0.]
    trugridmids = (ztrugrid[1:]+ztrugrid[:-1])/2.
    obsgridmids = (zobsgrid[1:]+zobsgrid[:-1])/2.
    trugriddifs = ztrugrid[1:]-ztrugrid[:-1]
    obsgriddifs = zobsgrid[1:]-zobsgrid[:-1]

    gridpdfs = []
    zfactors = []
    for z_tru in trugridmids:
        if meta.sigma == True:
            zfactor = meta.noisefact*(z_tru-test.zlos[0])/zrange
        else:
            zfactor = meta.noisefact
        zfactors.append(zfactor)
        #gridpdf = np.array([sys.float_info.epsilon]*len(gridmids))
        gridpdf = []
        for z_obs in obsgridmids:
            val = sys.float_info.epsilon
            for pn in xrange(test.npeaks[j]):
                p = np.exp(-1.*(z_tru-z_obs+test.shift[j][pn])**2/2./(test.varZs[j][pn]*zfactor)**2)/np.sqrt(2*np.pi*(test.varZs[j][pn]*zfactor)**2)
                val += p
            for x in xrange(meta.degen):
                val += test.distdegen[x].pdf(np.array([z_tru,z_obs]))
            # normalize probabilities to integrate (not sum)) to 1
            gridpdf.append(val)
        #gridpdf = gridpdf/np.dot(gridpdf,obsgriddifs)
        gridpdfs.append(gridpdf)
    gridpdfs = np.array(gridpdfs)
    sumx = np.sum(gridpdfs,axis=0)*obsgriddifs
    sumy = np.sum(gridpdfs,axis=1)*trugriddifs
    #lf = np.array([np.array([allsummed[zo]*allsummed[zt] for zo in us.lrange(self.gridmids)]) for zt in us.lrange(self.gridmids)])
    #print(gridpdfs)

    return(gridpdfs,sumx,sumy,ztrugrid,zobsgrid,zfactors)


def plot_lfs(meta,test):
    lfdir = os.path.join(meta.datadir,'lfs')
    j = test.randos[0]#for j in us.lrange(test.randos):
    f = plt.figure(figsize=(5,10))
    sps = f.add_subplot(2,1,1)
    f.suptitle(meta.name+r' $p_{'+str(j)+r'}(z_{obs}|z_{tru})$')
    sps.set_ylabel(r'$z_{obs}$')
    sps.set_xlabel(r'$z_{tru}$')
    #sps.set_ylim(test.binlos[0]-test.bindif,test.binhis[-1]+test.bindif)

    lf = makelfs(meta,test,j)
    #print(''np.shape(lf))
                #lf = np.array([np.array([l[zo]*l[zt] for zo in us.lrange(self.gridmids)]) for zt in us.lrange(self.gridmids)])
#                 lfs.append(lf)

    ztrugrid = lf[3]#np.arange(test.zlos[0],test.zhis[-1]+1./100,1./100)
    zobsgrid = lf[4]#np.arange(test.zlos[0]-zrange,test.zhis[-1]+zrange+1./100,1./100)
    zrange = test.zhis[-1]-test.zlos[0]
    trugridmids = (ztrugrid[1:]+ztrugrid[:-1])/2.
    obsgridmids = (zobsgrid[1:]+zobsgrid[:-1])/2.
    trugriddifs = ztrugrid[1:]-ztrugrid[:-1]
    obsgriddifs = zobsgrid[1:]-zobsgrid[:-1]
    extended = np.arange(test.zlos[0]-zrange,test.zhis[-1]+zrange+test.bindif,test.bindif)
    extmids = (extended[1:]+extended[:-1])/2.
    extdifs = extended[1:]-extended[:-1]

    sps.pcolorfast(ztrugrid,zobsgrid,np.sqrt(np.transpose(lf[0])),cmap=cm.gray_r)

    sps_pdfs = f.add_subplot(2,1,2)
    sps_pdfs.set_title(r'Example PDFs')
    dummy_x,dummy_y = np.array([-1,-2,-3]),np.array([-1,-2,-3])
    #plotstep(sps_pdfs,dummy_x,dummy_y,c=c_tru,s=s_tru,w=w_tru,l=l_tru+r'$z$',d=d_tru,a=a_tru)
    #plotstep(sps_pdfs,dummy_x,dummy_y,c=c_exp,s=s_map,w=w_exp,l=r' MLE $z$',d=d_map,a=a_map)
    sps_pdfs.legend(loc='upper right',fontsize='x-small')
    for x in us.lrange(test.inttrus):
        pdf = [[test.intobss[x][p],lf[5][x]*test.varZs[j][p],1.] for p in xrange(test.npeaks[j])]
        if meta.degen != 0:
            for y in xrange(meta.degen):
                distdegen = us.tnorm(test.mudegen[y][0],test.sigdegen[y][0],(test.zlos[0],test.zhis[-1]))
                pdf.append([test.mudegen[y][1],test.sigdegen[y][1],distdegen.pdf(test.inttrus[x])])
        pdf = np.array(pdf)
        pdfs = us.gmix(pdf,(zobsgrid[0],zobsgrid[-1]))
        plot_ys = pdfs.pdfs(obsgridmids)
        binnedys = pdfs.pdfs(extmids)
        plot_y = np.sum(plot_ys,axis=0)
        plot_y = plot_y/sum(plot_y*obsgriddifs)
        binnedy = np.sum(binnedys,axis=0)
        binnedy = binnedy/sum(binnedy*extdifs)
        sps_pdfs.plot(obsgridmids,plot_y,color=meta.colors[x])
        #plotstep(sps_pdfs,extended,binnedy,c=meta.colors[x],a=0.5)
        sps_pdfs.vlines(test.inttrus[x],0.,max(plot_y),color=meta.colors[x],linestyle=s_tru,linewidth=w_tru,dashes=d_tru,alpha=a_tru)
        for p in us.lrange(test.intobss[x]):
            sps.scatter(test.inttrus[x],test.intobss[x][p],color=test.meta.colors[x])
            sps_pdfs.vlines(test.intobss[x][p],0.,max(plot_y),color=meta.colors[x],linestyle=s_map,linewidth=w_map,dashes=d_map,alpha=a_map)

    sps_pdfs.set_ylabel(r'$p(\vec{d}|z)$')
    sps_pdfs.set_xlabel(r'$z$')
    sps_pdfs.set_xlim(test.binends[0]-10.*test.bindif,test.binends[-1]+10.*test.bindif)
    sps_pdfs.set_ylim(0.,(test.zhis[-1]-test.zlos[0])/test.bindif)

#     sps_obs = f.add_subplot(2,1,2)
#     #print(np.shape(zobsgrid),np.shape(lf[1]))
#     sps_obs.plot(lf[1],obsgridmids)
#     sps_obs.set_xlabel('Sum')
#     sps_obs.set_ylabel(r'$z_{obs}$')

#     sps_tru = f.add_subplot(2,2,3)
#     #print(np.shape(ztrugrid),np.shape(lf[2]))
#     sps_tru.plot(trugridmids,lf[2])
#     sps_tru.set_ylabel('Sum')
#     sps_tru.set_xlabel(r'$z_{tru}$')

#     sps_tru.set_xlim(ztrugrid[0],ztrugrid[-1])
#     sps_tru.set_ylim(0.,2.)
#     sps.set_xlim(ztrugrid[0],ztrugrid[-1])
#     sps.set_ylim(zobsgrid[0],zobsgrid[-1])

    f.savefig(os.path.join(meta.simdir,'zobsvztru.pdf'),bbox_inches='tight', pad_inches = 0)
