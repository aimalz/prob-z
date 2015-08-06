import matplotlib.pyplot as plt

import numpy as np
import sys
import os

import timeit
import random
import math as m

# #set true value of N(z)=theta
# #tuples of form z_center, spread,magnitude
# realistic_prep = [(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
# realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in zmids] for tup in realistic_prep])
# realistic = [sum(realistic_comps[k]) for k in binnos]
# realsum = sum(realistic)
# realistic_pdf = np.array([realistic[k]/realsum/zdifs[k] for k in binnos])

# plotrealistic = [sum(real) for real in realistic_comps]
# plotrealisticsum = sum(plotrealistic)
# plotrealistic_comps = np.transpose([[r/plotrealisticsum for r in real] for real in realistic_comps])
# plotrealistic_pdf = np.array([plotrealistic[k]/plotrealisticsum for k in binnos])

#plot the true p(z) and its components
def plot_priorgen(metainfo):

  meta = metainfo

  realsum = sum(meta.realistic)
  realistic_pdf = meta.realistic/meta.zdifs/realsum
  plotrealistic = np.array([sum(r) for r in meta.realistic_comps])
  plotrealisticsum = sum(plotrealistic)#meta.realistic_comps)
  plotrealistic_comps = np.transpose(meta.realistic_comps/plotrealisticsum)
  plotrealistic_pdf = plotrealistic/plotrealisticsum

  #print 'plot_priorgen'
  f = plt.figure(figsize=(5,5))
  #print 'one'
  sys.stdout.flush()
  sps = f.add_subplot(1,1,1)
  f.suptitle('True p(z)')# for $J=$'+str(ngals_seed))
  sps.step(meta.allzmids,plotrealistic_pdf,c='k',label='True p(z)')
  for k in range(0,len(meta.real)):
    sps.step(meta.allzmids,plotrealistic_comps[k],c=meta.colors[k],label='component '+str(meta.real[k][2])+'N('+str(meta.real[k][0])+','+str(meta.real[k][1])+')')
  sps.set_ylabel('p(z)')
  sps.set_xlabel('z')
  sps.legend(fontsize='x-small',loc='upper left')
  f.savefig(os.path.join(meta.topdir,'physPz.png'))
  #print 'done?'
  return

#plot samples of true N(z) for one set of parameters and one survey size
def plot_true_setup(survinfo):

  (meta,p_run,s_run) = survinfo

  f = plt.figure(figsize=(5,5))#*nsurvs,5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
  #print 'created subplots'
  #for s in survnos:
  sps = f.add_subplot(1,1,1)#,nsurvs,s+1)
  sps.set_title(r''+str(meta.params[p_run.p])+' Parameter True $N(z)$ for '+str(s_run.seed)+' galaxies')
  sps.set_xlabel(r'binned $z$')
  sps.set_ylabel(r'$\ln N(z)$')
  sps.set_ylim(-1.,s_run.seed)#m.log(s_run.seed/meta.zdif)))
  #sps.hlines(logtrueNz[s],zlos,zhis,color='k',linestyle='--',label=r'true $\ln N(z)$')
  sps.step(p_run.zmids,s_run.logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
  # thing.step(zmids,[-10]*nbins,color='k',label=r'$J='+str(seed_ngals[s])+r'$')
  return((f,sps))

def plot_true((f,sps),sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

#  print 'plot_true'
  #print plt.get_backend()
  #for n_run in sampnos:
  sps.step(p_run.zmids,n_run.logsampNz,color=meta.colors[n_run.n%6],label=r'true $\ln N(z)$ '+str(n_run.n+1),where='mid')#,alpha=0.1)
  return(f,sps)

def plot_true_wrapup((f,sps),survinfo):

  (meta,p_run,s_run) = survinfo

  sps.legend(loc='upper left',fontsize='x-small')
  f.savefig(os.path.join(s_run.topdir_s,'trueNz.png'))
  #print 'done'
  return

#compare true N(z) to true and observed samples
#randos = [random.choice(range(0,seed_ngals[s])) for s in survnos]

# #plot sampled and observed N(z)
# def plot_obs():
#   print 'plot_obs'
#   f,sps = plt.subplots(1, nsurvs, figsize=(5*nsurvs,5),sharey='row')
#   #for i in range(0,seed_ngals):
#   for s in survnos:#only one seed this time, don't bother with loop
#     sps = f.add_subplot(1,nsurvs,s+1)
# #    if nsurvs > 1:
# #        thing = sps[s]
# #    else:
# #        thing = sps
#     sps.set_title('Simulated Data for J='+str(seed_ngals[s]))
#     sps.set_ylabel(r'$N(z)$')
#     sps.set_xlabel(r'$z$')
#     sps.semilogy()
#     sps.set_ylim(1e-2,10**(3+s))
#     sps.step(binmids,full_trueNz[s],color='r',label=r'sample $N(z)$')
#     sps.step(binmids,full_trueNz[s],color='b',label=r'observed $N(z)$')
#     sps.step(binmids,full_trueNz[s],color='k',label=r'true $N(z)$')
#     #thing.step(binmids,avgsamp[s],color='b',label=r'average sample $N(z)$',linestyle='--')
#     #thing.fill_between(binmids,minsamp[s],maxsamp[s],color='b',alpha=0.5,label=r'true sample $N(z)$ RMS errors')
#     #thing.step(binmids,avgobs[s],color='r',label=r'average observed $N(z)$',linestyle='--')
#     #thing.fill_between(binmids,minobs[s],maxobs[s],color='r',alpha=0.5,label=r'observed $N(z)$ RMS errors')
#     for n in sampnos:
#         sps.step(binmids,sampNz[s][n],c='b',linestyle='-')#,label='true N(z) for draw \#'+str(rando))
#         sps.step(binmids,obsNz[s][n],c='r',linestyle='-')#,label='observed N(z) for draw \#'+str(rando))
#     sps.legend(loc='upper left',fontsize='x-small')
#   f.savefig(os.path.join(topdir,'obsNz.png'))
#   #print 'done'
#   return

#plot some random p(z)
# def plot_lfs():
#   print('plot_randos')
#   randz = random.sample(binmids,ncolors)#(pobs[-1][0],ncolors)
#   randpobs = []
#   for j in colornos:
#     func = sp.stats.norm(loc=randz[j],scale=zdif*(randz[j]+1.))
#     lo = np.array([max(sys.float_info.epsilon,func.cdf(binends[k])) for k in new_binnos])
#     hi = np.array([max(sys.float_info.epsilon,func.cdf(binends[k+1])) for k in new_binnos])
#     spread = (hi-lo)/zdif
#     #normalize probabilities to sum to 1
#     summed = sum(spread)
#     p = spread/summed
#     randpobs.append(p)
#   f = plt.figure(figsize=(5,5))
#   sps = f.add_subplot(1,1,1)
#   f.suptitle('Observed Galaxy Posteriors')
#   for k in colornos:
#     sps.step(binmids,randpobs[k],c=colors[k])
#     sps.set_ylabel('p(z)')
#     sps.set_xlabel('z')
#   f.savefig(os.path.join(topdir,'samplepzs.png'))
#   return

def plot_pdfs(sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

  a = min(float(meta.ncolors)/m.sqrt(s_run.seed),1.)
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
  f.suptitle('Observed galaxy posteriors')
  #print('plot_lfs')
  #randos = random.sample(pobs[-1][0],ncolors)
  for p in n_run.pobs:
    sps.step(n_run.binmids,p,where='mid',alpha=a)
  sps.set_ylabel(r'$p(z|\vec{d})$')
  sps.set_xlabel(r'$z$')
  f.savefig(os.path.join(p_run.topdir_p,'samplepzs.png'))
  return

def plot_truevmap_setup(metainfo):

  meta = metainfo

  global a_tvm
  a_tvm = 1./meta.samps
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
  f.suptitle('True Redshifts vs. MAP Redshifts')
  #print('plot_lfs')
  #randos = random.sample(pobs[-1][0],ncolors)
  sps.set_ylabel(r'Observed $z$')
  sps.set_xlabel(r'True $z$')
  return((f,sps))

def plot_truevmap((f,sps),sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

  sps.scatter(n_run.trueZs,n_run.obsZs,alpha=a_tvm)
  return((f,sps))

def plot_truevmap_wrapup((f,sps),survinfo):

  (meta,p_run,s_run) = survinfo

  f.savefig(os.path.join(s_run.topdir_s,'truevmap.png'))
  return

#plot some samples from prior
def plot_priorsamps(sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

#  print 'plot_prior_samps'
  #print plt.get_backend()
  f = plt.figure(figsize=(5,5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
  sps = f.add_subplot(1,1,1)
  priorsamps = np.exp(np.array(n_run.priordist.sample_ps(meta.ncolors)[0]))
#  for s in survnos:
#    sps = f.add_subplot(1,nsurvs,s+1)
#    if nsurvs > 1:
#        thing = sps[s]
#    else:
#        thing = sps
  sps.set_title(r'Prior samples for $J_{0}='+str(s_run.seed)+r'$')
  sps.set_xlabel(r'binned $z$')
  sps.set_ylabel(r'$\ln N(z)$')
  sps.set_ylim(0.,s_run.seed)#max(n_run.full_logflatNz)+m.log(s_run.seed/meta.zdif)))
  #sps.hlines(trueNz[s],zlos,zhis,color='k',linestyle='--',label=r'true $\ln N(z)$')
  sps.step(n_run.binmids,n_run.full_logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
  #thing.step(zmids,[-10]*nbins,color='k',label=r'$J='+str(seed_ngals[s])+r'$')
#  for n in sampnos:
#      sps.step(zmids,sampNz[s][n],color='k',linewidth=2,label=r'true $\ln N(z)$ '+str(n+1),where='mid')
  for c in meta.colornos:
      sps.step(n_run.binmids,priorsamps[c],color=meta.colors[c],where='mid')
  sps.legend(loc='upper left',fontsize='x-small')
  f.savefig(os.path.join(s_run.topdir_s,'priorsamps.png'))
  #print 'done'
  return

#plot initial values
def plot_ivals_setup(sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

    #  print('plot_ivals')
  #a = m.sqrt(meta.ncolors/n_run.nwalkers)
  f,sps = plt.subplots(1, meta.ninits, figsize=(5*meta.ninits,5),sharey=True,sharex=True)
  f.suptitle('Initialization of '+str(n_run.nwalkers)+' walkers')
#  p = 0
  return((f,sps))

def plot_ivals((f,sps),initinfo):

  (meta,p_run,s_run,n_run,i_run) = initinfo

# #  print('plot_ivals')
#   a = m.sqrt(ncolors/nwalkers)
  #f = plt.figure(figsize=(5*meta.ninits,5))#plt.subplots(nsurvs, ntests, figsize=(5*ntests,5*nsurvs),sharey='row',sharex=True)
#   f.suptitle('Initialization of '+str(nwalkers)+' walkers')
# #  p = 0
#   for s in survnos:
#     for t in testnos:
# #      p+=1
  sps = f.add_subplot(1,meta.ninits,i_run.i+1)
  sps.set_ylabel(r'$\ln N(z)$ with $J_{0}='+str(s_run.seed)+'$')
  sps.set_xlabel(r'$z$')
  sps.set_title(meta.init_names[i_run.i])
      #sps[s][t].set_ylabel(r'$\ln N(z)$ with $J='+str(seed_ngals[s])+'$')
      #sps[s][t].set_xlabel(r'$z$')
      #sps[s][t].set_title(setups[t])
  for iguess in i_run.iguesses:
    sps.step(n_run.binmids,iguess,alpha=0.5,where='mid')
  sps.step(n_run.binmids,i_run.mean,color='k',linewidth=2,where='mid')
        #sps[s][t].step(zmids[0:ndims[lenno]],iguess,alpha=0.5)
        #sps[s][t].step(zmids[0:ndims[lenno]],means[s][t],color='k',linewidth=2)
  return((f,sps))

def plot_ivals_wrapup((f,sps),sampinfo):

  (meta,p_run,s_run,n_run) = sampinfo

  f.savefig(os.path.join(n_run.topdir_n,'initializations.png'),dpi=100)
  return

# #plot likelihoods as test
# npois = [np.arange(1,5*seed_ngals[s]) for s in survnos]
# #poisson likelihood: ln[exp[-J]J^N/N!] or is it ln[exp[-N]N^J/J!]?
# explf = [[[-N+ngals[s][n]*np.log(N)-np.log(float(m.factorial(ngals[s][n]))) for N in npois[s]] for n in sampnos] for s in survnos]
# #explf = [[[x if ~np.isnan(x) and ~np.isinf(x) else np.log(sys.float_info.min) for x in e] for e in lf] for lf in explf]
# poissonlf = [[[sp.stats.poisson.logpmf(seed_ngals[s],N) for N in npois[s]] for n in sampnos] for s in survnos]
# obsparams = [[[logflatPz[s]+np.log(N) for N in npois[s]] for n in sampnos] for s in survnos]
# #obs likelihood: ln[p(N|{d})]-ln[p(N)]
# obslf = [[[posts[s][n].lnprob(tp)-priordists[s].logpdf(tp) for tp in obsparams[s][n]] for n in sampnos] for s in survnos]

# f,sps = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5))
# for s in survnos:
#     sps[s].set_title(r'Log-probability as function of $\int N(z)dz\sim'+str(seed_ngals[s])+r'$')
#     sps[s].set_xlabel(r'$\exp[\vec{\theta}]\cdot\vec{\Delta}$')
#     sps[s].set_ylabel(r'$\ln[p(\{\vec{d}_{j}\}_{J}|\vec{\theta})]$')
#     sps[s].set_ylim(-10*seed_ngals[s],0.)#max(-seed_ngals[s]*np.log(seed_ngals[s]),np.log(sys.float_info.min)),0.)
#     sps[s].plot(npois[s],[0]*(5*seed_ngals[s]-1),c='b',label='Poisson Likelihood')
#     #sps[s].plot(npois[s],[0]*(5*seed_ngals[s]-1),c='g',label='Predicted Likelihood',linestyle='-')
#     sps[s].plot(npois[s],[0]*(5*seed_ngals[s]-1),c='r',label='Data-based Likelihood',linestyle='--')
#     sps[s].plot(npois[s],[0]*(5*seed_ngals[s]-1),c='g',label='SciPy Poisson PMF')
#     for n in sampnos:
#         sps[s].plot(npois[s],explf[s][n],c='b')
#         #sps[s].plot(npois[s],testlf[s][n],c='g',linestyle='-')
#         sps[s].plot(npois[s],obslf[s][n],c='r',linestyle='--')
#         sps[s].plot(npois[s],poissonlf[s][n],c='g')
#     sps[s].legend(loc='lower right')
# f.savefig(os.path.join(topdir,'lf.png'),dpi=100)


# def iplots(args):
#   start_time = timeit.default_timer()
#   if args[0] == 'meta':
#     plot_priorgen(args[1:])
#   if args[0] == 'sampNz':

#   elapsed = str(timeit.default_timer()-start_time
#   plottimer = open(plottime,'a')
#   plottimer.write(str(timeit.default_timer())+' '+arg+' '+str(elapsed)+'\n')
#   plottimer.close()

#   def p_level(self,)

#     #define true N(z),P(z) for plotting given number of galaxies
#     full_trueNz = np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(s_run.trueNz,np.array([sys.float_info.epsilon]*len(binback)))))
#     full_logtrueNz = np.log(full_trueNz)#[[m.log(full_trueNz[s][k]) for k in new_binnos] for s in survnos]
#     full_truePz = np.append(np.array([sys.float_info.epsilon]*len(binfront)),(np.append(s_run.truePz,np.array([sys.float_info.epsilon]*len(binback)))))
#     full_logtruePz = np.log(full_truePz)#[[m.log(full_truePz[s][k]) for k in new_binnos] for s in survnos]

#     #define flat N(z),P(z) for plotting
#     full_flatNz = np.array([s_run.flat]*self.new_nbins)
#     full_logflatNz = np.array([s_run.logflat]*self.new_nbins)
#     full_flatPz = np.array([p_run.avgprob]*self.new_nbins)
#     full_logflatPz = np.array([p_run.logavgprob]*self.new_nbins)

#initial plots
def iplots(meta,p_runs,s_runs,n_runs,i_runs):

  metainfo = meta
  plot_priorgen(meta)#plot underlying distribution for theta

  for p in meta.paramnos:
    for s in meta.survnos:

      survinfo = (meta,p_runs[(p)],s_runs[(p,s)])

      (f_true,sps_true) = plot_true_setup(survinfo)
      (f_truevmap,sps_truevmap) = plot_truevmap_setup(metainfo)

      for n in meta.sampnos:

        sampinfo = (meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)])

        (f_true,sps_true) = plot_true((f_true,sps_true),sampinfo)#plot true theta for each sample of each survey
        (f_truevmap,sps_truevmap) = plot_truevmap((f_truevmap,sps_truevmap),sampinfo)
        (f_ivals,sps_ivals) = plot_ivals_setup(sampinfo)

        if n==0:
          plot_priorsamps(sampinfo)#plot samples from prior

          if s==meta.survnos[-1]:

            plot_pdfs(sampinfo)#plot some random zPDFs

        for i in meta.initnos:
          initinfo = (meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i_runs[(p,s,n,i)])

          (f_ivals,sps_ivals) = plot_ivals((f_ivals,sps_ivals),initinfo)#plot initial values for sampler

        plot_ivals_wrapup((f_ivals,sps_ivals),sampinfo)

      plot_true_wrapup((f_true,sps_true),survinfo)
      plot_truevmap_wrapup((f_truevmap,sps_truevmap),survinfo)
