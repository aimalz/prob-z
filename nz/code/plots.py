import matplotlib.pyplot as plt
from setup import *

# #plot the true p(z) and its components
# def plot_priorgen():
#   print 'plot_priorgen'
#   f = plt.figure(figsize=(5,5))
#   #print 'one'
#   sys.stdout.flush()
#   sps = f.add_subplot(1,1,1)
#   f.suptitle('True p(z)')# for $J=$'+str(ngals_seed))
#   sps.step(zmids,plotrealistic_pdf,c='k',label='True p(z)')
#   for k in range(0,len(plotrealistic_comps)):
#     sps.step(zmids,plotrealistic_comps[k],c=colors[k],label='component '+str(realistic_prep[k][2])+'N('+str(realistic_prep[k][0])+','+str(realistic_prep[k][1])+')')
#   sps.set_ylabel('p(z)')
#   sps.set_xlabel('z')
#   sps.legend(fontsize='x-small',loc='upper left')
#   f.savefig(os.path.join(topdir,'physPz.png'))
#   #print 'done?'
#   return

#plot samples of true N(z)
def plot_true():
  print 'plot_true'
  #print plt.get_backend()
  f = plt.figure(figsize=(5*nsurvs,5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
  #print 'created subplots'
  for s in survnos:
    sps = f.add_subplot(1,nsurvs,s+1)
    sps.set_title(r'True $N(z)$ for '+str(seed_ngals[s])+' galaxies')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    #sps.hlines(logtrueNz[s],zlos,zhis,color='k',linestyle='--',label=r'true $\ln N(z)$')
    sps.step(zmids,logflatNz[s],color='k',label=r'flat $\ln N(z)$',where='mid')
    # thing.step(zmids,[-10]*nbins,color='k',label=r'$J='+str(seed_ngals[s])+r'$')
    for n in sampnos:
        sps.step(zmids,logsampNz[s][n],color=colors[n%6],label=r'true $\ln N(z)$ '+str(n+1),where='mid')#,alpha=0.1)
    sps.set_ylim((-1.,ymax[s]+1.))
    sps.legend(loc='upper left',fontsize='x-small')
  f.savefig(os.path.join(topdir,'trueNz.png'))
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
def plot_lfs():
  print('plot_lfs')
  #randos = random.sample(pobs[-1][0],ncolors)
  a = min(float(ncolors)/m.sqrt(seed_ngals[-1]),1.)
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
  f.suptitle('Observed galaxy posteriors')
  for p in pobs[-1][0]:
    sps.step(binmids,p,where='mid',alpha=a)
    sps.set_ylabel(r'$p(z|\vec{d})$')
    sps.set_xlabel(r'$z$')
  f.savefig(os.path.join(topdir,'samplepzs.png'))
  return

#plot some samples from prior
def plot_prior_samps():
  print 'plot_prior_samps'
  #print plt.get_backend()
  f = plt.figure(figsize=(5*nsurvs,5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
  priorsamps = np.exp(np.array([priordists[s].sample_ps(ncolors) for s in range(0,nsurvs)]))
  for s in survnos:
    sps = f.add_subplot(1,nsurvs,s+1)
#    if nsurvs > 1:
#        thing = sps[s]
#    else:
#        thing = sps
    sps.set_title(r'Prior samples for $J_{0}='+str(seed_ngals[s])+r'$')
    sps.set_xlabel(r'binned $z$')
    sps.set_ylabel(r'$\ln N(z)$')
    sps.set_ylim((0,ymax_e[s]+m.log(seed_ngals[s])))
    #sps.hlines(trueNz[s],zlos,zhis,color='k',linestyle='--',label=r'true $\ln N(z)$')
    sps.step(zmids,flatNz[s],color='k',label=r'flat $\ln N(z)$',where='mid')
    #thing.step(zmids,[-10]*nbins,color='k',label=r'$J='+str(seed_ngals[s])+r'$')
    for n in sampnos:
      sps.step(zmids,sampNz[s][n],color='k',linewidth=2,label=r'true $\ln N(z)$ '+str(n+1),where='mid')
    for c in colornos:
      sps.step(binmids,priorsamps[s][c],color=colors[c],where='mid')
    sps.legend(loc='upper left',fontsize='x-small')
  f.savefig(os.path.join(topdir,'priorsamps.png'))
  #print 'done'
  return

#plot initial values
def plot_ivals():
  print('plot_ivals')
  a = m.sqrt(ncolors/nwalkers)
  f = plt.figure(figsize=(5*ntests,5*nsurvs))#plt.subplots(nsurvs, ntests, figsize=(5*ntests,5*nsurvs),sharey='row',sharex=True)
  f.suptitle('Initialization of '+str(nwalkers)+' walkers')
  p = 0
  for s in survnos:
    for t in testnos:
      p+=1
      sps = f.add_subplot(nsurvs,ntests,p)
      sps.set_ylabel(r'$\ln N(z)$ with $J_{0}='+str(seed_ngals[s])+'$')
      sps.set_xlabel(r'$z$')
      sps.set_title(setups[t])
      #sps[s][t].set_ylabel(r'$\ln N(z)$ with $J='+str(seed_ngals[s])+'$')
      #sps[s][t].set_xlabel(r'$z$')
      #sps[s][t].set_title(setups[t])
      for iguess in iguesses[s][t]:
        sps.step(binmids,iguess,alpha=0.5,where='mid')
      sps.step(binmids,means[s][t],color='k',linewidth=2,where='mid')
        #sps[s][t].step(zmids[0:ndims[lenno]],iguess,alpha=0.5)
        #sps[s][t].step(zmids[0:ndims[lenno]],means[s][t],color='k',linewidth=2)
  f.savefig(os.path.join(topdir,'initializations.png'),dpi=100)
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
