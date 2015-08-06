import matplotlib
import matplotlib.pyplot as plt
from setup import *

import timeit
import random
import math as m
import statistics
import hickle as hkl
import cPickle

#complicated plots

#prepare to plot autocorrelation times as function of iteration number, one plot per survey
def plottimes_init(survinfo):

  (meta,p_run,s_run) = survinfo

  #print 'plottimes_init'
  plt.rc('text', usetex=True)
  #f_times,sps_times = plt.subplots(1, 1, figsize = (5,5))
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
  global a_times
  a_times = 2./p_run.ndims/meta.samps
  #if nsamps == 1:
  #  sps_times = [sps_times]
  ##sps_times = f_times.add_subplot(1,1,1)
  #for n in meta.sampnos:
  sps.set_title(str(meta.samps)+' Autocorrelation Times for '+str(s_run.seed)+' galaxies')
  sps.set_ylabel('autocorrelation time')
  sps.set_xlabel('number of iterations')
  sps.set_ylim(0,100)
  sps.set_xlim(0,meta.maxiters+meta.miniters)
  for i in meta.initnos:
    sps.scatter([0],[-1],c=meta.colors[i],label=meta.init_names[i],linewidths=0.1)
  sps.legend(fontsize = 'small',loc='upper right')
    #print 'done'
  return(f,sps)

def plottimes(testinfo,sps):

    (meta,p_run,s_run,n_run,i_run,r) = testinfo

    yfile = open(os.path.join(i_run.topdirs_o[0],meta.filenames[r]),'r')
    plot_y = hkl.load(yfile).T
    yfile.close()

    varfile = open(i_run.fitness[0],'rb')
    [[tot_var_y],each_var_y] = cPickle.load(varfile)
    varfile.close()
    var_y = statistics.variance(plot_y)
    tot_var_y = tot_var_y+var_y/meta.ntimes
    each_var_y.append(var_y)
    varfile = open(i_run.fitness[0],'wb')
    cPickle.dump([[tot_var_y],each_var_y],varfile)
    varfile.close()

    sps.scatter([meta.iternos[r]]*meta.params[p_run.p],plot_y,c=meta.colors[i_run.i],alpha=a_times,linewidths=0.1,s=meta.params[p_run.p],rasterized=True)
    return

#prepare to plot acceptance fractions as function of iteration number, one plot per survey
def plotfracs_init(survinfo):

  (meta,p_run,s_run) = survinfo

  #print 'plotfracs_init'
  plt.rc('text', usetex=True)
  global a_fracs
  a_fracs = 1./p_run.ndims/meta.samps
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
  #if nsurvs == 1:
  #  sps_fracs = [sps_fracs]
  #for s in survnos:
  sps.set_title(str(meta.samps)+' Acceptance Fractions for '+str(s_run.seed)+' galaxies')
  sps.set_ylim(0,1)
  sps.set_xlim(0,meta.maxiters+meta.miniters)
  sps.set_ylabel('acceptance fraction')
  sps.set_xlabel('number of iterations')
  for i in meta.initnos:
      sps.scatter([0],[-1],c=meta.colors[i],label=meta.init_names[i],linewidths=0.1,s=p_run.ndims)
  sps.legend(fontsize = 'small')
    #print 'done'
  return(f,sps)

def plotfracs(testinfo,sps):

    (meta,p_run,s_run,n_run,i_run,r) = testinfo

    yfile = open(os.path.join(i_run.topdirs_o[1],meta.filenames[r]),'r')
    plot_y = hkl.load(yfile).T
    yfile.close()

    varfile = open(i_run.fitness[1],'rb')
    [[tot_var_y],each_var_y] = cPickle.load(varfile)
    varfile.close()
    var_y = statistics.variance(plot_y)
    tot_var_y = tot_var_y+var_y/meta.ntimes
    each_var_y.append(var_y)
    varfile = open(i_run.fitness[1],'wb')
    cPickle.dump([[tot_var_y],each_var_y],varfile)
    varfile.close()

    #for i in walknos:
    sps.scatter([meta.iternos[r]]*n_run.nwalkers,plot_y,c=meta.colors[i_run.i],alpha=a_fracs,linewidths=0.1,s=p_run.ndims,rasterized=True)
    return

#prepare to plot lnprobs as function of iteration number, one plot per survey
def plotprobs_init(survinfo):

  (meta,p_run,s_run) = survinfo

  #print 'plotprobs_init'
  plt.rc('text', usetex=True)
  global a_probs
  a_probs = 1./meta.ncolors/meta.samps
  #f_probs,sps_probs = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5),sharex=True,sharey=True)#one subplot per sample
  f = plt.figure(figsize=(5,5))
  sps = f.add_subplot(1,1,1)
#  if nsurvs == 1:
#    sps_probs = [sps_probs]
#  for s in survnos:
 # sps.set_ylim(-25,0)
  sps.set_title(r'Sample Probability Evolution for $J_{0}='+str(s_run.seed)+r'$')
  sps.set_ylabel('log probability of walker')
  sps.set_xlabel('iteration number')
  sps.set_xlim(0,meta.maxiters)
  dummy_rec = [0]*meta.nsteps
  for i in meta.initnos:
      sps.plot([-1.]*meta.nsteps,dummy_rec,c=meta.colors[i],label=meta.init_names[i])
  sps.legend(fontsize='x-small',loc='lower right')
    #print 'done'
  return(f,sps)

def plotprobs(testinfo,sps):

    (meta,p_run,s_run,n_run,i_run,r) = testinfo

    yfile = open(os.path.join(i_run.topdirs_o[2],meta.filenames[r]),'r')
    plot_y = np.swapaxes(hkl.load(yfile),0,1).T#hkl.load(yfile).T
    yfile.close()

    varfile = open(i_run.fitness[2],'rb')
    [[tot_var_y],each_var_y] = cPickle.load(varfile)
    varfile.close()
    var_y = sum([statistics.variance(plot_y[w])/n_run.nwalkers for w in n_run.walknos])
    tot_var_y = tot_var_y+var_y/meta.ntimes
    each_var_y.append(var_y)
    varfile = open(i_run.fitness[2],'wb')
    cPickle.dump([[tot_var_y],each_var_y],varfile)
    varfile.close()

    #randwalks = random.sample(n_run.walknos,meta.ncolors)
    for w in n_run.walknos:#randwalks:
        sps.plot(meta.eachtimenos[r],plot_y[w],c=meta.colors[i_run.i],alpha=a_probs,rasterized=True)
    return

#plot accepted N(z) aggregate and as a function of iteration number#one plot per survey

def plotchains_init(initinfo):

  (meta,p_run,s_run,n_runs,i_runs) = initinfo

  #print 'plotchains_init'
  plt.rc('text', usetex=True)
  global a_chain
  global a_samp
  a_samp = 1./meta.ninits/meta.ncolors#nwalkers
  a_chain = 1./meta.ninits/meta.ncolors#nwalkers#/ntests#/howmany*miniters
  #prepare to plot what some of the samples look like
  f_samps = plt.figure(figsize=(5*meta.samps,5*2))#,sps_samps = plt.subplots(2,meta.samps,figsize =(5*meta.samps,5*2),sharex=True)#one subplot per sample
  gs_samps = matplotlib.gridspec.GridSpec(2,meta.samps)
  sps_samps = [[f_samps.add_subplot(gs_samps[l,n]) for n in meta.sampnos] for l in range(0,2)]
  #prepare to plot evolution of chains
  maxk = max([n_runs[n].nbins for n in meta.sampnos])
  f_chains = plt.figure(figsize=(5*maxk,5*meta.samps))
  gs_chains = matplotlib.gridspec.GridSpec(meta.samps,maxk)
  sps_chains = [[f_chains.add_subplot(gs_chains[n,k]) for k in range(0,maxk)] for n in meta.sampnos]
  #for n in meta.sampnos:
  #    for k in n_runs[n].new_binnos:
  #f_chains, sps_chains = plt.subplots(meta.nsamps,n_run.new_nbins, figsize=(5*ndim,5*meta.nsamps),sharey='row',sharex=True)#one subplot per sample per parameter
  dummy_chain = [-1.]*meta.nsteps
  ##plot a few random samples
  #f_rando, sps_rando = plt.subplots(ntests, nsurvs, figsize = (5*nsurvs,5*ntests))
  #t = 0
  for n in meta.sampnos:
    sps_samps[0][n].set_ylim(-1.,m.log(s_run.seed/meta.zdif)+1.)
    sps_samps[0][n].set_xlabel(r'$z$')
    sps_samps[0][n].set_ylabel(r'$\ln N(z)$')
    sps_samps[0][n].set_title(str(n+1)+r' Sampled $\ln N(z)$ for $J_{0}='+str(s_run.seed)+r'$')
    sps_samps[1][n].set_ylim(0.,s_run.seed/meta.zdif+s_run.seed)
    sps_samps[1][n].set_xlabel(r'$z$')
    sps_samps[1][n].set_ylabel(r'$N(z)$')
    sps_samps[1][n].set_title(str(n+1)+r' Sampled $N(z)$ for $J_{0}='+str(s_run.seed)+r'$')
    #for t in testnos:
    #    sps_samps[0][s].hlines(dummy_lnsamp,binlos,binhis,color=colors[t],label=setups[t])
    #    sps_samps[1][s].hlines(dummy_samp,binlos,binhis,color=colors[t],label=setups[t])
    maxn = max(n_runs[n].binnos for n in meta.sampnos)
    for k in n_runs[n].binnos:
        #sps_chains[n][k] = f_chains.add_subplot(meta.samps,maxk,n*maxk+k)
        sps_chains[n][k].set_ylim(-m.log(s_run.seed),m.log(s_run.seed/meta.zdif)+1.)#n_runs[n].full_logsampNz[k]+m.log(s_run.seed/meta.zdif))
        sps_chains[n][k].set_xlim(0,meta.maxiters)
        sps_chains[n][k].set_xlabel('iteration number')
        sps_chains[n][k].set_ylabel(r'$\ln N_{'+str(k+1)+'}(z)$')
        sps_chains[n][k].set_title(str(s_run.seed)+r' galaxies: Parameter '+str(k+1)+' of '+str(n_runs[n].nbins))
        for i in meta.initnos:
            varfile = open(i_runs[n][i].fitness[3],'wb')
            cPickle.dump([[0.,0.],[[],[]]],varfile)
            varfile.close()

            sps_chains[n][k].plot([-1.]*meta.nsteps,dummy_chain,color=meta.colors[i],label=meta.init_names[i])
#            else:
#             for t in testnos:
#               sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color=colors[t])
#             sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,linewidth=2,color='k')
        sps_chains[n][k].legend(fontsize='small',loc='upper right')
        #print 'done;'
  return((f_samps,sps_samps),(f_chains,sps_chains))#,(f_rando,sps_rando))

def plotchains(testinfo,sps_samps,sps_chains):#,sps_rando):

    (meta,p_run,s_run,n_run,i_run,r) = testinfo

  #if r!=0:
    yfile = open(os.path.join(i_run.topdirs_o[3],meta.filenames[r]),'r')
    plot_y_prep = hkl.load(yfile)
    plot_y_ls = np.swapaxes(plot_y_prep,0,1)
    plot_y_s = np.exp(plot_y_ls)
    plot_y_c = plot_y_ls.T
    yfile.close()

    #randiters = random.sample(meta.timenos,meta.ncolors)
    randwalks = random.sample(n_run.walknos,meta.ncolors)
    #for w in randsamps:
    #    for x in randiters:
    #      sps_rando[s][t].hlines(plot_y_s[x][w],binlos,binhis,rasterized=True)

    for x in meta.timenos:#randiters:#range_iters_each:
      for w in randwalks:#walknos:
          #fit_ls = np.dot((plot_y_ls[x][w]-full_logsampNz[s][n]),(plot_y_ls[x][w]-full_logsampNz[s][n]))
          #fit_s = np.dot((plot_y_s[x][w]-full_sampNz[s][n]),(plot_y_s[x][w]-full_sampNz[s][n]))
          sps_samps[0][n_run.n].hlines(plot_y_ls[x][w],n_run.binlos,n_run.binhis,color=meta.colors[i_run.i],alpha=a_samp,rasterized=True)
          sps_samps[1][n_run.n].hlines(plot_y_s[x][w],n_run.binlos,n_run.binhis,color=meta.colors[i_run.i],alpha=a_samp,rasterized=True)
          for k in n_run.binnos:
            sps_chains[n_run.n][k].plot(meta.eachtimenos[r],plot_y_c[k][w],color=meta.colors[i_run.i],alpha=a_chain,rasterized=True)

    varfile = open(i_run.fitness[3],'rb')
    [[tot_ls,tot_s],[each_ls,each_s]] = cPickle.load(varfile)
    varfile.close()
    var_ls = sum([sum([statistics.pvariance(plot_y_c[k][w],mu=n_run.full_logsampNz[k])/n_run.nbins for k in n_run.binnos])/n_run.nwalkers for w in n_run.walknos])
    var_s = sum([sum([statistics.pvariance(np.exp(plot_y_c[k][w]),mu=n_run.full_sampNz[k])/n_run.nbins for k in n_run.binnos])/n_run.nwalkers for w in n_run.walknos])
    tot_ls = tot_ls+var_ls/meta.ntimes
    tot_s = tot_s+var_s/meta.ntimes
    each_ls.append(var_ls)
    each_s.append(var_s)
    varfile = open(i_run.fitness[3],'wb')
    cPickle.dump([[tot_ls,tot_s],[each_ls,each_s]],varfile)
    varfile.close()

#     fit_ls_prep = plot_y_ls-n_run.full_logsampNz
#     fit_s_prep = plot_y_s-n_run.full_sampNz
#     fittest = open(i_run.fitness,'rb')
#     [fit_ls,fit_s] = cPickle.load(fittest)
#     fittest.close()
#     for w in n_run.walknos:
#         f_ls,f_s = 0.,0.
#         for x in meta.timenos:#eachiternos[r]:
#             f_ls += np.dot(fit_ls_prep[x][w],fit_ls_prep[x][w])#plot_y_ls[x][w]-full_logsampNz[s][n],plot_y_ls[x][w]-full_logsampNz[s][n])
#             f_s += np.dot(fit_s_prep[x][w],fit_s_prep[x][w])#plot_y_s[x][w]-full_sampNz[s][n],plot_y_s[x][w]-full_sampNz[s][n])
#         fit_ls += f_ls
#         fit_s += f_s
#     fit_ls = fit_ls/meta.ntimes/n_run.nwalkers
#     fit_s = fit_s/meta.ntimes/n_run.nwalkers
#     fittest = open(i_run.fitness,'wb')
#     cPickle.dump([fit_ls,fit_s],fittest)
#     fittest.close()
    return

def plotchains_wrapup(allinfo,(sampinfo,chaininfo)):#,randinfo):
  (meta,p_run,s_run,n_runs,i_runs) = allinfo
  sps_samps = sampinfo[1]
  sps_chains = chaininfo[1]
  #sps_rando = randinfo[1]
  for n in meta.sampnos:
      n_run = n_runs[n]
      dummy_lnsamp = [-1.]*n_run.nbins
      dummy_samp = [0.]*n_run.nbins
      sps_samps[0][n].step(p_run.zhis,s_run.logflatNz,color='k',alpha=0.5,label=r'Flat $\ln N(z)$',linewidth=2,where='post',linestyle='--')
      sps_samps[1][n].step(p_run.zhis,s_run.flatNz,color='k',alpha=0.5,label=r'Flat $N(z)$',linewidth=2,where='post',linestyle='--')
      #sps_samps[0][n].hlines(n_run.full_logflatNz,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2,linestyle='--',label=r'flat $\ln N(z)$')#,linewidth=2,where='post',linestyle='--')
      #sps_samps[1][n].hlines(n_run.full_flatNz,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2,linestyle='--',label=r'flat $N(z)$')#,linewidth=2,where='post',linestyle='--')
#       for t in testnos:
#           sps_samps[0][s].hlines(dummy_lnsamp,binlos,binhis,color=colors[t],label=setups[t])#+r'$\sigma^{2}='+str(sampvar_l)+r'$')
#           sps_samps[1][s].hlines(dummy_samp,binlos,binhis,color=colors[t],label=setups[t])#+r'$\sigma^{2}='+str(sampvar_s)+r'$')
      logdifstack,difstack = n_run.logstack-n_run.full_logsampNz,n_run.stack-n_run.full_sampNz
      logvarstack,varstack = np.dot(logdifstack,logdifstack)/n_run.nbins,np.dot(difstack,difstack)/n_run.nbins
      #varlogstack = sum([statistics.pvariance([n_run.logstack[k]],mu=n_run.full_logsampNz[k]) for k in n_run.binnos])/n_run.nbins
      sps_samps[0][n].step(n_run.binhis,n_run.logstack,color='k',alpha=0.5,linewidth=2,where='post',label=r'Stacked $\ln N(z)$ $\sigma^{2}='+str(logvarstack)+r'$')#,linewidth=2,where='post')
      sps_samps[1][n].step(n_run.binhis,n_run.stack,color='k',alpha=0.5,linewidth=2,where='post',label=r'Stacked $N(z)$ $\sigma^{2}='+str(varstack)+r'$')#,linewidth=2,where='post')
      #sps_samps[0][n].hlines(n_run.logstack,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2)
      #sps_samps[1][n].hlines(n_run.stack,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2)
      #sps_samps[0][n].hlines(n_run.logstack,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2,label=r'Stacked $\ln N(z)$: $\sigma^{2}='+str(round(logvarstack))+r'$')
      #sps_samps[1][n].hlines(n_run.stack,n_run.binlos,n_run.binhis,color='k',alpha=0.5,linewidth=2,label=r'Stacked $N(z)$: $\sigma^{2}='+str(round(varstack))+r'$')
#          for t in range(0,ntests):
#              sps_rando[s][t].hlines(logsampNz[s][n],binlos,binhis,color='k',alpha=0.5,label=r'sample $\ln N(z)$',linewidth=2)
#              sps_rando[s][t].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2,linestyle='-')
      sps_samps[0][n].step(n_run.binhis,n_run.full_logsampNz,color='k',label=r'True $\ln N(z)$',linewidth=2,where='post')
      sps_samps[1][n].step(n_run.binhis,n_run.full_sampNz,color='k',label=r'True $N(z)$',linewidth=2,where='post')
      #sps_samps[0][n].hlines(n_run.full_logsampNz,n_run.binlos,n_run.binhis,color='k',linewidth=2)
      #sps_samps[1][n].hlines(n_run.full_sampNz,n_run.binlos,n_run.binhis,color='k',linewidth=2)
      #sps_samps[0][n].hlines(n_run.full_logsampNz,n_run.binlos,n_run.binhis,color='k',linewidth=2,label=r'True $\ln N(z)$')
      #sps_samps[1][n].hlines(n_run.full_sampNz,n_run.binlos,n_run.binhis,color='k',linewidth=2,label=r'True $N(z)$')
              #sps_samps[s].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2)
      for i in meta.initnos:
          i_run = i_runs[n][i]
          varfile = open(i_run.fitness[3],'rb')
          [[tot_ls,tot_s],[each_ls,each_s]] = cPickle.load(varfile)
          varfile.close()

          #varfile = open(i_run.fitness,'rb')
          #   #print(cPickle.load(varfile))
          #   #pdb.set_trace()
          #[sampvar_l,sampvar_s] = cPickle.load(varfile)#varfile.read().T
          #varfile.close()
             #sampvar_l = sum(sampvar_l)
             #sampvar_s = sum(sampvar_s)
          sps_samps[0][n].hlines(dummy_lnsamp,n_run.binlos,n_run.binhis,color=meta.colors[i],label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_ls)+r'$')#str(sampvar_l)+r'$')
          sps_samps[1][n].hlines(dummy_samp,n_run.binlos,n_run.binhis,color=meta.colors[i],label=meta.init_names[i]+'\n'+r'$\sigma^{2}='+str(tot_s)+r'$')
      for k in n_run.binnos:
          sps_chains[n][k].step(meta.iternos,[n_run.full_logflatNz[k]]*meta.nsteps,color='k',label='Flat value',linestyle='--')
          #for n in sampnos:
          sps_chains[n][k].plot(meta.iternos,[n_run.full_logsampNz[k]]*meta.nsteps,color='k',linewidth=2,label='True value')# '+str(n+1))
          sps_chains[n][k].plot(meta.iternos,[n_run.logstack[k]]*meta.nsteps,color='k',alpha=0.5,linewidth=2,label='Stacked value')# '+str(n+1))
          sps_chains[n][k].legend(fontsize='xx-small',loc='lower right')
      sps_samps[0][n].legend(fontsize='xx-small',loc='upper left')
      sps_samps[1][n].legend(fontsize='xx-small',loc='upper left')
  return

#plotnames = ['acorr.pdf','fracs.pdf','lnprobs.pdf','results.pdf','compare.pdf']#,'rand_results.pdf']

#initialize each plot
def plots_setup(allinfo,q):
    if q != 3:
      allinfo = allinfo[:-2]
      #(meta,p_run,s_run) = allinfo
      if q == 0:
        info = plottimes_init(allinfo)
      if q == 1:
        info = plotfracs_init(allinfo)
      if q == 2:
        info = plotprobs_init(allinfo)
    if q == 3:
      #(meta,p_run,s_run,n_runs) = allinfo
      allinfo = allinfo#[:-1]
      info = plotchains_init(allinfo)
    return(info)

#plot one output file
def plotone(testinfo,q,info):
    start_time = timeit.default_timer()
    (meta,p_run,s_run,n_run,i_run,r) = testinfo
    if q == 0:
        plottimes(testinfo,info[1])
    if q == 1:
        plotfracs(testinfo,info[1])
    if q == 2:
        plotprobs(testinfo,info[1])
    if q == 3:
        plotchains(testinfo,info[0][1],info[1][1])#,info[2][1])
    elapsed = timeit.default_timer() - start_time
    plottimer = open(meta.plottime,'a')
    plottimer.write(str(timeit.default_timer())+' '+str(q)+': '+str((p_run.p,s_run.s,n_run.n,i_run.i,r))+' '+str(elapsed)+'\n')
    plottimer.close()
    return

#save each plot
def plots_wrapup(allinfo,q,info):
    #pr = cProfile.Profile()
    #pr.enable()
    if q == 3:
        plotchains_wrapup(allinfo,(info[0],info[1]))#,info[2])
        (meta,p_run,s_run,n_runs,i_runs) = allinfo
        info[0][0].savefig(os.path.join(s_run.topdir_s,meta.plotnames[q+1]),dpi=100)
        info[1][0].savefig(os.path.join(s_run.topdir_s,meta.plotnames[q]),dpi=100)
        #info[2][0].savefig(os.path.join(topdir,plotnames[q+2]),dpi=100)
    #    printout = plotnames[q]+' and '+plotnames[q+1]
    else:
        (meta,p_run,s_run) = allinfo[:-2]
        info[0].savefig(os.path.join(s_run.topdir_s,meta.plotnames[q]),dpi=100)
    #    printout = plotnames[q]
    #pr.disable()
    #sout = StringIO.StringIO()
    #sortby = 'tottime'
    #ps = pstats.Stats(pr, stream=sout).sort_stats(sortby)
    #ps.print_stats()
    #print(printout)
    #print(sout.getvalue())
    return
