import pdb
#pdb.set_trace()
import matplotlib
matplotlib.use('pdf')
import timeit
import sys
import setup
import os
#import cProfile
#import pstats

#from setup import *
import plots
#import matplotlib.pyplot as plt
from setup import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

def plot_inits():
  print 'plot_inits '+str(os.getpid())
  #print plt.get_backend()
  start_time = timeit.default_timer()
  #setup_plots.plot_priorgen()
  #setup_plots.plot_true()
  #setup_plots.plot_obs()
  setup_plots.plot_lfs()
  setup_plots.plot_prior_samps()
  setup_plots.plot_ivals()
  elapsed = timeit.default_timer() - start_time
  print('iplots '+str(elapsed))
  return

import multiprocessing as mp

# render the initial plots, in a child process so OSX doesn't flip a shit.
#print 'initial plotting...'
init_plot_proc = mp.Process(target=plot_inits)
print 'parent: ' + str(os.getpid())
init_plot_proc.start()
# wait for them to finish, this is entirely optional.
init_plot_proc.join()
#print 'initial plotting done'
#sys.exit();

start_time = timeit.default_timer()
nq = nstats#number of plot processes
qnos = range(0,nq)
queues=[mp.Queue() for q in qnos]

#sample miniters of one draw of one survey
def sampling(sampler,ivals):
    start_time = timeit.default_timer()
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(ivals,miniters,thin=howmany)
    ivals = [walk[-1] for walk in sampler.chain]
    times = sampler.get_autocorr_time()#ndims
    fracs = sampler.acceptance_fraction#nwalkers
    probs = sampler.lnprobability#np.swapaxes(sampler.lnprobability,0,1)#niters*nwalkers
    chains = sampler.chain#np.swapaxes(sampler.chain,0,1)#niters*nwalkers*ndims
    elapsed = timeit.default_timer() - start_time
    #print(setups[n]+' '+str(ndims[k])+' complete')
    return ivals,[times,fracs,probs,chains],elapsed

#sample maxiters of one draw of one survey
def samplings((s,n,t)):
    ivals = iguesses[s][t]
    sampler = samplers[s][n]
    for r in runnos:
        ivals,outputs,elapsed = sampling(sampler,ivals)
        for i in statnos:
            outfile = open(outnames[s][n][t][i][r],'w')
            hkl.dump(outputs[i],outfile,mode='w')
            outfile.close()
            queues[i].put((s,n,t,r))
        calctimer = open(calctime,'a')
        calctimer.write(str(timeit.default_timer())+' '+str((s,n,t,r))+' '+str(elapsed)+'\n')
        calctimer.close()
    return

def fsamp(*args):
    return samplings(*args)

#prepare to plot autocorrelation times as function of iteration number
def plottimes_init():
  print 'plottimes_init'
  plt.rc('text', usetex=True)
  f_times,sps_times = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5))#one plot per survey
  global a_times
  a_times = 2./ndims[lenno]/nsamps
  if nsurvs == 1:
    sps_times = [sps_times]
  for s in survnos:
    sps_times[s].set_title(str(nsamps)+' Autocorrelation Times for '+str(seed_ngals[s])+' galaxies')
    sps_times[s].set_ylabel('autocorrelation time')
    sps_times[s].set_xlabel('number of iterations')
    sps_times[s].set_ylim(0,100)
    sps_times[s].set_xlim(0,maxiters+miniters)
    for t in testnos:
        sps_times[s].scatter([0],[-1],c=colors[t],label=setups[t],linewidths=0.1,s=ndim)
    sps_times[s].legend(fontsize = 'small',loc='upper right')
    #print 'done'
  return(f_times,sps_times)

def plottimes((s,n,t,r),sps_times):
    yfile = open(os.path.join(outpaths[s][n][t][0],filenames[r]),'r')
    plot_y = hkl.load(yfile).T
    sps_times[s].scatter([plot_iters[r]]*ndims[lenno],plot_y,c=colors[t],alpha=a_times,linewidths=0.1,s=ndim,rasterized=True)
    yfile.close()
    return

#prepare to plot acceptance fractions as function of iteration number
def plotfracs_init():
  print 'plotfracs_init'
  plt.rc('text', usetex=True)
  global a_fracs
  a_fracs = 1./ndims[lenno]/nsamps
  f_fracs,sps_fracs = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5))#one plot per survey
  if nsurvs == 1:
    sps_fracs = [sps_fracs]
  for s in survnos:
    sps_fracs[s].set_title(str(nsamps)+' Acceptance Fractions for '+str(seed_ngals[s])+' galaxies')
    sps_fracs[s].set_ylim(0,1)
    sps_fracs[s].set_xlim(0,maxiters+miniters)
    sps_fracs[s].set_ylabel('acceptance fraction')
    sps_fracs[s].set_xlabel('number of iterations')
    for t in testnos:
        sps_fracs[s].scatter([0],[-1],c=colors[t],label=setups[t],linewidths=0.1,s=ndim)
    sps_fracs[s].legend(fontsize = 'small')
    #print 'done'
  return(f_fracs,sps_fracs)

def plotfracs((s,n,t,r),sps_fracs):
    yfile = open(os.path.join(outpaths[s][n][t][1],filenames[r]),'r')
    plot_y = hkl.load(yfile).T
    yfile.close()
    #for i in walknos:
    sps_fracs[s].scatter([plot_iters[r]]*nwalkers,plot_y,c=colors[t],alpha=a_fracs,linewidths=0.1,s=ndim,rasterized=True)
    return

#prepare to plot lnprobs as function of iteration number
def plotprobs_init():
  print 'plotprobs_init'
  plt.rc('text', usetex=True)
  global a_probs
  a_probs = 1./ncolors/nsamps
  f_probs,sps_probs = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5),sharex=True,sharey=True)#one plot per survey
  if nsurvs == 1:
    sps_probs = [sps_probs]
  for s in survnos:
    sps_probs[s].set_ylim(-100,0)
    sps_probs[s].set_title(r'Sample Probability Evolution for $J_{0}='+str(seed_ngals[s])+r'$')
    sps_probs[s].set_ylabel('log probability of walker')
    sps_probs[s].set_xlabel('iteration number')
    for t in testnos:
        sps_probs[s].plot(plot_iters_all,[0]*iters_all,c=colors[t],label=setups[t])
    sps_probs[s].legend(fontsize='small',loc='lower right')
    #print 'done'
  return(f_probs,sps_probs)

def plotprobs((s,n,t,r),sps_probs):
    yfile = open(os.path.join(outpaths[s][n][t][2],filenames[r]),'r')
    plot_y = np.swapaxes(hkl.load(yfile),0,1).T#hkl.load(yfile).T
    yfile.close()
    randwalks = random.sample(walknos,ncolors)
    for w in randwalks:
        sps_probs[s].plot(plot_iters_ranges[r],plot_y[w],c=colors[t],alpha=a_probs,rasterized=True)
    return

def plotchains_init():
  print 'plotchains_init'
  plt.rc('text', usetex=True)
  global a_chain
  global a_samp
  a_samp = 1./nsamps/ncolors#ncolors#nwalkers
  a_chain = 1./nsamps/ntests#nwalkers#/ntests#/howmany*miniters
  #prepare to plot what some of the samples look like
  f_samps,sps_samps = plt.subplots(2, nsurvs, figsize = (5*nsurvs,5*2),sharex=True)#one plot per sample per survey
  #prepare to plot evolution of chains
  f_chains, sps_chains = plt.subplots(nsurvs, ndim, figsize=(5*ndim,5*nsurvs),sharey='row',sharex=True)
  dummy_chain = [-1.]*iters_all
  ##plot a few random samples
  #f_rando, sps_rando = plt.subplots(ntests, nsurvs, figsize = (5*nsurvs,5*ntests))
  for s in survnos:
    sps_samps[0][s].set_ylim(-1.,ymax[s]+1.)
    sps_samps[0][s].set_xlabel(r'$z$')
    sps_samps[0][s].set_ylabel(r'$\ln N(z)$')
    sps_samps[0][s].set_title(r'Sampled $\ln N(z)$ for $J_{0}='+str(seed_ngals[s])+r'$')
    sps_samps[1][s].set_ylim(0.,seed_ngals[s]/zdif)
    sps_samps[1][s].set_xlabel(r'$z$')
    sps_samps[1][s].set_ylabel(r'$N(z)$')
    sps_samps[1][s].set_title(r'Sampled $N(z)$ for $J_{0}='+str(seed_ngals[s])+r'$')
    #for t in testnos:
    #    sps_samps[0][s].hlines(dummy_lnsamp,binlos,binhis,color=colors[t],label=setups[t])
    #    sps_samps[1][s].hlines(dummy_samp,binlos,binhis,color=colors[t],label=setups[t])
    for k in dimnos:
        sps_chains[s][k].set_ylim(-1.,ymax[s]+1.)
        sps_chains[s][k].set_xlabel('iteration number')
        sps_chains[s][k].set_ylabel(r'$\ln N_{'+str(k+1)+'}(z)$')
        sps_chains[s][k].set_title(str(seed_ngals[s])+r' galaxies: Parameter '+str(k+1)+' of '+str(ndims[lenno]))
        for n in sampnos:
          if n == 0:
            for t in testnos:
              sps_chains[s][k].plot(plot_iters_all,dummy_chain,color=colors[t],label=setups[t])
#           else:
#             for t in testnos:
#               sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color=colors[t])
#             sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,linewidth=2,color='k')
        sps_chains[s][k].legend(fontsize='small',loc='upper right')
        #print 'done;'
  return((f_samps,sps_samps),(f_chains,sps_chains))#,(f_rando,sps_rando))

def plotchains((s,n,t,r),sps_samps,sps_chains):#,sps_rando):
  #if r!=0:
    yfile = open(os.path.join(outpaths[s][n][t][3],filenames[r]),'r')
    plot_y_prep = hkl.load(yfile)
    plot_y_ls = np.swapaxes(plot_y_prep,0,1)
    plot_y_s = np.exp(plot_y_ls)
    plot_y_c = plot_y_s.T
    yfile.close()
    randiters = random.sample(range_iters_each,ncolors)
    randwalks = random.sample(walknos,ncolors)
    #for w in randsamps:
    #    for x in randiters:
    #      sps_rando[s][t].hlines(plot_y_s[x][w],binlos,binhis,rasterized=True)
    for w in randwalks:#walknos:
        for x in randiters:#range_iters_each:
            #fit_ls = np.dot((plot_y_ls[x][w]-full_logsampNz[s][n]),(plot_y_ls[x][w]-full_logsampNz[s][n]))
            #fit_s = np.dot((plot_y_s[x][w]-full_sampNz[s][n]),(plot_y_s[x][w]-full_sampNz[s][n]))
            sps_samps[0][s].hlines(plot_y_ls[x][w],binlos,binhis,color=colors[t],alpha=a_samp,rasterized=True)
            sps_samps[1][s].hlines(plot_y_s[x][w],binlos,binhis,color=colors[t],alpha=a_samp,rasterized=True)
        for k in dimnos:
            sps_chains[s][k].plot(plot_iters_ranges[r],plot_y_c[k][w],color=colors[t],alpha=a_chain,rasterized=True)
    fit_ls_prep = plot_y_ls-full_logsampNz[s][n]
    fit_s_prep = plot_y_s-full_sampNz[s][n]
    fittest = open(fitness[s][n][t],'rb')
    [fit_ls,fit_s] = cPickle.load(fittest)
    fittest.close()
    for w in walknos:
        f_ls,f_s = 0.,0.
        for x in range_iters_each:
            f_ls += np.dot(fit_ls_prep[x][w],fit_ls_prep[x][w])#plot_y_ls[x][w]-full_logsampNz[s][n],plot_y_ls[x][w]-full_logsampNz[s][n])
            f_s += np.dot(fit_s_prep[x][w],fit_s_prep[x][w])#plot_y_s[x][w]-full_sampNz[s][n],plot_y_s[x][w]-full_sampNz[s][n])
        fit_ls += f_ls
        fit_s += f_s
    fit_ls = fit_ls/iters_each/nwalkers
    fit_s = fit_s/iters_each/nwalkers
    fittest = open(fitness[s][n][t],'wb')
    cPickle.dump([fit_ls,fit_s],fittest)
    fittest.close()
    return

def plotchains_wrapup(sampinfo,chaininfo):#,randinfo):
  sps_samps = sampinfo[1]
  sps_chains = chaininfo[1]
  #sps_rando = randinfo[1]
  dummy_lnsamp = [-1.]*new_nbins
  dummy_samp = [0.]*new_nbins
  for s in survnos:
      sps_samps[0][s].step(binmids,full_logflatNz[s],color='k',alpha=0.5,label=r'flat $\ln N(z)$',linewidth=2,where='mid',linestyle='--')
      sps_samps[1][s].step(binmids,full_flatNz[s],color='k',alpha=0.5,label=r'flat $N(z)$',linewidth=2,where='mid',linestyle='--')
#       for t in testnos:
#           sps_samps[0][s].hlines(dummy_lnsamp,binlos,binhis,color=colors[t],label=setups[t])#+r'$\sigma^{2}='+str(sampvar_l)+r'$')
#           sps_samps[1][s].hlines(dummy_samp,binlos,binhis,color=colors[t],label=setups[t])#+r'$\sigma^{2}='+str(sampvar_s)+r'$')
      for n in sampnos:
          logdifsheldon,difsheldon = logsheldon[s][n]-full_logsampNz[s][n],sheldon[s][n]-full_sampNz[s][n]
          logvarsheldon,varsheldon = np.dot(logdifsheldon,logdifsheldon),np.dot(difsheldon,difsheldon)
          sps_samps[0][s].step(binmids,logsheldon[s][n],color='w',alpha=0.5,label=r'Sheldon $\ln N(z)$ '+str(n+1)+r' $\sigma^{2}='+str(int(logvarsheldon))+r'$',linewidth=2,where='mid')
          sps_samps[1][s].step(binmids,sheldon[s][n],color='w',alpha=0.5,label=r'Sheldon $N(z)$ '+str(n+1)+r' $\sigma^{2}='+str(int(varsheldon))+r'$',linewidth=2,where='mid')
          sps_samps[0][s].hlines(logsheldon[s][n],binlos,binhis,color='w',alpha=0.5,linewidth=2)
          sps_samps[1][s].hlines(sheldon[s][n],binlos,binhis,color='w',alpha=0.5,linewidth=2)
#          for t in range(0,ntests):
#              sps_rando[s][t].hlines(logsampNz[s][n],binlos,binhis,color='k',alpha=0.5,label=r'sample $\ln N(z)$',linewidth=2)
#              sps_rando[s][t].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2,linestyle='-')
          sps_samps[0][s].step(binmids,full_logsampNz[s][n],color='k',label=r'true $\ln N(z)$ '+str(n+1),linewidth=2,where='mid')
          sps_samps[1][s].step(binmids,full_sampNz[s][n],color='k',label=r'true $N(z)$ '+str(n+1),linewidth=2,where='mid')
          sps_samps[0][s].hlines(full_logsampNz[s][n],binlos,binhis,color='k',linewidth=2)
          sps_samps[1][s].hlines(full_sampNz[s][n],binlos,binhis,color='k',linewidth=2)
              #sps_samps[s].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2)
          for t in testnos:
             varfile = open(fitness[s][n][t],'rb')
             #print(cPickle.load(varfile))
             #pdb.set_trace()
             [sampvar_l,sampvar_s] = cPickle.load(varfile)#varfile.read().T
             varfile.close()
             #sampvar_l = sum(sampvar_l)
             #sampvar_s = sum(sampvar_s)
             sps_samps[0][s].hlines(dummy_lnsamp,binlos,binhis,color=colors[t],label=setups[t]+'\n'+r'$\sigma^{2}='+str(int(sampvar_l))+r'$')
             sps_samps[1][s].hlines(dummy_samp,binlos,binhis,color=colors[t],label=setups[t]+'\n'+r'$\sigma^{2}='+str(int(sampvar_s))+r'$')
      for k in dimnos:
          sps_chains[s][k].step(plot_iters_all,[full_logflatNz[s][k]]*iters_all,color='k',label='flat value',linestyle='--')
          for n in sampnos:
              sps_chains[s][k].plot(plot_iters_all,[full_logsampNz[s][n][k]]*iters_all,color='k',linewidth=2,label='true value '+str(n+1))
              sps_chains[s][k].plot(plot_iters_all,[logsheldon[s][n][k]]*iters_all,color='k',alpha=0.5,linewidth=2,label='Sheldon value '+str(n+1))
          sps_chains[s][k].legend(fontsize='xx-small',loc='lower right')
      sps_samps[0][s].legend(fontsize='xx-small',loc='upper left')
      sps_samps[1][s].legend(fontsize='xx-small',loc='upper left')
  return

plotnames = ['acorr.pdf','fracs.pdf','lnprobs.pdf','results.pdf','compare.pdf']#,'rand_results.pdf']

#initialize each plot
def plots_setup(q):
    if q == 0:
        info = plottimes_init()
    if q == 1:
        info = plotfracs_init()
    if q == 2:
        info = plotprobs_init()
    if q == 3:
        info = plotchains_init()
    return info

#plot one output file
def plotone((s,n,t,r),q,info):
    start_time = timeit.default_timer()
    if q == 0:
        plottimes((s,n,t,r),info[1])
    if q == 1:
        plotfracs((s,n,t,r),info[1])
    if q == 2:
        plotprobs((s,n,t,r),info[1])
    if q == 3:
        plotchains((s,n,t,r),info[0][1],info[1][1])#,info[2][1])
    elapsed = timeit.default_timer() - start_time
    plottimer = open(plottime,'a')
    plottimer.write(str(timeit.default_timer())+' '+str((s,n,t,r,q))+' '+str(elapsed)+'\n')
    plottimer.close()
    return

#save each plot
def plots_wrapup(q,info):
    #pr = cProfile.Profile()
    #pr.enable()
    if q == 3:
        plotchains_wrapup(info[0],info[1])#,info[2])
        info[0][0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
        info[1][0].savefig(os.path.join(topdir,plotnames[q+1]),dpi=100)
        #info[2][0].savefig(os.path.join(topdir,plotnames[q+2]),dpi=100)
    #    printout = plotnames[q]+' and '+plotnames[q+1]
    else:
        info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
    #    printout = plotnames[q]
    #pr.disable()
    #sout = StringIO.StringIO()
    #sortby = 'tottime'
    #ps = pstats.Stats(pr, stream=sout).sort_stats(sortby)
    #ps.print_stats()
    #print(printout)
    #print(sout.getvalue())
    return

#entire plotting process
def plotall(q):
#      plt.rc('text', usetex=True)
      plot_info = plots_setup(q)
      print('plot process started '+str(os.getpid()))
      while(True):
        vals = queues[q].get()
#         if (vals=='init'):
#             print('initializing plots now')
#             plot_info = plots_setup(q)
        if (vals=='done'):
            starttime = timeit.default_timer()
            print(str(os.getpid())+' saving '+plotnames[q]+' at '+str(starttime))
            plots_wrapup(q,plot_info)
            endtime = timeit.default_timer()
            print(str(os.getpid())+' saved '+plotnames[q]+' at '+str(endtime)+': '+str(endtime-starttime)+' elapsed')
            return
        else:
            (s,n,t,r) = vals
            plotone((s,n,t,r),q,plot_info)

def fplot(*args):
    return plotall(*args)


procs = [mp.Process(target=fplot,args=(q,)) for q in qnos]
params = [(s,n,t) for s in survnos for n in sampnos for t in testnos]# for r in runnos]

#multiprocessed sampler/plotter
for p in procs:
    p.start()

# for q in queues:
#     q.put('init')

#number of processors to use, leave one free for other things
nps = mp.cpu_count()-1
pool = mp.Pool(nps)
pool.map(fsamp, params)

for q in queues:
    q.put('done')

for p in procs:
    p.join()

elapsed = timeit.default_timer() - start_time
print(str(timeit.default_timer())+' mcmc complete: '+str(elapsed))
