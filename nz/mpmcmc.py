import matplotlib
matplotlib.use('pdf')
import timeit
import sys
import setup
import os

#from setup import *
import plots as setup_plots
#import matplotlib.pyplot as plt
from setup import *
import matplotlib.pyplot as plt

def plot_inits():
  print 'plot_inits'
  print os.getpid()
  print plt.get_backend()
  start_time = timeit.default_timer()
  # plt.rc('text', usetex=True)
  elapsed = timeit.default_timer() - start_time
  print('setup '+str(elapsed))
  start_time = timeit.default_timer()
  setup_plots.plot_priorgen()
  setup_plots.plot_samps()
  setup_plots.plot_obs()
  setup_plots.plot_randos()
  setup_plots.plot_prior_samps()
  setup_plots.plot_ivals()
  elapsed = timeit.default_timer() - start_time
  print('iplots '+str(elapsed))
  return

import multiprocessing as mp

# render the initial plots, in a child process so OSX doesn't flip a shit.
print 'initial plotting...'
init_plot_proc = mp.Process(target=plot_inits)
print 'parent: ' + str(os.getpid())
init_plot_proc.start()
# wait for them to finish, this is entirely optional.
init_plot_proc.join()
print 'initial plotting done'
#sys.exit();
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
    probs = np.swapaxes(sampler.lnprobability,0,1)#niters*nwalkers
    chains = np.swapaxes(sampler.chain,0,1)#niters*nwalkers*ndims
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
        for q in qnos:
            queues[q].put((s,n,t,r))
            calctimer = open(calctime,'a')
            calctimer.write(str(time.time())+' '+str((s,n,t,r,q))+' '+str(elapsed)+'\n')
            calctimer.close()
        #calctimer.write(str(time.time())+' '+str((s,t,n,r)))
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
  for s in survnos:
    sps_times[s].set_title('Autocorrelation Times for '+str(nsamps)+' runs of '+str(seed_ngals[s])+' galaxies')
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
    #plot_y = cPickle.load(yfile).T
    plot_y = hkl.load(yfile).T
    #for i in range(0,ndims[k]):
    sps_times[s].scatter([plot_iters[r]]*ndims[lenno],plot_y,c=colors[t],alpha=a_times,linewidths=0.1,s=ndim,rasterized=True)
    yfile.close()

#prepare to plot acceptance fractions as function of iteration number
def plotfracs_init():
  print 'plotfracs_init'
  plt.rc('text', usetex=True)
  global a_fracs
  a_fracs = 1./ndims[lenno]/nsamps
  f_fracs,sps_fracs = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5))#one plot per survey
  for s in survnos:
    sps_fracs[s].set_title('Acceptance Fractions for '+str(nsamps)+' runs of '+str(seed_ngals[s])+' galaxies')
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

#prepare to plot lnprobs as function of iteration number
def plotprobs_init():
  print 'plotprobs_init'
  plt.rc('text', usetex=True)
  global a_probs
  a_probs = 1./nwalkers/nsamps
  f_probs,sps_probs = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5))#one plot per survey
  for s in survnos:
    sps_probs[s].set_ylim(-100,0)
    sps_probs[s].set_title(str(seed_ngals[s])+' Galaxies')
    sps_probs[s].set_ylabel('log probability of walker')
    sps_probs[s].set_xlabel('iteration number')
    for t in testnos:
        sps_probs[s].plot(plot_iters_all,[0]*iters_all,c=colors[t],label=setups[t])
    sps_probs[s].legend(fontsize='x-small',loc='lower right')
    #print 'done'
  return(f_probs,sps_probs)

def plotprobs((s,n,t,r),sps_probs):
    #print 'sntr: ' + str((s,n,t,r))
    yfile = open(os.path.join(outpaths[s][n][t][2],filenames[r]),'r')
    plot_y = hkl.load(yfile).T
    yfile.close()
    for w in walknos:
        sps_probs[s].plot(plot_iters_ranges[r],plot_y[w],c=colors[t],alpha=a_probs,rasterized=True)

def plotchains_init():
  print 'plotchains_init'
  plt.rc('text', usetex=True)
  global a_chain
  global a_samp
  a_samp = 1./nsamps#nwalkers
  a_chain = 1./nsamps#nwalkers#/ntests#/howmany*miniters
  #prepare to plot what some of the samples look like
  f_samps,sps_samps = plt.subplots(1, nsurvs, figsize = (5*nsurvs,5*nsamps))#one plot per sample per survey
  #prepare to plot evolution of chains
  f_chains, sps_chains = plt.subplots(nsurvs, ndim, figsize=(5*ndim,5*nsurvs),sharey='col')
  ##plot a few random samples
  #f_rando, sps_rando = plt.subplots(ntests, nsurvs, figsize = (5*nsurvs,5*ntests))
  for s in survnos:
    sps_samps[s].set_ylim(np.log(seed_ngals[s])-5,np.log(seed_ngals[s])+5)
    sps_samps[s].set_xlabel(r'$z$')
    sps_samps[s].set_ylabel(r'$\ln N(z)$')
    sps_samps[s].set_title(r' Sampled $N(z)$ for $J='+str(seed_ngals[s])+r'$')
    #sps_rando[s].set_ylim(np.log(seed_ngals[s])-5,np.log(seed_ngals[s])+5)
    #sps_rando[s].set_xlabel(r'$z$')
    #sps_rando[s].set_ylabel(r'$\ln N(z)$')
    #sps_rando[s].set_title(r'Sampled $N(z)$ for $J='+str(seed_ngals[s])+r'$ with '+setups[t])
    for t in testnos:
        sps_samps[s].hlines(full_logflatNz[s],binlos,binhis,color=colors[t],label=setups[t])
    for k in dimnos:
        sps_chains[s][k].set_ylim([-5,5+np.log(seed_ngals[s])])
        sps_chains[s][k].set_xlabel('iteration number')
        sps_chains[s][k].set_ylabel(r'$\ln N_{'+str(k+1)+'}(z)$')
        sps_chains[s][k].set_title(r'$J='+str(seed_ngals[s])+r'$: Parameter '+str(k+1)+' of '+str(ndims[lenno]))
        for n in sampnos:
          if n == 0:
            sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color='k',label='true value')
            for t in testnos:
              sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color=colors[t],label=setups[t])
          else:
            sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color='k')
            for t in testnos:
              sps_chains[s][k].plot(plot_iters_all,[logobsNz[s][n][k]]*iters_all,color=colors[t])
        sps_chains[s][k].legend(fontsize='small',loc='lower right')
        #print 'done;'
  return((f_samps,sps_samps),(f_chains,sps_chains))#,(f_rando,sps_rando))

def plotchains((s,n,t,r),sps_samps,sps_chains):#,sps_rando):
    yfile = open(os.path.join(outpaths[s][n][t][3],filenames[r]),'r')
    plot_y_s = hkl.load(yfile)
    plot_y_c = plot_y_s.T
    yfile.close()
    randsamps = random.sample(walknos,ncolors)
    randiters = random.sample(range_iters_each,1)
    #for w in randsamps:
    #    for x in randiters:
    #      sps_rando[s][t].hlines(plot_y_s[x][w],binlos,binhis,rasterized=True)
    for w in randsamps:#walknos:
        if r > 0:
          for x in randiters:#runnos:#runnos
            sps_samps[s].hlines(plot_y_s[x][w],binlos,binhis,color=colors[t],alpha=a_samp,rasterized=True)
        for k in dimnos:
            sps_chains[s][k].plot(plot_iters_ranges[r],plot_y_c[k][w],color=colors[t],alpha=a_chain,rasterized=True)

def plotchains_wrapup(sampinfo):#,randinfo):
  sps_samps = sampinfo[1]
  #sps_rando = randinfo[1]
  for s in survnos:
      sps_samps[s].hlines(full_logflatNz[s],binlos,binhis,color='k',label=r'flat $\ln N(z)$',linewidth=2)
      for n in sampnos:
#          for t in range(0,ntests):
#              sps_rando[s][t].hlines(logsampNz[s][n],binlos,binhis,color='k',alpha=0.5,label=r'sample $\ln N(z)$',linewidth=2)
#              sps_rando[s][t].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2,linestyle='-')
          if n==0:
              sps_samps[s].hlines(logsampNz[s][n],binlos,binhis,color='k',alpha=0.5,label=r'sample $\ln N(z)$',linewidth=2)
              sps_samps[s].hlines(logobsNz[s][n],binlos,binhis,label=r'observed $\ln N(z)$',color='k',linewidth=2,linestyle='-')
          else:
              sps_samps[s].hlines(logobsNz[s][n],binlos,binhis,color='k',linewidth=2)
              sps_samps[s].hlines(logsampNz[s][n],binlos,binhis,color='k',linestyle='-',linewidth=2)
      sps_samps[s].legend(fontsize='xx-small',loc='lower left')

plotnames = ['acorr.pdf','fracs.pdf','lnprobs.pdf','results.pdf','compare.pdf']#,'rand_results.pdf']

#initialize each plot
def plots_setup(q):
    #time_info = None
    #frac_info = None
    #prob_info = None
    #samp_info = None
    #chain_info = None
    if q == 0:
        info = plottimes_init()#time_info = plottimes_init()
    if q == 1:
        info = plotfracs_init()#frac_info = plotfracs_init()
    if q == 2:
        info = plotprobs_init()#prob_info = plotprobs_init()
    if q == 3:
        info = plotchains_init()#samp_info,chain_info = plotchains_init()
    return info#(time_info,frac_info,prob_info,samp_info,chain_info)

#plot one output file
def plotone((s,n,t,r),q,info):#(time_info,frac_info,prob_info,samp_info,chain_info)):
    start_time = timeit.default_timer()
    if q == 0:
        plottimes((s,n,t,r),info[1])#time_info[1])
    if q == 1:
        plotfracs((s,n,t,r),info[1])#frac_info[1])
    if q == 2:
        plotprobs((s,n,t,r),info[1])#prob_info[1])
    if q == 3:
        plotchains((s,n,t,r),info[0][1],info[1][1])#,info[2][1])#samp_info[1],chain_info[1])
    elapsed = timeit.default_timer() - start_time
    plottimer = open(plottime,'a')
    plottimer.write(str(time.time())+' '+str((s,n,t,r,q))+' '+str(elapsed)+'\n')
    plottimer.close()
    return

#save each plot
def plots_wrapup(q,info):#(time_info,frac_info,prob_info,samp_info,chain_info)):
    if q == 3:
        plotchains_wrapup(info[0])#,info[2])#samp_info)
        info[0][0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)#samp_info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
        info[1][0].savefig(os.path.join(topdir,plotnames[q+1]),dpi=100)#chain_info[0].savefig(os.path.join(topdir,plotnames[q+1]),dpi=100)
        #info[2][0].savefig(os.path.join(topdir,plotnames[q+2]),dpi=100)
    else:
        info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
#     if q == 0:
#         time_info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
#     if q == 1:
#         frac_info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
#     if q == 2:
#         prob_info[0].savefig(os.path.join(topdir,plotnames[q]),dpi=100)
#     if q == 3:
    return

#entire plotting process
def plotall(q):
#     if q == -1:
#       plot_inits()
#       return
#     else:
      plt.rc('text', usetex=True)
      plot_info = plots_setup(q)#(time_info,frac_info,prob_info,samp_info,chain_info) = plots_setup(q)
      print('plot process started '+str(os.getpid()))
      while(True):
        vals = queues[q].get()
#         if (vals=='init'):
#             print('initializing plots now')
#             plot_info = plots_setup(q)#(time_info,frac_info,prob_info,samp_info,chain_info) = plots_setup(q)
        if (vals=='done'):
            plots_wrapup(q,plot_info)#(time_info,frac_info,prob_info,samp_info,chain_info))
            print('saving plots now')
            return
        else:
            (s,n,t,r) = vals
            plotone((s,n,t,r),q,plot_info)#(time_info,frac_info,prob_info,samp_info,chain_info))

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
