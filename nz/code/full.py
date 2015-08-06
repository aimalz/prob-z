import multiprocessing as mp

from setup import *
meta = setup()

from perparam import *
p_runs = {p:perparam(meta, p) for p in meta.paramnos}

from persurv import *
s_runs = {(p,s):persurv(meta,p_runs[(p)],s) for s in meta.survnos for p in p_runs.keys()}

from persamp import *
n_runs = {(p,s,n):persamp(meta,p_runs[(p)],s_runs[(p,s)],n) for n in meta.sampnos for (p,s) in s_runs.keys()}

from perinit import *
i_runs = {(p,s,n,i):perinit(meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i) for i in meta.initnos for (p,s,n) in n_runs.keys()}

from iplots import *
plot_inits_proc = mp.Process(target=iplots,args=(meta,p_runs,s_runs,n_runs,i_runs))
plot_inits_proc.start()
plot_inits_proc.join()

from permcmc import *

  #sample maxiters of one draw of one survey
def samplings(idinfo):

    (p,s,n,i) = idinfo
    p_run,s_run,n_run,i_run = p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i_runs[(p,s,n,i)]
    ivals = i_run.iguesses
    sampler = n_run.sampler

    for r in meta.stepnos:
      #runinfo = (meta,p_run,s_run,n_run,i_run,r)
      runinfo = (p,s,n,i,r)#(p_run.p,s_run.s,n_run.n,i_run.i,r)
      ivals,outputs,elapsed = sampling(n_run.sampler,ivals,meta.miniters,meta.thinto)

      for q in meta.statnos:
        outfile = open(os.path.join(i_run.topdir_i,meta.outnames[q][r]),'w')
        hkl.dump(outputs[q],outfile,mode='w')
        outfile.close()
        queues[q].put(runinfo)

      calctimer = open(meta.calctime,'a')
      calctimer.write(str(timeit.default_timer())+' '+str(runinfo)+' '+str(elapsed)+'\n')
      calctimer.close()

def fsamp(*args):
    return samplings(*args)

def postplot(idinfo):
    (p,s,n,i) = idinfo
    for r in meta.stepnos:
      #runinfo = (meta,p_run,s_run,n_run,i_run,r)
      runinfo = (p,s,n,i,r)#(p_run.p,s_run.s,n_run.n,i_run.i,r)
      for q in meta.statnos:
        queues[q].put(runinfo)

def pplot(*args):
    return postplot(*args)

from plots import *

#entire plotting process
def plotall(allinfo,queues,q):
    #allinfo = (meta,p_run,s_run,n_list,i_list)
#      plt.rc('text', usetex=True)
    info = plots_setup(allinfo,q)
    #print('plot process started '+str(os.getpid()))
    #starttime = timeit.default_timer()
    while(True):
        vals = queues[q].get()
#         if (vals=='init'):
#             print('initializing plots now')
#             plot_info = plots_setup(allinfo,q)
        if (vals=='done'):
            #starttime = timeit.default_timer()
            #print(str(os.getpid())+' saving '+plotnames[q]+' at '+str(starttime))
            #if q !=3:
            #  plots_wrapup(allinfo,q,info)
            #else:
            #  #(meta,p_run,s_run,n_list) = allinfo
            #  #allinfo = (meta,p_run,s_run,n_list,i_list)
            plots_wrapup(allinfo,q,info)
            #endtime = timeit.default_timer()
            #print(str(os.getpid())+' saved '+plotnames[q]+' at '+str(endtime)+': '+str(endtime-starttime)+' elapsed')
            return
        else:
            (p,s,n,i,r) = vals
            plotinfo = (meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i_runs[(p,s,n,i)],r)
            #(meta,p_run,s_run,n_run,i_run,r) = vals
            plotone(plotinfo,q,info)

def fplot(*args):
    return plotall(*args)

for p in meta.paramnos:

  for s in meta.survnos:

    plotonly = os.path.exists(os.path.join(s_runs[(p,s)].topdir_s,meta.plotnames[meta.nstats]))

    #list_nruns = {0:n_runs[(p,s,0)]}
    #list_iruns = {(0,0):i_runs[(p,s,0,0)]}#[{0:i_runs[(p,s,n,i) for i in meta.initnos]} for n in meta.sampnos]
    #for n in meta.sampnos[1:]:
    #  list_nruns[n] = n_runs[(p,s,n)]
    #  for i in meta.initnos[1:]:
    #    list_iruns[(n,i)] = i_runs[(p,s,n,i)]
    #initinfo = (meta,p_runs[(p)],s_runs[(p,s)],list_nruns,list_iruns)
    survinfo = (meta,p_runs[(p)],s_runs[(p,s)])
    n_list = [n_runs[(p,s,n)] for n in meta.sampnos]
    sampinfo = (meta,p_runs[(p)],s_runs[(p,s)],n_list)# for n in meta.sampnos]
    i_list = [[i_runs[(p,s,n,i)] for i in meta.initnos] for n in meta.sampnos]
    initinfo = (meta,p_runs[(p)],s_runs[(p,s)],n_list,i_list)
    #all_idinfo = i_runs.keys()
    #some_idinfo = {(p,s,n,i):perinit(meta,p_runs[p],s_runs[s],n,i) for n in meta.sampnos for i in meta.initnos}

    #all_idinfo = [(p,s,n,i) for p in meta.paramnos for s in meta.survnos for n in meta.sampnos for i in initnos]# for r in runnos]
    #initinfo = [i_runs[(p,s,n,i)] for i in meta.initnos]
    #initinfo = [(meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i_runs[(p,s,n,i)]) for i in meta.initnos for n in meta.sampnos]
    runinfo = [(p,s,n,i) for i in meta.initnos for n in meta.sampnos]# for s in meta.survnos for p in meta.paramnos]

    #set up for multiprocessing of plots and caculations
    nq = meta.nstats#*meta.nparams*meta.nsurvs#number of plot processes
    qnos = range(0,nq)
    queues=[mp.Queue() for q in qnos]

    procs = [mp.Process(target=fplot,args=(initinfo,queues,q)) for q in qnos]
    #procs = [mp.Process(target=fplot,args=(survinfo,queues,q)) for q in qnos[:-1]]
    #procs.append(mp.Process(target=fplot,args=(sampinfo,queues,nq-1)))

    for p in procs:
      p.start()

    # for p in meta.paramnos:
    #   for s in meta.survnos:
    #   # for q in queues:
    # #     q.put('init')

    nps = mp.cpu_count()-1#number of processors to use, leave one free for other things
    pool = mp.Pool(nps)
    #pool.map(fsamp, i_runs.keys())

    if plotonly:
      print('just plotting')
    # # postqueue = [(p,s,n,i,r) for r in meta.stepnos for i in meta.initnos for n in meta.sampnos]
      pool.map(pplot, runinfo)

    else:
      print('beginning calculation')
      pool.map(fsamp, runinfo)

    for q in queues:
      q.put('done')

    for p in procs:
      p.join()