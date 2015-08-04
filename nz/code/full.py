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
iplots(meta,p_runs,s_runs,n_runs,i_runs)

from permcmc import *
def fsamp(*args):
    return samplings(*args)

import multiprocessing as mp

from plots import *
def fplot(*args):
    return plotall(*args)

for p in meta.paramnos:

  for s in meta.survnos:

    survinfo = (meta,p_runs[(p)],s_runs[(p,s)])
    list_nruns = {n:n_runs[(p,s,n) for n in meta.sampnos]}
    list_iruns = [{i:i_runs[(p,s,n,i) for i in meta.initnos]} for n in meta.sampnos]
    initinfo = (meta,p_runs[(p)],s_runs[(p,s)],list_nruns,list_iruns)

    #set up for multiprocessing of plots and caculations
    nq = meta.nstats#*meta.nparams*meta.nsurvs#number of plot processes
    qnos = range(0,nq-1)
    queues=[mp.Queue() for q in qnos]

    procs = [mp.Process(target=fplot,args=(survinfo,queues,q)) for q in qnos]
    procs.append(mp.Process(target=fplot,args=(initinfo,queues,nq)))

    # for p in meta.paramnos:
    #   for s in meta.survnos:
    #   # for q in queues:
    # #     q.put('init')

    nps = mp.cpu_count()-1#number of processors to use, leave one free for other things
    pool = mp.Pool(nps)
    #all_idinfo = i_runs.keys()
    #some_idinfo = {(p,s,n,i):perinit(meta,p_runs[p],s_runs[s],n,i) for n in meta.sampnos for i in meta.initnos}
    initinfo = [(meta,p_runs[(p)],s_runs[(p,s)],n_runs[(p,s,n)],i_runs[(p,s,n,i)]) for i in meta.initnos for n in meta.sampnos]

    #pool.map(fsamp, i_runs.keys())
    pool.map(fsamp, initinfo)

    # #initial plots
    # iplots_all = pool.apply_async(iplots,[meta,p_runs,s_runs,n_runs,i_runs])
    # iplots_all.start()
    # iplots_all.join()
    # #iplots(meta,p_runs,s_runs,n_runs,i_runs)

    #all_idinfo = [(p,s,n,i) for p in meta.paramnos for s in meta.survnos for n in meta.sampnos for i in initnos]# for r in runnos]

    for q in queues:
      q.put('done')

    for p in procs:
      p.join()