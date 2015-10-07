"""
full module runs MCMC and plotting in parallel
"""

#!/usr/bin/python2
import matplotlib
import traceback
import distribute
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from inputs import setup
from util import *
plt.rc('text', usetex=True)
import sys
import traceback
from key import key

from perparam import perparam
from persurv import persurv
from persamp import persamp
from perinit import perinit

import plots
import iplots
import multiprocessing as mp

# define producer functions
def samplings(runs, idinfo):
    i_run = runs['i_runs'][idinfo]
    for _ in i_run.samplings():
        print('sampling ' + str(idinfo) + ": this should probably return something")

def fsamp(idinfo):
    print ('starting key: ' + str(idinfo))
    sys.stdout.flush()
    try:
        return samplings(init_runs, idinfo)
    except:
        e = sys.exc_info()[0]
        print 'Nested Exception: '
        print traceback.format_exc()
        sys.stdout.flush()
        raise e
    finally:
        print ('Done')

def fcalc(idinfo):
    if ~meta.mcmc:
        return

# currently no way to bypass plotting
# def noplot(meta, runs, dist, idinfo):
#     if ~meta.plots:
#         return

# all the work happens here
def main():
    meta = setup()
    p_runs = {key(p=p):perparam(meta, p) for p in lrange(meta.params)}
    s_runs = {p.add(s=s):persurv(meta, p_runs[p], s) for s in lrange(meta.survs) for p in p_runs.keys()}
    n_runs = {s.add(n=n):persamp(meta, s_runs[s], n) for n in xrange(meta.samps) for s in s_runs.keys()}
    i_runs = {n.add(i=i):perinit(meta, n_runs[n], i) for i in lrange(meta.inits) for n in n_runs.keys()}
    runs = {'p_runs': p_runs,
            's_runs': s_runs,
            'n_runs': n_runs,
            'i_runs': i_runs}
    global init_runs
    init_runs = runs

    # make initial plots
    distribute.run_offthread_sync(iplots.initial_plots, meta, runs)

    nps = mp.cpu_count()#-1
    for s_run in s_runs.values():
        print ('starting run of: ' + str(s_run.key))
        # fork off all of the plotter threads,
        dist = distribute.distribute(plots.all_plotters,
                                     start = True,
                                     meta = meta,
                                     p_run = s_run.p_run,
                                     s_run = s_run,
                                     n_runs = s_run.n_runs,
                                     i_runs = [i_run for n_run in s_run.n_runs for i_run in n_run.i_runs])
        # inject the distribution handler into the metadata object
        print ('plotter threads started')
        s_run.dist = dist
        print ('setting dist on {} to {}'.format(s_run, s_run.dist))
        pool = mp.Pool(nps)

        # may add back plot-only functionality later
        keys = [s_run.key.add(n=n, i=i) for i in lrange(meta.inits) for n in xrange(meta.samps)]
        print ('generating {} keys'.format(len(keys)))
        pool.map(fsamp, keys)
        dist.finish()
        print('ending run of: ' + str(s_run.key))

if __name__ == '__main__':
    main()
