# import cProfile
# import pstats
# import StringIO
# pr = cProfile.Profile()
# pr.enable()

import cPickle as cpkl
import os
import distribute
import multiprocessing as mp
import traceback
import psutil
import timeit
import sys
import cProfile
import pstats
import StringIO

from inmcmc import setup
from permcmc import pertest
import plotmcmc as plots

# define producer functions
def samplings(runs, idinfo):
    meta = runs[idinfo]
    for _ in meta.samplings():
        print('sampling ' + str(idinfo) + ": this should probably return something")
#     print('ending run of: ' + str(idinfo))
#     meta.dist.finish()

def fsamp(idinfo):
    print ('starting key ' + str(idinfo))
    sys.stdout.flush()
    try:
        print('tried sampling')
        return samplings(init_runs, idinfo)
    except:
        print('failed sampling')
        e = sys.exc_info()[0]
        print 'Nested Exception: '
        print traceback.format_exc()
        sys.stdout.flush()
        raise e
    finally:
        print ('Done with '+str(idinfo))

#def fplot(runs,key):
#    runs[key].dist.finish()
#    return

testdir = os.path.join('..','tests')
#topdir = open(os.path.join(testdir,'topdirs.p'),'rb')

def main():


    runs  = {}
    with open(os.path.join(testdir,'tests-mcmc.txt'),'rb') as names:#open(os.path.join(testdir,'topdirs.p'),'rb') as names:
        #names = cpkl.load(names)
        for name in names:
            meta = setup(name[:-1])
            runs[name[:-4]] = meta

    # make initial plots
#     distribute.run_offthread_sync(plots.initial_plots(runs))
    #runs = {'runs':tests}
    global init_runs
    init_runs = runs

    # fork off all of the plotter threads,
    for run_name in runs.keys():
        print ('starting run of ' + str(run_name))
        dist = distribute.distribute(plots.all_plotters,
                                     start = True,
                                     meta = runs[run_name])
        print ('plotter threads for: '+ run_name + ' started')
    # inject the distribution handler into the metadata object
        runs[run_name].dist = dist
        print ('setting dist on {} to {}'.format(run_name, runs[run_name].dist))

    nps = mp.cpu_count()-1
    pool = mp.Pool(nps)
    meta.pool = pool

    # may add back plot-only functionality later
    #keys = runs.keys() #[run.key.add(t=name) for name in lrange(names)]
    print ('generating {} keys'.format(len(runs.keys())))
    start_time = timeit.default_timer()
    pool.map(fsamp, runs.keys())
    elapsed = timeit.default_timer() - start_time

    for run in runs.keys():
        runs[run].dist.finish()
        print('ending run of: ' + str(run))

if __name__ == '__main__':
    main()

# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()
