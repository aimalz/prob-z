"""
full-sim module runs data generation procedure
"""

import os
import multiprocessing as mp
import traceback
import cPickle as cpkl

from insim import setup
from persim import pertest
from plotsim import *

testdir = os.path.join('..','tests')

topdir = os.path.join(testdir,'topdirs.p')
if os.path.isfile(topdir):
    os.remove(topdir)

def namesetup(meta):
    td = os.path.join(meta.testdir,'topdirs.p')
    inadd = meta.inadd+'.txt'
    if os.path.isfile(td):
          with open(td,'rb') as topdir:
              topdirs = cpkl.load(topdir)#cPickle.dump(self.topdir,topdir)
#               print('reading '+str(topdirs))
              topdirs[inadd] = meta.topdir
          with open(td,'wb') as topdir:
              cpkl.dump(topdirs,topdir)
#               print('writing '+str(topdirs))
    else:
          topdirs = {inadd:meta.topdir}
          with open(td,'wb') as topdir:
                cpkl.dump(topdirs,topdir)
#                 print('writing '+str(topdirs))
    return

global alltests
alltests = {}

def onerun(inname):
    testname = inname[:-1]
    meta = setup(testname)
    alltests[testname] = meta
    test = pertest(meta)
    print('will plots fail?')
    initial_plots(meta,test)
    print('plots did not fail!')
    namesetup(meta)
    return

nps = mp.cpu_count()-1
pool = mp.Pool(nps)

with open(os.path.join(testdir,'tests-sim.txt'),'rb') as testnames:
    pool.map(onerun, testnames)
