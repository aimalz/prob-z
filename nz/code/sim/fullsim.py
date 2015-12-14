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
          with open(os.path.join(meta.testdir,'topdirs.p'),'rb') as topdir:
              topdirs = cpkl.load(topdir)#cPickle.dump(self.topdir,topdir)
#               print('reading '+str(topdirs))
              topdirs[inadd] = meta.topdir
          with open(os.path.join(meta.testdir,'topdirs.p'),'wb') as topdir:
              cpkl.dump(topdirs,topdir)
#               print('writing '+str(topdirs))
    else:
          topdirs = {inadd:meta.topdir}
          with open(os.path.join(meta.testdir,'topdirs.p'),'wb') as topdir:
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
    initial_plots(meta,test)
    #print(alltests)
    namesetup(meta)
    return

nps = mp.cpu_count()-1
pool = mp.Pool(nps)

with open(os.path.join(testdir,'tests-sim.txt'),'rb') as testnames:
    pool.map(onerun, testnames)

# for meta in alltests.values():
#     namesetup(meta)

#     for nameplusline in testnames:
#         testname = nameplusline[:-1]
#         meta = setup(testname)
#         test = pertest(meta)
#         initial_plots(meta,test)
