"""
full-sim module runs data generation procedure
"""

from insim import setup
from persim import pertest
from plotsim import *
import os
import multiprocessing as mp
import traceback

testdir = os.path.join('..','tests')

topdir = os.path.join(testdir,'topdirs.p')
if os.path.isfile(topdir):
    os.remove(topdir)

def onerun(inname):
    testname = inname[:-1]
    meta = setup(testname)
    test = pertest(meta)
    initial_plots(meta,test)

nps = mp.cpu_count()
pool = mp.Pool(nps)

with open(os.path.join(testdir,'tests.txt'),'rb') as testnames:
    pool.map(onerun, testnames)


#     for nameplusline in testnames:
#         testname = nameplusline[:-1]
#         meta = setup(testname)
#         test = pertest(meta)
#         initial_plots(meta,test)
