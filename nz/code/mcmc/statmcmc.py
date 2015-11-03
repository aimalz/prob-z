"""
stat-mcmc module calculates intermediate statistics for monitoring state
"""

import statistics
import numpy as np
import cPickle as cpkl
import os

# unite stats for each output
class calcstats(object):
    def __init__(self, meta):
        self.meta = meta
    def update(self, ydata):
        stats = self.compute(ydata)
        self.meta.key.add_stats(self.meta.topdir, self.name, stats)

class stat_chains(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.tot_ls = 0.
        self.tot_s = 0.
        self.var_ls = []
        self.var_s = []
        self.name = 'chains'
        self.vslogstack = None
        self.vslogmapNz = None
        self.vslogexpNz = None
        self.vsstack = None
        self.vsmapNz = None
        self.vsexpNz = None
        if meta.logtrueNz is not None:
            # print(meta.logstack,meta.logtrueNz)
            vslogstack = meta.logstack-meta.logtrueNz
            # print(vslogstack)
            self.vslogstack = np.dot(vslogstack,vslogstack)/self.meta.nbins
            # print(self.vslogstack)
            vslogmapNz = meta.logmapNz-meta.logtrueNz
            self.vslogmapNz = np.dot(vslogmapNz,vslogmapNz)/self.meta.nbins
            vslogexpNz = meta.logexpNz-meta.logtrueNz
            self.vslogexpNz = np.dot(vslogexpNz,vslogexpNz)/self.meta.nbins
        if meta.trueNz is not None:
            # print(meta.stack,meta.trueNz)
            vsstack = meta.stack-meta.trueNz
            # print(vsstack)
            self.vsstack = np.dot(vsstack,vsstack)/self.meta.nbins
            # print(self.vsstack)
            vsmapNz = meta.mapNz-meta.trueNz
            self.vsmapNz = np.dot(vsmapNz,vsmapNz)/self.meta.nbins
            vsexpNz = meta.expNz-meta.trueNz
            self.vsexpNz = np.dot(vsexpNz,vsexpNz)/self.meta.nbins
        outdict = {'vslogstack': self.vslogstack,
               'vsstack': self.vsstack,
               'vslogmapNz': self.vslogmapNz,
               'vsmapNz': self.vsmapNz,
               'vslogexpNz': self.vslogexpNz,
               'vsexpNz': self.vsexpNz}
        #print(type(outdict))
        #print(outdict)
        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as statchains:
            cpkl.dump(outdict,statchains)

    def compute(self, ydata):#ntimes*nwalkers*nbins
        y = np.swapaxes(ydata.T,0,1).T#nwalkers*nbins*ntimes
        if self.meta.logtrueNz is None:
            my = np.array([[[sum(by)/len(by)]*self.meta.ntimes for by in wy] for wy in y])#nwalkers*nbins*ntimes
        else:
            my = np.array([[[k]*self.meta.ntimes for k in self.meta.logtrueNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes
        sy = np.swapaxes((y-my),1,2)#nwalkers*time*nbins
        ey = np.exp(y)
        if self.meta.trueNz is None:
            mey = np.array([[[sum(bey)/len(bey)]*self.meta.ntimes for bey in wey] for wey in ey])#nwalkers*nbins*ntimes
        else:
            mey = np.array([[[k]*self.meta.ntimes for k in self.meta.trueNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes
        sey = np.swapaxes((ey-mey),1,2)
        var_ls = sum([sum([np.dot(sy[w][i],sy[w][i]) for i in xrange(self.meta.ntimes)]) for w in xrange(self.meta.nwalkers)])/(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
        var_s = sum([sum([np.dot(sey[w][i],sey[w][i]) for i in xrange(self.meta.ntimes)]) for w in xrange(self.meta.nwalkers)])/(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
        self.var_ls.append(var_ls)
        self.var_s.append(var_s)
        #print('stats var_ls='+str(self.var_ls))
        #print('stats var_s='+str(self.var_s))
        self.tot_ls = self.tot_ls+var_ls
        self.tot_s = self.tot_s+var_s
        #print('stats tot_ls='+str(self.tot_ls))
        #print('stats tot_s='+str(self.tot_s))
        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as indict:
            outdict = cpkl.load(indict)
            #print(type(outdict))
            #print('before addition'+str(outdict))
        outdict['tot_ls'] = self.tot_ls
        outdict['tot_s'] = self.tot_s
        outdict['var_ls'] = self.var_ls
        outdict['var_s'] = self.var_s
            #print(type(outdict))
        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as indict:
            cpkl.dump(outdict,indict)
            #print('after addition'+str(outdict))

#         return { 'vslogstack': self.vslogstack,
#                'vsstack': self.vsstack,
#                'vslogmapNz': self.vslogmapNz,
#                'vsmapNz': self.vsmapNz,
#                'vslogexpNz': self.vslogexpNz,
#                'vsexpNz': self.vsexpNz,
#                'tot_ls': self.tot_ls,
#                'tot_s': self.tot_s,
#                'var_ls': self.var_ls,
#                'var_s': self.var_s
#               }

class stat_probs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.var_y = []
        self.name = 'probs'
    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w])/self.meta.nwalkers for w in xrange(self.meta.nwalkers)])
        self.summary = self.summary+var_y
        self.var_y.append(var_y)
        return { 'summary': self.summary,
                 'var_y': self.var_y }

class stat_fracs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.var_y = []
        self.name = 'fracs'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.summary = self.summary+var_y
        self.var_y.append(var_y)
        return { 'summary': self.summary,
                 'var_y': self.var_y }

class stat_times(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.var_y = []
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = np.var(y)#statistics.variance(y)
        self.summary = self.summary+var_y
        self.var_y.append(var_y)
        return { 'summary': self.summary,
                 'var_y': self.var_y }

def cft(xtimes,lag):
    lent = len(xtimes)-lag
    allt = xrange(lent)
    ans = np.array([xtimes[t+lag]*xtimes[t] for t in allt])
    return ans

def cf(xtimes):#xtimes has ntimes elements
    cf0 = np.dot(xtimes,xtimes)
    allt = xrange(len(xtimes)/2)
    cf = np.array([sum(cft(xtimes,lag)[len(xtimes)/2:]) for lag in allt])/cf0
    return cf

def cfs(x,mode):#xbinstimes has nbins by ntimes elements
    if mode == 'walkers':
        xbinstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xbinstimes])/len(xbinstimes)
    if mode == 'bins':
        xwalkerstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xwalkerstimes])/len(xwalkerstimes)
    return cfs

def acors(xtimeswalkersbins,mode):
    if mode == 'walkers':
        xwalkersbinstimes = np.swapaxes(xtimeswalkersbins,1,2)#nwalkers by nbins by nsteps
        taus = np.array([1. + 2.*sum(cfs(xbinstimes,mode)) for xbinstimes in xwalkersbinstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    if mode == 'bins':
        xbinswalkerstimes = xtimeswalkersbins.T#nbins by nwalkers by nsteps
        taus = np.array([1. + 2.*sum(cfs(xwalkerstimes,mode)) for xwalkerstimes in xbinswalkerstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    return taus
