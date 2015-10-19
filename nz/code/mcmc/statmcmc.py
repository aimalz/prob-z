"""
stat-mcmc module calculates intermediate statistics for monitoring state
"""

import statistics
import numpy as np

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
        self.tot_ls = 0
        self.tot_s = 0
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
            vslogstack = meta.logstack-meta.logtrueNz
            self.vslogstack = np.dot(vslogstack,vslogstack)/self.meta.nbins**2
            vslogmapNz = meta.logmapNz-meta.logtrueNz
            self.vslogmapNz = np.dot(vslogmapNz,vslogmapNz)/self.meta.nbins**2
            vslogexpNz = meta.logexpNz-meta.logtrueNz
            self.vslogexpNz = np.dot(vslogexpNz,vslogexpNz)/self.meta.nbins**2
        if meta.trueNz is not None:
            vsstack = meta.stack-meta.trueNz
            self.vsstack = np.dot(vsstack,vsstack)/self.meta.nbins**2
            vsmapNz = meta.mapNz-meta.trueNz
            self.vsmapNz = np.dot(vsmapNz,vsmapNz)/self.meta.nbins**2
            vsexpNz = meta.expNz-meta.trueNz
            self.vsexpNz = np.dot(vsexpNz,vsexpNz)/self.meta.nbins**2
    def compute(self, ydata):
      y = np.swapaxes(ydata,0,1).T
      ey = np.exp(y)
      var_ls = sum([sum([np.dot(y[k][w],y[k][w])/self.meta.ntimes**2 for k in xrange(self.meta.nbins)])/self.meta.nbins**2 for w in xrange(self.meta.nwalkers)])/self.meta.nwalkers**2
      var_s = sum([sum([np.dot(ey[k][w],ey[k][w])/self.meta.ntimes**2 for k in xrange(self.meta.nbins)])/self.meta.nbins**2 for w in xrange(self.meta.nwalkers)])/self.meta.nwalkers**2
      self.var_ls.append(var_s)
      self.var_ls.append(var_s)
      self.tot_ls = self.tot_ls+var_ls
      self.tot_s = self.tot_s+var_s
      return { 'vslogstack': self.vslogstack,
               'vsstack': self.vsstack,
               'vslogmapNz': self.vslogmapNz,
               'vsmapNz': self.vsmapNz,
              'vslogexpNz': self.vslogexpNz,
               'vsexpNz': self.vsexpNz,
              'tot_ls': self.tot_ls,
               'tot_s': self.tot_s,
               'var_ls': self.var_ls,
               'var_s': self.var_s
          }

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

# def cfs(onewalkoneparam,lag):# per dimension per walker per lag
#     mean = sum(onewalkoneparam)/float(len(onewalkoneparam))
#     terms = [(onewalkoneparam[lag+n]-mean)*(onewalkoneparam[n]-mean) for n in xrange(0,len(onewalkoneparam)-lag)]
#     c = sum(terms)/(len(onewalkoneparam)-lag)
#     return c

# def cf(onewalk,lag):# per walker per lag
#     terms = [cfs(onewalk[k],lag) for k in xrange(len(onewalk))]
#     c  = sum(terms)/float(len(chain[0]))
#     return c

def cf(xtimes):#x has ntimes elements
    cf0 = np.dot(xtimes,xtimes)
    #print(str(np.shape(xsteps))+' should be nsteps')
    cf = np.correlate(xtimes, xtimes, mode='full')/cf0
    #ans = cf[len(cf)/2:]
    return cf

def cfs(xbinstimes):#xs has nbins by ntimes elements
    #print(str(np.shape(xbinssteps))+' should be (nbins,nsteps)')
    cfs = np.array([sum(cf(xtimes)) for xtimes in xbinstimes])/len(xbinstimes)
    return cfs

def acors(xtimeswalkersbins):
    #print(str(np.shape(xstepswalkersbins))+' should be (ntimes,nwalkers,nbins)')
    xwalkersbinstimes = np.swapaxes(xtimeswalkersbins,1,2)#nwalkers by nbins by nsteps
    #print(str(np.shape(xwalkersbinssteps))+' should be (nwalkers,nbins,ntimes)')
    taus = np.array([sum(cfs(xbinstimes)) for xbinstimes in xwalkersbinstimes])/len(xwalkersbinstimes)# 1+2*sum(...)
    #print(str(np.shape(taus))+' should be nwalkers')
    #print('acor times='+str(taus))
    return taus
