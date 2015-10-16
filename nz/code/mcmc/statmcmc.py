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
        self.name = 'chains'
    def compute(self, ydata):
      y = np.swapaxes(ydata,0,1).T
      var_ls = sum([sum([statistics.variance(y[k][w]) for k in xrange(self.meta.nbins)])/self.meta.nwalkers**2 for w in xrange(self.meta.nwalkers)])
      var_s = sum([sum([statistics.variance(np.exp(y[k][w])) for k in xrange(self.meta.nbins)])/self.meta.nwalkers**2 for w in xrange(self.meta.nwalkers)])
      self.tot_ls = self.tot_ls+var_ls
      self.tot_s = self.tot_s+var_s
      return { 'tot_ls': self.tot_ls,
               'tot_s': self.tot_s,
               'var_ls': var_ls,
               'var_s': var_s
          }

class stat_probs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.name = 'probs'
    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w])/self.meta.nwalkers for w in xrange(self.meta.nwalkers)])
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                 'var_y': var_y }

class stat_fracs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.name = 'fracs'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                 'var_y': var_y }

class stat_times(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.summary = 0
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                'var_y': var_y }

# def cfs(w,k,lag):# per dimension per walker
#     mean = sum(chain)/float(len(chain))
#     terms = [(chain[lag+n]-mean)*(chain[n]-mean) for n in xrange(0,len(chain)-lag)]
#     c = sum(terms)/(len(chain)-lag)
#     return c

# def cf(chains,lag):# per walker
#     terms = [cfs(chain,lag) for chain in chains]
#     c  = sum(terms)/float(len(chain-lag))
#     return ans

# def acors(chains):
#     acfs = []
#     for chain in chains:
#         acf = acor(chain)
#         acfs.append(acf)
#     return acfs

# def acor(chain):# per walker
#     tau = 1+2*sum([cf(chain,t)/cf(chain,0) for t in xrange(len(chain))])
#     return tau
