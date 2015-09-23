# calculate intermediate statistics for monitoring state

import statistics
import numpy as np

# unite stats for each output
class calcstats(object):
    def __init__(self, i_run):
        self.i_run = i_run
        self.n_run = i_run.n_run
    def update(self, ydata):
        stats = self.compute(ydata)
        self.i_run.key.add_stats(self.i_run.meta.topdir, self.name, stats)

class stat_chains(calcstats):
    def __init__(self, i_run):
        calcstats.__init__(self, i_run)
        self.tot_ls = 0
        self.tot_s = 0
        self.name = 'chains'
    def compute(self, ydata):
      y = np.swapaxes(ydata,0,1).T
      var_ls = sum([sum([statistics.pvariance(y[k][w],mu=self.n_run.full_logsampNz[k])/self.n_run.nbins for k in self.n_run.binnos])/self.n_run.nwalkers for w in self.n_run.walknos])
      var_s = sum([sum([statistics.pvariance(np.exp(y[k][w]),mu=self.n_run.full_sampNz[k])/self.n_run.nbins for k in self.n_run.binnos])/self.n_run.nwalkers for w in self.n_run.walknos])
      self.tot_ls = self.tot_ls+var_ls
      self.tot_s = self.tot_s+var_s
      return { 'tot_ls': self.tot_ls,
               'tot_s': self.tot_s,
               'var_ls': var_ls,
               'var_s': var_s
          }

class stat_probs(calcstats):
    def __init__(self, i_run):
        calcstats.__init__(self, i_run)
        self.summary = 0
        self.name = 'probs'
    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w])/self.n_run.nwalkers for w in self.n_run.walknos])
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                 'var_y': var_y }

class stat_fracs(calcstats):
    def __init__(self, i_run):
        calcstats.__init__(self, i_run)
        self.summary = 0
        self.name = 'fracs'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                 'var_y': var_y }

class stat_times(calcstats):
    def __init__(self, i_run):
        calcstats.__init__(self, i_run)
        self.summary = 0
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.summary = self.summary+var_y
        return { 'summary': self.summary,
                 'var_y': var_y }
