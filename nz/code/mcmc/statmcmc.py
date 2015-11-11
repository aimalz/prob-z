"""
stat-mcmc module calculates intermediate statistics for monitoring state
"""

import statistics
import numpy as np
import cPickle as cpkl
import os
import scipy as sp

# unite stats for each output
class calcstats(object):
    def __init__(self, meta):
        self.meta = meta
    def update(self, ydata):
        stats = self.compute(ydata)
        self.meta.key.add_stats(self.meta.topdir, self.name, stats)

class stat_both(calcstats):
    def __init__(self,meta):
        calcstats.__init__(self,meta)

        self.name = 'both'

        self.ll_stack = self.lnlike(self.meta.logstack)
        self.ll_mapNz = self.lnlike(self.meta.logmapNz)
        self.ll_expNz = self.lnlike(self.meta.logexpNz)

        print(str(self.meta.key)+' ll_stack='+str(self.ll_stack))
        print(str(self.meta.key)+' ll_mapNz='+str(self.ll_mapNz))
        print(str(self.meta.key)+' ll_expNz='+str(self.ll_expNz))

        self.llr_stack = []
        self.llr_mapNz = []
        self.llr_expNz = []

        outdict = {'llr_stack': np.array(self.llr_stack),
                  'llr_mapNz': np.array(self.llr_mapNz),
                  'llr_expNz': np.array(self.llr_expNz)
                   }
        with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
            cpkl.dump(outdict,statboth)

    def lnlike(self, theta):
        return self.meta.postdist.lnprob(theta)-self.meta.postdist.priorprob(theta)

    def compute(self,ydata):
        probs = ydata['probs']
        chains = ydata['chains']
        for w in xrange(self.meta.nwalkers):
            for x in xrange(self.meta.ntimes):
                ll_samp = probs[w][x]-self.meta.postdist.priorprob(chains[w][x])
                self.llr_stack.append(2.*ll_samp-2.*self.ll_stack)
                self.llr_mapNz.append(2.*ll_samp-2.*self.ll_mapNz)
                self.llr_expNz.append(2.*ll_samp-2.*self.ll_expNz)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as indict:
            outdict = cpkl.load(indict)

        outdict['llr_stack'] = np.array(self.llr_stack)
        outdict['llr_mapNz'] = np.array(self.llr_mapNz)
        outdict['llr_expNz'] = np.array(self.llr_expNz)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
            cpkl.dump(outdict,statboth)

class stat_chains(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)

        self.name = 'chains'

        #self.tot_var_ls = 0.
        #self.tot_var_s = 0.
        self.var_ls = []
        self.var_s = []
        self.vslogstack = None
        self.vslogmapNz = None
        self.vslogexpNz = None
        self.vsstack = None
        self.vsmapNz = None
        self.vsexpNz = None

        #self.tot_chi_ls = 0.
        #self.tot_chi_s = 0.
        self.chi_ls = []
        self.chi_s = []
        self.cslogstack = None
        self.cslogmapNz = None
        self.cslogexpNz = None
        self.csstack = None
        self.csmapNz = None
        self.csexpNz = None

        if self.meta.logtrueNz is not None:
            vslogstack = self.meta.logstack-self.meta.logtrueNz
            self.vslogstack = np.dot(vslogstack,vslogstack)
            vslogmapNz = self.meta.logmapNz-self.meta.logtrueNz
            self.vslogmapNz = np.dot(vslogmapNz,vslogmapNz)
            vslogexpNz = self.meta.logexpNz-self.meta.logtrueNz
            self.vslogexpNz = np.dot(vslogexpNz,vslogexpNz)

            self.cslogstack = np.average((self.meta.logstack-self.meta.logtrueNz)**2)
            self.cslogmapNz = np.average((self.meta.logmapNz-self.meta.logtrueNz)**2)
            self.cslogexpNz = np.average((self.meta.logexpNz-self.meta.logtrueNz)**2)

        if self.meta.trueNz is not None:
            vsstack = meta.stack-meta.trueNz
            self.vsstack = np.dot(vsstack,vsstack)
            vsmapNz = meta.mapNz-meta.trueNz
            self.vsmapNz = np.dot(vsmapNz,vsmapNz)
            vsexpNz = meta.expNz-meta.trueNz
            self.vsexpNz = np.dot(vsexpNz,vsexpNz)

            self.csstack = np.average((self.meta.stack-self.meta.trueNz)**2)
            self.csmapNz = np.average((self.meta.mapNz-self.meta.trueNz)**2)
            self.csexpNz = np.average((self.meta.expNz-self.meta.trueNz)**2)

        outdict = {'vslogstack': self.vslogstack,
               'vsstack': self.vsstack,
               'vslogmapNz': self.vslogmapNz,
               'vsmapNz': self.vsmapNz,
               'vslogexpNz': self.vslogexpNz,
               'vsexpNz': self.vsexpNz,
               'cslogstack': self.cslogstack,
               'csstack': self.csstack,
               'cslogmapNz': self.cslogmapNz,
               'csmapNz': self.csmapNz,
               'cslogexpNz': self.cslogexpNz,
               'csexpNz': self.csexpNz   }

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as statchains:
            cpkl.dump(outdict,statchains)

    def compute(self, ydata):#ntimes*nwalkers*nbins

        flatdata = np.array([ydata.T[b].flatten() for b in xrange(self.meta.nbins)])
        eydata = np.exp(ydata)
        eflatdata = np.array([eydata.T[b].flatten() for b in xrange(self.meta.nbins)])

        vy = abs(np.linalg.det(np.cov(flatdata)))#np.average([[statistics.variance(walk) for walk in ydata.T[b]] for b in xrange(self.meta.nbins)])
        vey = abs(np.linalg.det(np.cov(eflatdata)))#np.average([[statistics.variance(walk) for walk in eydata.T[b]] for b in xrange(self.meta.nbins)])
        y = np.swapaxes(ydata.T,0,1).T#nwalkers*nbins*ntimes
        ey = np.swapaxes(eydata.T,0,1).T#np.exp(y)

        if self.meta.logtrueNz is None:
            my = np.array([[[sum(by)/len(by)]*self.meta.ntimes for by in wy] for wy in y])#nwalkers*nbins*ntimes
        else:
            my = np.array([[[k]*self.meta.ntimes for k in self.meta.logtrueNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes

        if self.meta.trueNz is None:
            mey = np.array([[[sum(bey)/len(bey)]*self.meta.ntimes for bey in wey] for wey in ey])#nwalkers*nbins*ntimes
        else:
            mey = np.array([[[k]*self.meta.ntimes for k in self.meta.trueNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes

        sy = np.swapaxes((y-my),1,2)#nwalkers*ntimes*nbins to #nwalkers*nbins*ntimes
        sey = np.swapaxes((ey-mey),1,2)

        var_ls = np.average([[np.dot(sy[w][i],sy[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
        var_s = np.average([[np.dot(sey[w][i],sey[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
        self.var_ls.append(var_ls)
        self.var_s.append(var_s)
        #self.tot_var_ls = self.tot_var_ls+var_ls
        #self.tot_var_s = self.tot_var_s+var_s
        print('var_ls='+str(self.var_ls))
        print('var_s='+str(self.var_s))

        chi_ls = np.average(sp.stats.chisquare(flatdata.T)[0])#np.sum(sy**2)/vy#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vy)
        chi_s = np.average(sp.stats.chisquare(eflatdata.T)[0])#np.sum(sey**2)/vey#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vey)
        self.chi_ls.append(chi_ls)
        self.chi_s.append(chi_s)
        #self.tot_chi_ls = self.tot_chi_ls+chi_ls
        #self.tot_chi_s = self.tot_chi_s+chi_s
        print('chi_ls='+str(self.chi_ls))
        print('chi_s='+str(self.chi_s))

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as indict:
            outdict = cpkl.load(indict)

        #outdict['tot_var_ls'] = self.tot_var_ls
        #outdict['tot_var_s'] = self.tot_var_s
        outdict['var_ls'] = self.var_ls
        outdict['var_s'] = self.var_s
        #outdict['tot_chi_ls'] = self.tot_chi_ls
        #outdict['tot_chi_s'] = self.tot_chi_s
        outdict['chi_ls'] = self.chi_ls
        outdict['chi_s'] = self.chi_s

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as indict:
            cpkl.dump(outdict,indict)
            #print('after addition'+str(outdict))

#         return { 'vslogstack': self.vslogstack,
#                'vsstack': self.vsstack,
#                'vslogmapNz': self.vslogmapNz,
#                'vsmapNz': self.vsmapNz,
#                'vslogexpNz': self.vslogexpNz,
#                'vsexpNz': self.vsexpNz,
#                'tot_var_ls': self.tot_var_ls,
#                'tot_var_s': self.tot_var_s,
#                'var_ls': self.var_ls,
#                'var_s': self.var_s
#               }

class stat_probs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        #self.summary = 0
        self.name = 'probs'

#         # calculating log likelihood ratio test statistic for each relative to truth (and true relative to prior)
        if self.meta.logtrueNz is not None:
            self.lp_true = self.meta.postdist.lnprob(self.meta.logtrueNz)
        else:
            self.lp_true = self.meta.postdist.lnprob(self.meta.mean)

        self.lp_stack = self.meta.postdist.lnprob(self.meta.logstack)
        self.lp_mapNz = self.meta.postdist.lnprob(self.meta.logmapNz)
        self.lp_expNz = self.meta.postdist.lnprob(self.meta.logexpNz)

        self.var_y = []

    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w]) for w in xrange(self.meta.nwalkers)])/self.meta.nwalkers
        #self.llr_samp.append((2.*np.max(lik_y)-2.*self.ll_true))
        self.var_y.append(var_y)
        # self.summary = self.summary+var_y
        return { #'summary': self.summary,
                 'var_y': self.var_y,
                 'lp_true': self.lp_true,
                 'lp_stack': self.lp_stack,
                 'lp_mapNz': self.lp_mapNz,
                 'lp_expNz': self.lp_expNz
               }

class stat_fracs(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        #self.summary = 0
        self.var_y = []
        self.name = 'fracs'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        #self.summary = self.summary+var_y
        self.var_y.append(var_y)
        return { #'summary': self.summary,
                 'var_y': self.var_y }

class stat_times(calcstats):
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        #self.summary = 0
        self.var_y = []
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = np.var(y)#statistics.variance(y)
        #self.summary = self.summary+var_y
        self.var_y.append(var_y)
        return { #'summary': self.summary,
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