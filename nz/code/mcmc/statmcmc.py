"""
stat-mcmc module calculates intermediate statistics for monitoring state
"""

import statistics
import numpy as np
import cPickle as cpkl
import os
import scipy as sp
import csv

# unite stats for each output
class calcstats(object):
    """
    object class to set up and calculate summary statistics
    """
    def __init__(self, meta):
        self.meta = meta
    def update(self, ydata):
        if self.meta.plotonly == False:
            stats = self.compute(ydata)
            self.meta.key.add_stats(self.meta.topdir, self.name, stats)

# statistics involving both log posterior probabilities and parameter values
class stat_both(calcstats):
    """
    calculates statistics that require both posterior probabilities and parameter values: log likelihood ratio and MAP parameter values
    """
    def __init__(self,meta):
        calcstats.__init__(self,meta)

        self.name = 'both'

#         self.ll_stkNz = self.meta.postdist.lnlike(self.meta.logstkNz)
#         self.ll_mapNz = self.meta.postdist.lnlike(self.meta.logmapNz)
#         self.ll_expNz = self.meta.postdist.lnlike(self.meta.logexpNz)
        self.ll_intNz = self.meta.postdist.lnlike(self.meta.logintNz)
        self.ll_mmlNz = self.meta.postdist.lnlike(self.meta.logmmlNz)
        self.ll_smpNz = []
#         self.mapvals,self.maps = [],[]

#         self.llr_stkNz,self.llr_mapNz,
        self.llr_intNz,self.llr_mmlNz = [],[]

        outdict = {#'ll_stkNz': self.ll_stkNz,
#                   'll_mapNz': self.ll_mapNz,
#                   'll_expNz': self.ll_expNz,
                  'll_intNz': self.ll_intNz,
                  'll_mmlNz': self.ll_mmlNz,
                  'll_smpNz': self.ll_smpNz,
#                   'llr_stkNz': np.array(self.llr_stkNz),
#                   'llr_mapNz': np.array(self.llr_mapNz),
#                   'llr_expNz': np.array(self.llr_expNz),
                  'llr_intNz': np.array(self.llr_intNz),
                  'llr_mmlNz': np.array(self.llr_mmlNz)
                   }
        if self.meta.plotonly == False:
            with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
                cpkl.dump(outdict,statboth)

    def compute(self,ydata):

        self.probs = ydata['probs']
        self.chains = ydata['chains']

#         where = np.unravel_index(np.argmax(self.probs),(self.meta.nwalkers,self.meta.ntimes))
#         self.mapvals.append(self.chains[where])
#         self.maps.append(self.probs[where])

#         self.llr_stkNz = self.calclr(self.llr_stkNz,self.ll_stkNz)
#         self.llr_mapNz = self.calclr(self.llr_mapNz,self.ll_mapNz)
#         self.llr_expNz = self.calclr(self.llr_expNz,self.ll_expNz)
        self.llr_mmlNz = self.calclr(self.llr_mmlNz,self.ll_mmlNz)

        if self.meta.logtruNz is not None:
            self.calclr(self.ll_smpNz,0.)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as indict:
            outdict = cpkl.load(indict)

#         outdict['llr_stkNz'] = np.array(self.llr_stkNz)
#         outdict['llr_mapNz'] = np.array(self.llr_mapNz)
#         outdict['llr_expNz'] = np.array(self.llr_expNz)
        outdict['llr_mmlNz'] = np.array(self.llr_mmlNz)
        outdict['ll_smpNz'] = np.array(self.ll_smpNz).flatten()/2.
#         outdict['mapvals'] = np.array(self.mapvals)
#         outdict['maps'] = np.array(self.maps)

        with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
            cpkl.dump(outdict,statboth)
        return

    # likelihood ratio test
    def calclr(self,var,ll):

        for w in xrange(self.meta.nwalkers):
            for x in xrange(self.meta.ntimes):
                ll_smpNz = self.probs[w][x]-self.meta.postdist.priorprob(self.chains[w][x])
                var.append(2.*(ll_smpNz-ll))
#                 self.llr_stkNz.append(2.*ll_smpNz-2.*self.ll_stkNz)
#                 self.llr_mapNz.append(2.*ll_smpNz-2.*self.ll_mapNz)
#                 self.llr_expNz.append(2.*ll_smpNz-2.*self.ll_expNz)
        return(var)

# statistics involving parameter values
class stat_chains(calcstats):
    """
    calculates statistics that need parameter values: variance, chi^2, KLD
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)

        self.name = 'chains'

        self.var_ls = []
        self.var_s = []
#         self.vslogstkNz = None
#         self.vslogmapNz = None
# #         self.vslogexpNz = None
#         self.vsstkNz = None
#         self.vsmapNz = None
# #         self.vsexpNz = None

        self.chi_ls = []
        self.chi_s = []
#         self.cslogstkNz = None
#         self.cslogmapNz = None
# #         self.cslogexpNz = None
#         self.csstkNz = None
#         self.csmapNz = None
# #         self.csexpNz = None

#         self.kl_stkNzvtruNz,self.kl_truNzvstkNz = None,None
#         self.kl_mapNzvtruNz,self.kl_truNzvmapNz = None,None
# #         self.kl_expNzvtruNz,self.kl_truNzvexpNz = None,None
        self.kl_intNzvtruNz,self.kl_truNzvintNz = float('inf'),float('inf')
        self.kl_mmlNzvtruNz,self.kl_truNzvmmlNz = float('inf'),float('inf')
        self.kl_smpNzvtruNz,self.kl_truNzvsmpNz = float('inf'),float('inf')

        if self.meta.logtruNz is not None:
#             vslogstkNz = self.meta.logstkNz-self.meta.logtruNz
#             self.vslogstkNz = np.dot(vslogstkNz,vslogstkNz)
#             vslogmapNz = self.meta.logmapNz-self.meta.logtruNz
#             self.vslogmapNz = np.dot(vslogmapNz,vslogmapNz)
# #             vslogexpNz = self.meta.logexpNz-self.meta.logtruNz
# #             self.vslogexpNz = np.dot(vslogexpNz,vslogexpNz)

#             self.cslogstkNz = np.average((self.meta.logstkNz-self.meta.logtruNz)**2)
#             self.cslogmapNz = np.average((self.meta.logmapNz-self.meta.logtruNz)**2)
# #             self.cslogexpNz = np.average((self.meta.logexpNz-self.meta.logtruNz)**2)

#             self.kl_stkNzvtruNz,self.kl_truNzvstkNz = self.calckl(self.meta.logstkNz,self.meta.logtruNz)
#             self.kl_mapNzvtruNz,self.kl_truNzvmapNz = self.calckl(self.meta.logmapNz,self.meta.logtruNz)
# #             self.kl_expNzvtruNz,self.kl_truNzvexpNz = self.calckl(self.meta.logexpNz,self.meta.logtruNz)
            self.kl_intNzvtruNz,self.kl_truNzvintNz = self.calckl(self.meta.logintNz,self.meta.logtruNz)
            self.kl_mmlNzvtruNz,self.kl_truNzvmmlNz = self.calckl(self.meta.logmmlNz,self.meta.logtruNz)
            self.kl_smpNzvtruNz,self.kl_truNzvsmpNz = [],[]

#         if self.meta.truNz is not None:
#             vsstkNz = meta.stkNz-meta.truNz
#             self.vsstkNz = np.dot(vsstkNz,vsstkNz)
#             vsmapNz = meta.mapNz-meta.truNz
#             self.vsmapNz = np.dot(vsmapNz,vsmapNz)
# #             vsexpNz = meta.expNz-meta.truNz
# #             self.vsexpNz = np.dot(vsexpNz,vsexpNz)

#             self.csstkNz = np.average((self.meta.stkNz-self.meta.truNz)**2)
#             self.csmapNz = np.average((self.meta.mapNz-self.meta.truNz)**2)
#             self.csexpNz = np.average((self.meta.expNz-self.meta.truNz)**2)

        outdict = {#'vslogstkNz': self.vslogstkNz,
#                    'vsstkNz': self.vsstkNz,
#                    'vslogmapNz': self.vslogmapNz,
#                    'vsmapNz': self.vsmapNz,
# #                    'vslogexpNz': self.vslogexpNz,
# #                    'vsexpNz': self.vsexpNz,
#                    'cslogstkNz': self.cslogstkNz,
#                    'csstkNz': self.csstkNz,
#                    'cslogmapNz': self.cslogmapNz,
#                    'csmapNz': self.csmapNz,
# #                    'cslogexpNz': self.cslogexpNz,
# #                    'csexpNz': self.csexpNz,
#                    'kl_stkNzvtruNz': self.kl_stkNzvtruNz,
#                    'kl_mapNzvtruNz': self.kl_mapNzvtruNz,
# #                    'kl_expNzvtruNz': self.kl_expNzvtruNz,
                   'kl_smpNzvtruNz': self.kl_smpNzvtruNz,
                   'kl_intNzvtruNz': self.kl_intNzvtruNz,
                   'kl_mmlNzvtruNz': self.kl_mmlNzvtruNz,
#                    'kl_truNzvstkNz': self.kl_truNzvstkNz,
#                    'kl_truNzvmapNz': self.kl_truNzvmapNz,
# #                    'kl_truNzvexpNz': self.kl_truNzvexpNz,
                   'kl_truNzvsmpNz': self.kl_truNzvsmpNz,
                   'kl_truNzvintNz': self.kl_truNzvintNz,
                   'kl_truNzvmmlNz': self.kl_truNzvmmlNz
              }
        if self.meta.plotonly == False:
            with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as statchains:
                cpkl.dump(outdict,statchains)

    def compute(self, ydata):#ntimes*nwalkers*nbins

        self.ydata = ydata
        self.eydata = np.exp(self.ydata)
        y = np.swapaxes(self.ydata.T,0,1).T#nwalkers*nbins*ntimes
        ey = np.swapaxes(self.eydata.T,0,1).T#np.exp(y)

        if self.meta.logtruNz is None:
            my = np.array([[[sum(by)/len(by)]*self.meta.ntimes for by in wy] for wy in y])#nwalkers*nbins*ntimes
            mey = np.array([[[sum(bey)/len(bey)]*self.meta.ntimes for bey in wey] for wey in ey])#nwalkers*nbins*ntimes
        else:
            my = np.array([[[k]*self.meta.ntimes for k in self.meta.logtruNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes
            mey = np.array([[[k]*self.meta.ntimes for k in self.meta.truNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes

            for w in xrange(self.meta.nwalkers):

                with open(os.path.join(self.meta.topdir,'samples.csv'),'ab') as csvfile:
#                   print(type(self.ydata))
#                   print(type(self.ydata[0]))
                    out = csv.writer(csvfile,delimiter=' ')
                    out.writerows(self.ydata[w])#[[x for x in row] for row in self.ydata])

                for x in xrange(self.meta.ntimes):
                    #ulog = np.exp(self.chains[w][x])
                    #ulogpz = ulog/sum(ulog)
                    #logpz = np.log(ulogpz)
                    pq,qp = self.calckl(self.ydata[w][x],self.meta.logtruNz)
                    self.kl_smpNzvtruNz.append(pq)
                    self.kl_truNzvsmpNz.append(qp)

        self.sy = np.swapaxes((y-my),1,2)#nwalkers*ntimes*nbins to #nwalkers*nbins*ntimes
        self.sey = np.swapaxes((ey-mey),1,2)

        self.var_ls = self.calcvar(self.var_ls,self.sy)
        self.var_s = self.calcvar(self.var_s,self.sey)
        self.chi_ls = self.calcchi(self.chi_ls,self.sy,self.ydata)
        self.chi_s = self.calcchi(self.chi_s,self.sey,self.eydata)

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as indict:
            outdict = cpkl.load(indict)

        outdict['kl_smpNzvtruNz'] = np.array(self.kl_smpNzvtruNz)
        outdict['kl_truNzvsmpNz'] = np.array(self.kl_truNzvsmpNz)
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
#         return { 'vslogstkNz': self.vslogstkNz,
#                'vsstkNz': self.vsstkNz,
#                'vslogmapNz': self.vslogmapNz,
#                'vsmapNz': self.vsmapNz,
#                'vslogexpNz': self.vslogexpNz,
#                'vsexpNz': self.vsexpNz,
#                'tot_var_ls': self.tot_var_ls,
#                'tot_var_s': self.tot_var_s,
#                'var_ls': self.var_ls,
#                'var_s': self.var_s
#               }

    # variance of samples
    def calcvar(self,var,s):

        ans = np.average([[np.dot(s[w][i],s[w][i]) for i in xrange(len(s[w]))] for w in xrange(len(s))])
        var.append(ans)
#         var_ls = np.average([[np.dot(self.sy[w][i],self.sy[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
#         var_s = np.average([[np.dot(self.sey[w][i],self.sey[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
#         self.var_ls.append(var_ls)
#         self.var_s.append(var_s)
        #self.tot_var_ls = self.tot_var_ls+var_ls
        #self.tot_var_s = self.tot_var_s+var_s
        #print(self.meta.name+' var_ls='+str(self.var_ls))
        #print(self.meta.name+' var_s='+str(self.var_s))
        return(var)

    # chi^2 (or Wald test) of samples
    def calcchi(self,var,s,data):

        v = np.sum([np.average([statistics.variance(walk) for walk in data.T[b]]) for b in xrange(len(data.T))])#abs(np.linalg.det(np.cov(flatdata)))
        ans = np.average(s**2)/v
        var.append(ans)

#         flatdata = np.array([self.ydata.T[b].flatten() for b in xrange(self.meta.nbins)])
#         eflatdata = np.exp(flatdata)

#         vy = np.sum([np.average([statistics.variance(walk) for walk in self.ydata.T[b]]) for b in xrange(self.meta.nbins)])#abs(np.linalg.det(np.cov(flatdata)))
#         vey = np.sum([np.average([statistics.variance(walk) for walk in self.eydata.T[b]]) for b in xrange(self.meta.nbins)])#abs(np.linalg.det(np.cov(eflatdata)))

#         chi_ls = np.average(self.sy**2)/vy#np.average(sp.stats.chisquare(flatdata.T)[0])#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vy)
#         chi_s = np.average(self.sey**2)/vey#np.average(sp.stats.chisquare(eflatdata.T)[0])#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vey)
#         self.chi_ls.append(chi_ls)
#         self.chi_s.append(chi_s)
#         #self.tot_chi_ls = self.tot_chi_ls+chi_ls
#         #self.tot_chi_s = self.tot_chi_s+chi_s
#         print(self.meta.name+' chi_ls='+str(self.chi_ls))
#         print(self.meta.name+' chi_s='+str(self.chi_s))
        return(var)

    # KL Divergence test
    def calckl(self,lqn,lpn):
        pn = np.exp(lpn)*self.meta.bindifs
        qn = np.exp(lqn)*self.meta.bindifs
        p = pn/np.sum(pn)
        q = qn/np.sum(qn)
        logp = np.log(p)
        logq = np.log(q)
        klpq = np.sum(p*(logp-logq))
        klqp = np.sum(q*(logq-logp))
        return(round(klpq,3),round(klqp,3))

class stat_probs(calcstats):
    """
    calculates statistics requiring only probabilities:  log posterior probability for alternatives, variance of probabilities
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        #self.summary = 0
        self.name = 'probs'

#         # calculating log likelihood ratio test statistic for each relative to truth (and true relative to prior)
        if self.meta.logtruNz is not None:
            self.lp_truNz = self.meta.postdist.lnprob(self.meta.logtruNz)
            self.lik_truNz = self.meta.postdist.lnlike(self.meta.logtruNz)
        else:
            self.lp_truNz = self.meta.postdist.lnprob(self.meta.mean)
            self.lik_truNz = self.meta.postdist.lnlike(self.meta.mean)

#         self.lp_stkNz = self.meta.postdist.lnprob(self.meta.logstkNz)
#         self.lp_mapNz = self.meta.postdist.lnprob(self.meta.logmapNz)
# #         self.lp_expNz = self.meta.postdist.lnprob(self.meta.logexpNz)
        self.lp_mmlNz = self.meta.postdist.lnprob(self.meta.logmmlNz)

        self.var_y = []

        outdict = {'var_y': self.var_y,
                 'lp_truNz': self.lp_truNz,
                 'lp_mmlNz': self.lp_mmlNz
              }

        if self.meta.plotonly == False:
            with open(os.path.join(self.meta.topdir,'stat_probs.p'),'wb') as statprobs:
                cpkl.dump(outdict,statprobs)

    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w]) for w in xrange(self.meta.nwalkers)])/self.meta.nwalkers
        #self.llr_smpNz.append((2.*np.max(lik_y)-2.*self.ll_truNz))
        self.var_y.append(var_y)
        # self.summary = self.summary+var_y

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as indict:
            outdict = cpkl.load(indict)

        outdict['var_y'] = self.var_y

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'wb') as statprobs:
            cpkl.dump(outdict,statprobs)

#         return { #'summary': self.summary,
#                  'var_y': self.var_y,
#                  'lp_truNz': self.lp_truNz,
# #                  'lp_stkNz': self.lp_stkNz,
# #                  'lp_mapNz': self.lp_mapNz,
# # #                  'lp_expNz': self.lp_expNz
#                  'lp_mmlNz': self.lp_mmlNz
#                }

class stat_fracs(calcstats):
    """
    calculates summary statistics on acceptance fractions
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.var_y = []
        self.name = 'fracs'
    def compute(self, ydata):
        y = ydata.T
        var_y = statistics.variance(y)
        self.var_y.append(var_y)
        return {'var_y': self.var_y}

class stat_times(calcstats):
    """
    calculates summary statistics on autocorrelation times
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.var_y = []
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = np.var(y)
        self.var_y.append(var_y)
        return {'var_y': self.var_y}

# calculate autocorrelation times since emcee sometimes fails
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
