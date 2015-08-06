import os
import timeit
#import emcee
#import StringIO

import hickle as hkl

def sampling(sampler,ivals,miniters,thinto):
    start_time = timeit.default_timer()
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(ivals,miniters,thin=thinto)
    ovals = [walk[-1] for walk in sampler.chain]
    times = sampler.get_autocorr_time()#ndims
    fracs = sampler.acceptance_fraction#nwalkers
    probs = sampler.lnprobability#np.swapaxes(sampler.lnprobability,0,1)#niters*nwalkers
    chains = sampler.chain#np.swapaxes(sampler.chain,0,1)#niters*nwalkers*ndims
    elapsed = timeit.default_timer() - start_time
    #print(setups[n]+' '+str(ndims[k])+' complete')
    outputs = [times,fracs,probs,chains]
    return ovals,outputs,elapsed

# def run_samplings(run):
#   run.samplings()

# # class permcmc(object):

# #   def __init__(self,meta,p_run,s_run,n_run,i_run,r,idinfo):

# #     self.idinfo = idinfo
# #     self.r = r

# def sampling(self,sampler,ivals,miniters,thinto):
#     #start_time = timeit.default_timer()
#     sampler.reset()
#     pos, prob, state = sampler.run_mcmc(ivals,miniters,thin=thinto)
#     ovals = [walk[-1] for walk in sampler.chain]
#     times = sampler.get_autocorr_time()#ndims
#     fracs = sampler.acceptance_fraction#nwalkers
#     probs = sampler.lnprobability#np.swapaxes(sampler.lnprobability,0,1)#niters*nwalkers
#     chains = sampler.chain#np.swapaxes(sampler.chain,0,1)#niters*nwalkers*ndims
#     #elapsed = timeit.default_timer() - start_time
#     #print(setups[n]+' '+str(ndims[k])+' complete')
#     #ovals = ivals
#     outputs = [times,fracs,probs,chains]
#     #self.duration = elapsed
#     return ovals,outputs#,duration

# #   def saving(self,r):
# #     ovals,outputs = self.sampling(n_run.sampler,ivals,meta.miniters,meta.thinto)

# #     for t in meta.statnos:
# #         outfile = open(os.path.join(i_run.topdir_i,meta.outnames[t][r]),'w')
# #         hkl.dump(outputs[t],outfile,mode='w')
# #         outfile.close()
# #         meta.queues[t+1].put(self.idinfo)

# #     return

# # def saving(r):
# #     ivals,outputs,elapsed = sampling(sampler,ivals)
# #     for t in meta.statnos:
# #         outfile = open(os.path.join(self.topdir_i,meta.outnames[t][r]),'w')
# #         hkl.dump(outputs[t],outfile,mode='w')
# #         outfile.close()
# #         queues[i].put((p_run.p,s_run.s,n_run.n,self.i,r))
# #     calctimer = open(calctime,'a')
# #     calctimer.write(str(timeit.default_timer())+' '+str((p,s,n,t,r))+' '+str(elapsed)+'\n')
# #     calctimer.close()

# def fplot():
