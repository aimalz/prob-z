import numpy as np
import sys

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

import utils as u

class z_dens(object):

    def __init__(self, data_list, hpvar, truth_fun=None, vb=True):
        """
        An object representing the redshift density function (normalized redshift distribution function)

        Parameters
        ----------
        data_list: list
            list containing bin endpoints, logged interim prior bin values, and logged interim posterior PDF bin values
        hpvar: ndarray
            array of covariance matrix for hyperprior distribution
        truth_fun: function, optional
            function taking ndarray of redshifts to ndarray of true redshift density function values
        vb: boolean
            True to print progress messages to stdout, False to suppress
        """

        self.bin_ends = np.array(data_list[0])
        self.log_int_dens = np.array(data_list[1])
        self.log_pdfs = np.array(data_list[2:])

        if vb:
            print(str(len(self.bin_ends)-1)+' bins, '+str(len(self.log_pdfs))+' interim posterior PDFs')

        self.bin_range = self.bin_ends[:-1]-self.bin_ends[0]
        self.bin_mids = (self.bin_ends[1:]+self.bin_ends[:-1])/2.
        self.bin_difs = self.bin_ends[1:]-self.bin_ends[:-1]
        self.n_bins = len(self.bin_mids)

        self.n_pdfs = len(self.log_pdfs)
        self.pdfs = np.exp(self.log_pdfs)
        self.int_dens = np.exp(self.log_int_dens)

        self.hyper_prior_var = hpvar

        self.truth_fun = truth_fun

        return

    def mmap(self):
        """
        Calculates the marginalized maximum a posteriori estimator of the redshift density function

        Returns
        -------
        log_mmap_dens: ndarray
            array of redshift density function bin values
        """

        mmap_dist = np.zeros(self.n_bins)+sys.float_info.epsilon
        pdf_mmaps = [np.argmax(l) for l in self.log_pdfs]
        for m in pdf_mmaps:
              mmap_dist[m] += 1./self.bin_difs[m]

        self.mmap_dens = mmap_dist/self.n_pdfs
        self.log_mmap_dens = np.log(self.mmap_dens)

        return self.mmap_dens

    def plot(self, plot_loc=''):
        """
        Plots all available estimators of the redshift density function.
        """

        # set up for better looking plots
        title = 10
        label = 10
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['axes.titlesize'] = title
        mpl.rcParams['axes.labelsize'] = label
        mpl.rcParams['figure.subplot.left'] = 0.2
        mpl.rcParams['figure.subplot.right'] = 0.9
        mpl.rcParams['figure.subplot.bottom'] = 0.2
        mpl.rcParams['figure.subplot.top'] = 0.9
        mpl.rcParams['figure.subplot.wspace'] = 0.5
        mpl.rcParams['figure.subplot.hspace'] = 0.5

        self.f = plt.figure(figsize=(5, 10))
        self.sps = [self.f.add_subplot(2, 1, l+1) for l in xrange(0, 2)]
        self.f.subplots_adjust(hspace=0, wspace=0)
        sps_log = self.sps[0]
        sps = self.sps[1]

        sps_log.set_xlim(self.bin_ends[0]-max(self.bin_difs), self.bin_ends[-1]+max(self.bin_difs))
        sps_log.set_ylabel(r'$\ln n(z)$')
        sps.set_xlim(self.bin_ends[0]-max(self.bin_difs), self.bin_ends[-1]+max(self.bin_difs))
        sps.set_xlabel(r'$z$')
        sps.set_ylabel(r'$n(z)$')
        sps.ticklabel_format(style='sci',axis='y')

        if self.truth_fun is not None:
            z = np.linspace(self.bin_ends[0], self.bin_ends[-1], self.n_bins**2)
            fun = self.truth_fun(z)
            log_fun = np.log(fun)
            sps.plot(z, fun, linestyle=u.s_tru, color=u.c_tru, alpha=u.a_tru, linewidth=u.w_tru, label=u.l_tru+u.nz)
            sps_log.plot(z, log_fun, linestyle=u.s_tru, color=u.c_tru, alpha=u.a_tru, linewidth=u.w_tru, label=u.l_tru+u.lnz)

        if self.mmap_dens is not None:
            u.plot_step(sps, self.bin_ends, self.mmap_dens, s=u.s_map, c=u.c_map, a=u.a_map, w=u.w_map, d=u.d_map, l=u.l_map+u.nz)
            u.plot_step(sps_log, self.bin_ends, self.log_mmap_dens, s=u.s_map, c=u.c_map, a=u.a_map, w=u.w_map, d=u.d_map, l=u.l_map+u.lnz)

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.savefig(plot_loc+'plot.png')
