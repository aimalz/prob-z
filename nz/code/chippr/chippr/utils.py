global lnz, nz
lnz, nz = r'$\ln[n(z)]$', r'$n(z)$'

global s_tru, w_tru, a_tru, c_tru, d_tru, l_tru
s_tru, w_tru, a_tru, c_tru, d_tru, l_tru = '--', 0.5, 1., 'k', [(0,(1,0.0001))], 'True '
# global s_int,w_int,a_int,c_int,d_int,l_int
# s_int,w_int,a_int,c_int,d_int,l_int = '--',0.5,0.5,'k',[(0,(1,0.0001))],'Interim '
# global s_stk,w_stk,a_stk,c_stk,d_stk,l_stk
# s_stk,w_stk,a_stk,c_stk,d_stk,l_stk = '--',1.5,1.,'k',[(0,(3,2))],'Stacked '#[(0,(2,1))]
global s_map, w_map, a_map, c_map, d_map, l_map
s_map, w_map, a_map, c_map, d_map, l_map = '--', 1., 1., 'k', [(0,(3,2))], 'MMAP '#[(0,(1,1,3,1))]
# global s_exp,w_exp,a_exp,c_exp,d_exp,l_exp
# s_exp,w_exp,a_exp,c_exp,d_exp,l_exp = '--',1.,1.,'k',[(0,(1,1))],'MExp '#[(0,(3,3,1,3))]
# global s_mml,w_mml,a_mml,c_mml,d_mml,l_mml
# s_mml,w_mml,a_mml,c_mml,d_mml,l_mml = '--',2.,1.,'k',[(0,(1,1))],'MMLE '#[(0,(3,2))]
# global s_smp,w_smp,a_smp,c_smp,d,smp,l_smp
# s_smp,w_smp,a_smp,c_smp,d_smp,l_smp = '--',1.,1.,'k',[(0,(1,0.0001))],'Sampled '
# global s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe
# s_bfe,w_bfe,a_bfe,c_bfe,d_bfe,l_bfe = '--',2.,1.,'k',[(0,(1,0.0001))],'Mean of\n Samples '

def plot_step(sub_plot, bin_ends, plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Plots a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    plot_h(sub_plot, bin_ends, plot, s, c, a, w, d, l, r)
    plot_v(sub_plot, bin_ends, plot, s, c, a, w, d, r)

def plot_h(sub_plot, bin_ends,plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], l=None, r=False):
    """
    Helper function to plot horizontal lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    l: string, optional
        label for function
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.hlines(plot,
                   bin_ends[:-1],
                   bin_ends[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   label=l,
                   rasterized=r)
def plot_v(sub_plot, bin_ends, plot, s='--', c='k', a=1, w=1, d=[(0,(1,0.0001))], r=False):
    """
    Helper function to plot vertical lines of a step function

    Parameters
    ----------
    sub_plot: matplotlib.pyplot subplot object
        subplot into which step function is drawn
    bin_ends: list or ndarray
        list or array of endpoints of bins
    plot: list or ndarray
        list or array of values within each bin
    s: string, optional
        matplotlib.pyplot linestyle
    c: string, optional
        matplotlib.pyplot color
    a: int or float, [0., 1.], optional
        matplotlib.pyplot alpha (transparency)
    w: int or float, optional
        matplotlib.pyplot linewidth
    d: list of tuple, optional
        matplotlib.pyplot dash style, of form [(start_point, (points_on, points_off, ...))]
    r: boolean, optional
        True for rasterized, False for vectorized
    """

    sub_plot.vlines(bin_ends[1:-1],
                   plot[:-1],
                   plot[1:],
                   linewidth=w,
                   linestyle=s,
                   dashes=d,
                   color=c,
                   alpha=a,
                   rasterized=r)