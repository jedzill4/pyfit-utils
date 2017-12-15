
import numpy as np


def to_arrhenius(axis):
    yticks = axis.get_yticks()
    floor10 = np.power(10, np.floor(np.log10(yticks)))
    ceil10 = np.power(10, np.ceil(np.log10(yticks)))

    ymin = np.nanmin(floor10[floor10 > 0])
    ymax = np.nanmax(ceil10)
#    axis.set_ylim(ymin,ymax)
    
    axis.set_yscale('log')
    ticks = axis.get_xticks()
    lims = axis.get_xlim()
    axis.set_xticklabels([r'$\frac{1}{%.1f}$'%(1/val) for val in ticks[::]])
    axis.grid(True, which='minor')     
    axis.spines["top"].set_visible(True)
    
    twinax = axis.twiny()
    twinax.set_xlabel('Mean Temperature [C$^o$]')
    twinax.spines["bottom"].set_visible(False)
    twinax.spines["left"].set_visible(False)
    twinax.spines["right"].set_visible(False)
    twinax.set_xlim(lims)
    twinax.set_xticklabels(['%.1f'%i for i in (1/ticks[::]-273.15)])

