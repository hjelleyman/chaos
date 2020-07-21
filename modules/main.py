"""
"""
from modules.helper import *
import modules.lyapunov as lyp
# from helper import *
# import lyapunov as lyp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import itertools

import matplotlib._color_data as mcd
xkcd = list(mcd.XKCD_COLORS.values())

tolerence = 1e-1
imagefolder = 'images/'

def main(amin = 0.7, amax = 2, lmin = 0.1, lmax = 1, da = 0.01, dl =0.01, ablocks = 20, lblocks = 20, n_transient = 100000, n_attractor = 100000):
    x = np.linspace(-10,10,10)
    y = np.linspace(-10,10,10)
    
    l = np.arange(lmin,lmax,dl)
    a = np.arange(amin,amax,da)
    
    
    lyapunov_1 = xr.DataArray(np.zeros([len(x),len(y),len(l),len(a)]),coords={'x':x,'y':y,'l':l,'a':a}, dims = ['x', 'y', 'l', 'a'])
    lyapunov_2 = xr.DataArray(np.zeros([len(x),len(y),len(l),len(a)]),coords={'x':x,'y':y,'l':l,'a':a}, dims = ['x', 'y', 'l', 'a'])
    
    n_transient = int(n_transient)
    n_attractor = int(n_attractor)
#     save_initial_coditions_to_file(x,y,l,a)
#     plot_initial_conditions(*np.meshgrid(x,y,l,a))
    
    lblocked, ablocked = block_la(l,a,lblocks,ablocks)
            
    # llength = (a.max()-a.min())/(l.max()-l.min())*5
    # alength = (l.max()-l.min())/(a.max()-a.min())*5
    i = 0
    for lblock,ablock in list(itertools.product(lblocked,ablocked))[:]:
        t0 = time.time()
        i += 1
        
        xi,yi,lblock,ablock = np.meshgrid(x,y,lblock,ablock)


        system = lyp.system(xi,yi,lblock,ablock,n_transient,n_attractor)
        system.calcLyapunov()
        lyapunov_1.loc[yi[:,0,0,0],xi[0,:,0,0],lblock[0,0,:,0],ablock[0,0,0,:]] = system.lyapunov_1
        lyapunov_2.loc[yi[:,0,0,0],xi[0,:,0,0],lblock[0,0,:,0],ablock[0,0,0,:]] = system.lyapunov_2
        system = None
        print(f'{i}/{lblocks*ablocks} in {time.time()-t0:.2f}')
    return (lyapunov_1,lyapunov_2)
        
        
def plot_initial_conditions(x,y,l,a):
    """Plots grids of the initial conditions used for the analysis."""
    xlength = (x.max()-x.min())/(y.max()-y.min())*5
    ylength = (y.max()-y.min())/(x.max()-x.min())*5

    fig, ax = plt.subplots(figsize = (xlength,ylength))
    plt.title('Initial conditions grid on the x-y plane')
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(imagefolder + 'initial_xy.pdf')
    plt.savefig(imagefolder + 'initial_xy.eps')
    plt.show()

    llength = (a.max()-a.min())/(l.max()-l.min())*5
    alength = (l.max()-l.min())/(a.max()-a.min())*5

    fig, ax = plt.subplots(figsize = (llength,alength))
    plt.title('Initial conditions grid on the $\lambda$-a plane')
    plt.scatter(l,a, s = 0.1)
    plt.xlabel('$\lambda$')
    plt.ylabel('a')
    plt.savefig(imagefolder + 'initial_la.pdf')
    plt.savefig(imagefolder + 'initial_la.eps')
    plt.show()
    
    
def block_la(l,a,lblocks,ablocks):
    
    lblocked = []
    ablocked = []
    
    for i in range(ablocks):
        if i != ablocks - 1:
            ablocked += [a[i*len(a)//ablocks:(i+1)*len(a)//ablocks]]
        else:
            ablocked += [a[i*len(a)//ablocks:]]
    for i in range(lblocks):
        if i != lblocks - 1:
            lblocked += [l[i*len(l)//lblocks:(i+1)*len(l)//lblocks]]
        else:
            lblocked += [l[i*len(l)//lblocks:]]
            
    return lblocked, ablocked
            
            
def save_initial_coditions_to_file(x,y,l,a):
    with open('data/initial_conditions.txt','w+')as f:
        f.write('x:\n')
        f.write(str(list(x))+'\n\n')
        f.write('y:\n')
        f.write(str(list(y))+'\n\n')
        f.write('lambda:\n')
        f.write(str(list(l))+'\n\n')
        f.write('a:\n')
        f.write(str(list(a))+'\n\n')

if __name__ == '__main__':
    main()