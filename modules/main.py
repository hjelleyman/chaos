"""
"""
from modules.helper import *
import modules.lyapunov as lyp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

tolerence = 1e-1
imagefolder = 'images/'

def main():
    x = np.linspace(-10,10,10)
    y = np.linspace(-10,10,10)
    amin = 0.7
    amax = 2
    lmin = 0.1
    lmax = 1
    da = 0.01
    dl = 0.01
    ablocks = 10
    lblocks = 10
    
    l = np.arange(lmin,lmax,dl)
    a = np.arange(amin,amax,da)
    
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
            
    print(lblocked)
    
def plot_initial_conditions(x,y,l,a):
    fig, ax = plt.subplots(figsize = (5,5))
    plt.title('Initial conditions grid on the x-y plane')
    plt.scatter(x,y)
    plt.savefig(imagefolder + 'initial_xy.eps')
    plt.show()

    fig, ax = plt.subplots(figsize = (5,5))
    plt.title('Initial conditions grid on the l-a plane')
    plt.scatter(l,a, s = 0.1)
    plt.savefig(imagefolder + 'initial_la.eps')
    plt.show()

if __name__ == '__main__':
    main()