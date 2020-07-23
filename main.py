from modules.helper import *
import modules.lyapunov as lyp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

tolerence = 1e-1
imagefolder = 'images/'

import modules.main as main
import time

start = time.time()
l1,l2 = main.main(amin = 0.7, amax = 2, lmin = 0.1, lmax = 1, da = 0.01, dl =0.01, ablocks = 10, lblocks = 10, n_transient = 1e6, n_attractor = 1e4)
print(f'Total run time:\t{time.time()-start:.2f}')

l1.to_netcdf('data/lyapunov1_large_6_4_highres.nc')
l2.to_netcdf('data/lyapunov2_large_6_4_highres.nc')