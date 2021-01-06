import numpy as np
import xarray as xr
import itertools
import time

from modules import *


amin = 0.5
amax = 1
lmin = 0.5
lmax = 1
da = 0.005
dl = 0.005
ablocks = 10
lblocks = 10

x = np.linspace(-10, 10, 10)
y = np.linspace(0, 10, 10)

l = np.arange(lmin, lmax, dl)
a = np.arange(amin, amax, da)

n_transient = int(1e3)
n_attractor = int(1e3)

lblocked, ablocked = main.block_la(l, a, lblocks, ablocks)

lyapunov_1 = xr.DataArray(np.zeros([len(x), len(y), len(l), len(a)]), coords={
                          'x': x, 'y': y, 'l': l, 'a': a}, dims=['x', 'y', 'l', 'a'])
lyapunov_2 = xr.DataArray(np.zeros([len(x), len(y), len(l), len(a)]), coords={
                          'x': x, 'y': y, 'l': l, 'a': a}, dims=['x', 'y', 'l', 'a'])

i = 0
for lblock, ablock in list(itertools.product(lblocked, ablocked))[:]:
    t0 = time.time()
    i += 1

    xi, yi, lblock, ablock = np.meshgrid(x, y, lblock, ablock)

    system = gl.system(xi, yi, lblock, ablock, n_transient, n_attractor)
    system.calcLyapunov()
    lyapunov_1.loc[xi[:, 0, 0, 0], yi[0, :, 0, 0],
                   lblock[0, 0, :, 0], ablock[0, 0, 0, :]] = system.lyapunov_1
    lyapunov_2.loc[xi[:, 0, 0, 0], yi[0, :, 0, 0],
                   lblock[0, 0, :, 0], ablock[0, 0, 0, :]] = system.lyapunov_2
    del system
    print(f'{i}/{lblocks*ablocks} in {time.time()-t0:.2f}')


lyapunov_1.to_netcdf('data2/lyapunov1_small_6_4_highres_200820.nc')
lyapunov_2.to_netcdf('data2/lyapunov2_small_6_4_highres_200820.nc')

amin = 0.7
amax = 2
lmin = 0.1
lmax = 1
da = 0.01
dl = 0.01
ablocks = 15
lblocks = 15

x = np.linspace(-10, 10, 10)
y = np.linspace(0, 10, 10)

l = np.arange(lmin, lmax, dl)
a = np.arange(amin, amax, da)

n_transient = int(1e6)
n_attractor = int(1e4)

lblocked, ablocked = main.block_la(l, a, lblocks, ablocks)

lyapunov_1 = xr.DataArray(np.zeros([len(x), len(y), len(l), len(a)]), coords={
                          'x': x, 'y': y, 'l': l, 'a': a}, dims=['x', 'y', 'l', 'a'])
lyapunov_2 = xr.DataArray(np.zeros([len(x), len(y), len(l), len(a)]), coords={
                          'x': x, 'y': y, 'l': l, 'a': a}, dims=['x', 'y', 'l', 'a'])

i = 0
for lblock, ablock in list(itertools.product(lblocked, ablocked))[:]:
    t0 = time.time()
    i += 1

    xi, yi, lblock, ablock = np.meshgrid(x, y, lblock, ablock)

    system = gl.system(xi, yi, lblock, ablock, n_transient, n_attractor)
    system.calcLyapunov()
    lyapunov_1.loc[xi[:, 0, 0, 0], yi[0, :, 0, 0],
                   lblock[0, 0, :, 0], ablock[0, 0, 0, :]] = system.lyapunov_1
    lyapunov_2.loc[xi[:, 0, 0, 0], yi[0, :, 0, 0],
                   lblock[0, 0, :, 0], ablock[0, 0, 0, :]] = system.lyapunov_2
    del system
    print(f'{i}/{lblocks*ablocks} in {time.time()-t0:.2f}')


lyapunov_1.to_netcdf('data2/lyapunov1_large_6_4_highres_200820.nc')
lyapunov_2.to_netcdf('data2/lyapunov2_large_6_4_highres_200820.nc')
