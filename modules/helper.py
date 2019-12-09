import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def maping(x,y,l,a):
    """Applies one itteration of the map."""
    z = x + y*1j
    z1 = (1-l+l*np.abs(z)**a)*((z)/(np.abs(z)))**2 + 1
    return np.real(z1), np.imag(z1)

def jacobian(x,y,l,a):
    """Computes the Jacobian of the map."""
    J = np.zeros([*x.shape,2,2])
    
    J[...,0,0] = x*a*l*(x**2+y**2)**(a/2-2)*(x**2-y**2) + (1-l+l*(x**2+y**2)**(a/2))*(2*x/(x**2+y**2) -2*x*(x**2-y**2)/(x**2+y**2)**2)

    J[...,0,1] = y*a*l*(x**2+y**2)**(a/2-2)*(x**2-y**2) + (1-l+l*(x**2+y**2)**(a/2))*(-2*y/(x**2+y**2) -2*y*(x**2-y**2)/(x**2+y**2)**2)

    J[...,1,0] = 2*y*x**2*a*l*(x**2+y**2)**(a/2-2) + (1-l+l*(x**2+y**2)**(a/2))*(2*y/(x**2+y**2)-4*x**2*y/(x**2+y**2)**2)

    J[...,1,1] = 2*x*y**2*a*l*(x**2+y**2)**(a/2-2) + (1-l+l*(x**2+y**2)**(a/2))*(2*x/(x**2+y**2)-4*y**2*x/(x**2+y**2)**2)
    
    return J

def repeatmap(x,y,l,a,n=1, nosave=False):
    if not nosave:
        X = np.zeros([n,*x.shape])
        Y = np.zeros([n,*y.shape])
    
        X[0] = x
        Y[0] = y

        for i in range(1,int(n)):
            x,y = maping(x,y,l,a)
            X[i] = x
            Y[i] = y
            
    if nosave:
        for i in range(1,int(n)):
            x,y = maping(x,y,l,a)
        X = x
        Y = y
    return X,Y