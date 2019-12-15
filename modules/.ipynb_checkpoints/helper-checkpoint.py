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
            if i%100==0:
                update_progress(i/n)
    if nosave:
        for i in range(1,int(n)):
            x,y = maping(x,y,l,a)
            if i%100==0:
                update_progress(i/n)
        X = x
        Y = y
    return X,Y

def GSchmidt(A):
    """Performs Gramm-Schmidt Orthogonalisation"""
    
    u = A[:,0]
    v = A[:,1]
    
    A[:,1] = v - proj(u,v)
    
    A[:,0] = u/np.linalg.norm(u)
    A[:,1] = A[:,1]/np.linalg.norm(A[:,1])
    
    return A
    
    
def proj(u,v):
    """projects v onto u"""
    return (np.dot(u,v)/np.dot(u,u))*u




def vectorGSchmidt(A):
    """Gramm Schmidt Orthognalisation"""
    u = A[:,0]
    v = A[:,1]
    
    def _proj(u,v):
        """projects v onto u"""
        return (np.einsum('i...,i...->...',u,v)/np.einsum('i...,i...->...',u,u))*u
    
    v = v - _proj(u,v)
    
    u = u/np.linalg.norm(u, axis=0)
    v = v/np.linalg.norm(v, axis=0)
    
    
    A[:,0] = u
    A[:,1] = v
    
    return A

import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)