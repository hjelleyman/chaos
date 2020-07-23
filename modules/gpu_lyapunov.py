import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import time
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from numba import njit, cuda


chaos = np.loadtxt('data/chaos_al.dat')

wild_chaos = np.loadtxt('data/wildchaos_al.dat')

@njit(parallel=True,fastmath=True)
def _maping(x,y,l,a):
    """Applies one itteration of the map."""
    newx = (x**2 *(l* ((x**2 + y**2)**(a/2) - 1) + 2) - l * y**2 *((x**2 + y**2)**(a/2) - 1))/(x**2 + y**2) 
    newy = (2 * x* y *(l* ((x**2 + y**2)**(a/2) - 1) + 1))/(x**2 + y**2)
    return newx, newy

@njit(fastmath=True)
def _repeatmap_save(x,y,l,a, n,X,Y):
    X[0] = x
    Y[0] = y
    for i in range(1,int(n)):
        x,y = _maping(x,y,l,a)
        X[i] = x
        Y[i] = y
    newx = X
    newy = Y
    return [newx,newy]

@cuda.jit
def _repeatmap_nosave(X,Y,L,A, n):
    pos = cuda.grid(1)
    if pos < X.size:
        x = X[pos]
        y = Y[pos]
        l = L[pos]
        a = A[pos]
        for i in range(1,int(n)):
            newx = (x**2 *(l* ((x**2 + y**2)**(a/2) - 1) + 2) - l * y**2 *((x**2 + y**2)**(a/2) - 1))/(x**2 + y**2) 
            newy = (2 * x* y *(l* ((x**2 + y**2)**(a/2) - 1) + 1))/(x**2 + y**2)
            x = newx
            y = newy
        X[pos] = x
        Y[pos] = y

def _jacobian(x,y,l,a,J):
    shape = J.shape
    x_flat = x.reshape(x.shape[0],np.prod(x.shape[1:]))
    y_flat = y.reshape(x.shape[0],np.prod(x.shape[1:]))
    l_flat = l.flatten()
    a_flat = a.flatten()
    J0 = np.empty([J.shape[0],np.prod(J.shape[1:-2])])
    J1 = np.empty([J.shape[0],np.prod(J.shape[1:-2])])
    J2 = np.empty([J.shape[0],np.prod(J.shape[1:-2])])
    J3 = np.empty([J.shape[0],np.prod(J.shape[1:-2])])

    threadsperblock = 32 
    blockspergrid = (x_flat.shape[0] + (threadsperblock - 1)) // threadsperblock

    _j1[blockspergrid, threadsperblock](x_flat,y_flat,l_flat,a_flat,J0)
    _j2[blockspergrid, threadsperblock](x_flat,y_flat,l_flat,a_flat,J1)
    _j3[blockspergrid, threadsperblock](x_flat,y_flat,l_flat,a_flat,J2)
    _j4[blockspergrid, threadsperblock](x_flat,y_flat,l_flat,a_flat,J3)

    J[...,0,0] = J0.reshape(shape[:-2])
    J[...,0,1] = J1.reshape(shape[:-2])
    J[...,1,0] = J2.reshape(shape[:-2])
    J[...,1,1] = J3.reshape(shape[:-2])

    return J

@cuda.jit
def _j1(X,Y,L,A,J):
    pos = cuda.grid(1)
    len_t = X.shape[0]
    n = X.shape[1]
    if (pos < len_t):
        for i in range(n):
            x = X[pos,i]
            y = Y[pos,i]
            l = L[i]
            a = A[i]
            J[pos,i] = x*a*l*(x**2+y**2)**(a/2-2)*(x**2-y**2) + (1-l+l*(x**2+y**2)**(a/2))*(2*x/(x**2+y**2) -2*x*(x**2-y**2)/(x**2+y**2)**2)
@cuda.jit
def _j2(X,Y,L,A,J):
    pos = cuda.grid(1)
    len_t = X.shape[0]
    n = X.shape[1]
    if (pos < len_t):
        for i in range(n):
            x = X[pos,i]
            y = Y[pos,i]
            l = L[i]
            a = A[i]
            J[pos,i] = y*a*l*(x**2+y**2)**(a/2-2)*(x**2-y**2) + (1-l+l*(x**2+y**2)**(a/2))*(-2*y/(x**2+y**2) -2*y*(x**2-y**2)/(x**2+y**2)**2)
@cuda.jit
def _j3(X,Y,L,A,J):
    pos = cuda.grid(1)
    len_t = X.shape[0]
    n = X.shape[1]
    if (pos < len_t):
        for i in range(n):
            x = X[pos,i]
            y = Y[pos,i]
            l = L[i]
            a = A[i]
            J[pos,i] = 2*y*x**2*a*l*(x**2+y**2)**(a/2-2) + (1-l+l*(x**2+y**2)**(a/2))*(2*y/(x**2+y**2)-4*x**2*y/(x**2+y**2)**2)
@cuda.jit
def _j4(X,Y,L,A,J):
    pos = cuda.grid(1)
    len_t = X.shape[0]
    n = X.shape[1]
    if (pos < len_t):
        for i in range(n):
            x = X[pos,i]
            y = Y[pos,i]
            l = L[i]
            a = A[i]
            J[pos,i] = 2*x*y**2*a*l*(x**2+y**2)**(a/2-2) + (1-l+l*(x**2+y**2)**(a/2))*(2*x/(x**2+y**2)-4*y**2*x/(x**2+y**2)**2)



@cuda.jit
def angle_matrix_rotations(jacobian_matrix, angle_matrix, out):
    """Perform square matrix multiplication of C = A * B
    """
    n = cuda.grid(1)
    if n < angle_matrix.shape[2]:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    jacobian = jacobian_matrix[n,i,k]
                    angle = angle_matrix[k,j,n]
                    out[i,j,n] += jacobian * angle


class system:
    """system object which can contain coordinates and things like that."""
    
    def __init__(self,x,y,l,a,n_transient,n_attractor):
        self.x = x
        self.y = y
        self.l = l
        self.a = a
        self.n_transient = n_transient
        self.n_attractor = n_attractor
        
        print('Iterating over the transient of {:.0f} steps'.format(n_transient))
        t = time.time()
        X, Y = self.repeatmap(n = self.n_transient, nosave = True)
        print('    Done in {:.2f} s'.format(time.time() - t))

        print('Iterating over the attractor of {:.0f} steps'.format(n_attractor))
        t = time.time()
        self.X_attractor, self.Y_attractor = self.repeatmap(x=X,y=Y,n=int(n_attractor), nosave = False)
        print('    Done in {:.2f} s'.format(time.time() - t))

        print('Generating initial conditions')
        t = time.time()
        self.phi = np.random.uniform(0,2*np.pi,list(self.X_attractor.shape)[1:])
        self.E = np.array([[np.cos(self.phi),-np.sin(self.phi)],[np.sin(self.phi),np.cos(self.phi)]])
        print('    Done in {:.2f} s'.format(time.time() - t))

        print('Calculating the Jacobian Matrix for each point')
        t = time.time()
        self.J = self.jacobian(self.X_attractor,self.Y_attractor,self.l,self.a)
        print('    Done in {:.2f} s'.format(time.time() - t))
        print('System set up for analysis')

    # def maping(self):
    #     """Applies one itteration of the map."""
    #     z = self.x + self.y*1j
    #     z1 = z.copy()
    #     z1 = (1-self.l+self.l*np.abs(z)**self.a)*((z)/(np.abs(z)))**2 + 1
    #     z1[z == 0] = 0
    #     return np.real(z1), np.imag(z1)
    def maping(self):
        return _maping(self.x,self.y,self.l,self.a)
    
    def jacobian(self,x,y,l,a):
        """Computes the Jacobian of the map."""
        J = np.zeros([*x.shape,2,2])

        J = _jacobian(x,y,l,a,J)

        return J
    
    def repeatmap(self, n, x = np.array([-1]), y = np.array([-1]), a = np.array([-1]), l = np.array([-1]), nosave=False):
        if x.all() == -1:
            x = self.x
        if y.all() == -1:
            y = self.y
        if a.all() == -1:
            a = self.a
        if l.all() == -1:
            l = self.l
            
        if not nosave:
            X = np.empty([n,*y.shape])
            Y = np.empty([n,*y.shape])

            X, Y = _repeatmap_save(self.x,self.y,self.l,self.a, n,X,Y)
        else:
            X = self.x.flatten().copy()
            Y = self.y.flatten().copy()
            L = self.l.flatten().copy()
            A = self.a.flatten().copy()
            threadsperblock = 32 
            blockspergrid = (X.size + (threadsperblock - 1)) // threadsperblock
            _repeatmap_nosave[blockspergrid, threadsperblock](X,Y,L,A, n)
            X = X.reshape(*self.x.shape).copy()
            Y = Y.reshape(*self.x.shape).copy()
        return X,Y
    
    
    
    def calcLyapunov(self):
        x = self.x
        y = self.y
        l = self.l
        a = self.a
        n_attractor = self.n_attractor
        n_transient = self.n_transient
        J = self.J
        E = self.E
        
        print('Calculating the Lyapunov Exponents')
        time0 = time.time()
        lyapunov_1 = np.zeros(list(E.shape)[2:])
        lyapunov_2 = np.zeros(list(E.shape)[2:])


        angle_matrix     = E
        first_exponent   = np.zeros([n_attractor, *list(E.shape)[2:]])
        sum_of_exponents = np.zeros([n_attractor, *list(E.shape)[2:]])

        start = time.time()
        # Iterating over given time
        for t in range(n_attractor):

            # Subsetting the jacobian matrix
            jacobian_matrix = J[t].copy()

            j_flat = jacobian_matrix.reshape(np.prod(jacobian_matrix.shape[:-2]),2,2)
            a_flat = angle_matrix.reshape(2,2,np.prod(angle_matrix.shape[2:]))

            # rotating the angle matrix
            threadsperblock = 32 
            blockspergrid = (j_flat.shape[0] + (threadsperblock - 1)) // threadsperblock
            out = a_flat.copy()
            angle_matrix_rotations[blockspergrid, threadsperblock](j_flat, a_flat, out)
            a_flat = out

            angle_matrix = a_flat.reshape(angle_matrix.shape)

            # Calculating the Lyapunov Exponents
            first_exponent[t] = np.linalg.norm(angle_matrix[:,0], axis=0)
            sum_of_exponents[t] = np.linalg.det(np.einsum('ijxyla->xylaij', angle_matrix))

            # Renormalising
            angle_matrix = self.vectorGSchmidt(angle_matrix)

            # Progress Bar
            if t%(n_attractor//10)==0:
                print('{:.0f}% done iterating in {:.2f}s'.format(t/n_attractor*100, time.time()-start))
                start = time.time()

        self.lyapunov_1 = np.mean(np.log(first_exponent), axis=0)
        self.lyapunov_2 = np.mean(np.log(sum_of_exponents), axis=0)   
        print('This took {:.02f} seconds to run'.format(time.time()-time0))
        
    def vectorGSchmidt(self,A):
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
    
    def plot_Lyapunov_1(self, savefig=True, figname=None):
        
        if figname == None:
            figname = 'first_lyapunov'
        
        
        lyapunov_1 = self.lyapunov_1
        x = self.x
        y = self.y
        l = self.l
        a = self.a
        
        
        fig, ax = plt.subplots()
        divnorm = colors.DivergingNorm(vmin=lyapunov_1.min(axis=0).min(axis=0).min(), vcenter=0, vmax=lyapunov_1.max())
        plt.contourf(a[0,0,:,:],l[0,0,:,:],lyapunov_1.min(axis=0).min(axis=0), levels = 100,cmap = 'RdBu_r', norm=divnorm)
        cbar = plt.colorbar()
        for i in range(lyapunov_1.shape[0]):
            for j in range(lyapunov_1.shape[1]):
                plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_1[i,j], levels = [0,], colors=('k',),alpha=0.1)
        contour = plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_1.max(axis=0).max(axis=0), levels = [0,], colors=('blue',),alpha=1)
        plt.title('The first Lyapunov exponent')
        plt.ylabel('$\lambda$')
        plt.xlabel('a')
        cbar.ax.set_ylabel('First Lyapunov Exponent')
        plt.plot(chaos[:,0],chaos[:,1],'r--',lw=3)
        ax.set_ylim([l.min(),l.max()])
        ax.set_xlim([a.min(),a.max()])
        if savefig:
            plt.savefig(f'images/{figname}.pdf')
        plt.show()
        
        return contour.allsegs[0][0]
        
        
    def plot_Lyapunov_2(self, savefig=True, figname=None):
        
        if figname == None:
            figname = 'sum_of_first_2_lyapunov'
        
        lyapunov_2 = self.lyapunov_2
        x = self.x
        y = self.y
        l = self.l
        a = self.a
        
        
        fig, ax = plt.subplots()

        divnorm = colors.DivergingNorm(vmin=np.nanmin(np.nanmax(np.nanmax(lyapunov_2, axis=0), axis=0)), vcenter=0, vmax=np.nanmax(lyapunov_2))
        plt.contourf(a[0,0,:,:],l[0,0,:,:],np.nanmax(np.nanmax(lyapunov_2, axis=0), axis=0), levels = 100, cmap = 'RdBu_r', norm=divnorm)
        cbar = plt.colorbar()

        plt.plot(wild_chaos[:,0],wild_chaos[:,1],'--r',lw=3)

        for i in range(lyapunov_2.shape[0]):
            for j in range(lyapunov_2.shape[1]):
                plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_2[i,j], levels = [0,], colors=('k',),alpha=0.1)
        lyap_sum = plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_2.max(axis=0).max(axis=0), levels = [0,], colors=('blue',),alpha=1)

#         dat0 = lyap_sum.allsegs[0][0]

        plt.title('Sum of the first 2 Lyapunov exponents ')
        plt.ylabel('$\lambda$')
        plt.xlabel('a')
        cbar.ax.set_ylabel('Sum of the first 2 Lyapunov exponents')

        ax.set_ylim([l.min(),l.max()])
        ax.set_xlim([a.min(),a.max()])
        if savefig:
            plt.savefig(f'images/{figname}.pdf')
        plt.show()
        
#         return dat0
        
        
    def new_coords(self, da=0.01, mode=2):
        """Takes the lyapunov exponents and identifies new coordinates for new systems which could contain a 0 value"""
        if mode == 2:
            indicies = []
            a1 = np.empty(self.a.shape)
            # iterating over lambda
            for i in range(self.lyapunov_2.shape[2]):
                j = np.argwhere(np.diff(np.sign(self.lyapunov_2.max(axis=0).max(axis=0)[i,:])))
                
                if j!=[]:
                    indicies += [[i,j[0,0]]]
                    a1[:,:,i,:] = np.linspace(self.a[0,0,0,j[0,0]]-5*da,self.a[0,0,0,j[0,0]]+5*da,self.a.shape[-1])
            
        if mode == 1:
            indicies = []
            a1 = np.empty(self.a.shape)
            # iterating over lambda
            for i in range(self.lyapunov_1.shape[2]):
                j = np.argwhere(np.diff(np.sign(self.lyapunov_1.max(axis=0).max(axis=0)[i,:])))
                if j!=[]:
                    indicies += [[i,j[0,0]]]
                    a1[:,:,i,:] = np.linspace(self.a[0,0,0,j[0,0]]-5*da,self.a[0,0,0,j[0,0]]+5*da,self.a.shape[-1])
            
        return a1
    
    
#     def savedata(self, filename=None):
        
#         if filename == None:
#             filename = 'wild_chaos'
            
#         indicies = []
#         alist = []
#         llist = []
#         # iterating over lambda
#         for i in range(self.lyapunov_2.shape[2]):
#             j = np.argwhere(np.diff(np.sign(self.lyapunov_2.max(axis=0).max(axis=0)[i,:])))
#             print(self.a)
#             if j!=[]:
#                 indicies += [[i,j[0,0]]]
#                 alist += [self.a[0,0,0,j[0,0]]] 
#                 llist += [self.l[0,0,i,0]]
#         output = np.array([alist,llist])
#         np.savetxt(f'data/{filename}.dat', output.T, delimiter='   ')
    
    
    
    
    
    
    
    
    
    
    
    
def plot_multiLyapunov(systems, mode=2, savefig=True, figname=None):
    """Plots multiple lyapunov exponents on the same plot. This gets around the issue of having multiple """
    if mode == 2:
        print(systems)
#         divnorm = colors.DivergingNorm(vmin=max([np.nanmin(np.nanmax(np.nanmax(system.lyapunov_2, axis=0), axis=0)) for system in systems]), vcenter=0, vmax=max[np.nanmax(system.lyapunov_2) for system in systems])
        if figname == None:
            figname = 'sum_of_first_2_lyapunov'
        
        fig, ax = plt.subplots()
        for system in systems:

            lyapunov_2 = system.lyapunov_2
            x = system.x
            y = system.y
            l = system.l
            a = system.a



            plt.contourf(a[0,0,:,:],l[0,0,:,:],np.nanmax(np.nanmax(lyapunov_2, axis=0), axis=0), levels = 100, cmap = 'RdBu_r')
#                          , norm=divnorm)
            for i in range(lyapunov_2.shape[0]):
                for j in range(lyapunov_2.shape[1]):
                    plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_2[i,j], levels = [0,], colors=('k',),alpha=0.1)
            lyap_sum = plt.contour(a[0,0,:,:],l[0,0,:,:],lyapunov_2.max(axis=0).max(axis=0), levels = [0,], colors=('blue',),alpha=1)

#         cbar = plt.colorbar()
        plt.plot(wild_chaos[:,0],wild_chaos[:,1],'--r',lw=3)
        plt.title('Sum of the first 2 Lyapunov exponents ')
        plt.ylabel('$\lambda$')
        plt.xlabel('a')
#         cbar.ax.set_ylabel('Sum of the first 2 Lyapunov exponents')

        ax.set_ylim([l.min(),l.max()])
        ax.set_xlim([a.min(),a.max()])
        if savefig:
            plt.savefig(f'images/{figname}.pdf')
        plt.show()