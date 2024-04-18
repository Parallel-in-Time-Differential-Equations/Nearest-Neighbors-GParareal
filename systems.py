import scipy
from matplotlib import use
import numpy as np



try:
    import jax 
    import jax.numpy as jnp
    from jax import config
    config.update("jax_enable_x64", True)
    is_jax_installed = True
except ImportError as e:
    print('Jax not installed. Falling back to numpy')
    print(e)
    is_jax_installed = False


from utils import Normalize

      
    
class ODE():

    def __init__(self, name, mn, mx, u0, normalization=None, use_jax=True):
        self.name = name
        self.normalizer = Normalize(mn, mx, normalization)
        self.u0 = self.normalizer.fit(u0)
        self.use_jax = use_jax


    def get_vector_field(self):
        use_jax = self.use_jax
        f_orig = self._get_f(use_jax)

        def f_normalized(t, u):
            u = self.normalizer.inverse(u)
            out = f_orig(t, u)
            out = out * self.normalizer.get_scale()
            return out
        
        if use_jax and is_jax_installed:
            f_normalized = jax.jit(f_normalized)
        return f_normalized
    
    
    def _get_f(self, use_jax):
        if use_jax and is_jax_installed:
            return jax.jit(self._f_jax)
        else:
            return self._f_np
        
        
    @staticmethod
    def _f_jax(t, u):
        raise NotImplementedError('This is an abstract class')
    
    
    @staticmethod
    def _f_np(t, u):
        raise NotImplementedError('This is an abstract class')
    

    def set_default_init_cond(self, u0):
        self.u0 = self.normalizer.fit(u0)

    
    def get_init_cond(self, *args, u0=None, **kwargs):
        if u0 is None:
            u0 = self.u0
        else:
            u0 = self.normalizer.fit(u0)
        return np.array(u0, dtype=float)
    

    def get_dim(self):
        return self.u0.shape[0]


class FHN_ODE(ODE):
    def __init__(self, **kwargs):
        mn, mx = np.array([[-2, -1], [2.1, 1.2]])
        u0 = np.array([-1,1])
        super().__init__('FHN_ODE', mn, mx, u0, **kwargs)

    
    @staticmethod
    def _f_jax(t, u):
        a = 0.2 
        b = 0.2
        c = 3
        out = jnp.zeros(u.shape)
        out = out.at[0].set(c*(u[0] - ((u[0]**3)/3) + u[1]))
        out = out.at[1].set(-(1/c)*(u[0] - a + b*u[1]))
        return out
    

    @staticmethod
    def _f_np(t, u):
        a = 0.2 
        b = 0.2
        c = 3
        out = np.zeros(u.shape)
        out[0] = c*(u[0] - ((u[0]**3)/3) + u[1])
        out[1] = -(1/c)*(u[0] - a + b*u[1])
        return out
    

class Rossler(ODE):
    def __init__(self, **kwargs):
        mn, mx = np.array([[-10, -11, 0], [12, 8, 23]])
        u0 = np.array([0,-6.78,0.02]) 
        super().__init__('Rossler', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        a = 0.2
        b = 0.2
        c = 5.7
        out = jnp.zeros(u.shape)
        out = out.at[0].set(-u[1]-u[2])
        out = out.at[1].set(u[0]+(a*u[1]))
        out = out.at[2].set(b+u[2]*(u[0]-c))
        return out
    

    @staticmethod
    def _f_np(t, u):
        a = 0.2
        b = 0.2
        c = 5.7
        out = np.zeros(u.shape)
        out[0] = -u[1]-u[2]
        out[1] = u[0]+(a*u[1])
        out[2] = b+u[2]*(u[0]-c)
        return out
    

class Hopf(ODE):
    def __init__(self, tspan=[-20, 500], **kwargs):
        mn, mx = jnp.array([[-23, -23, 0], [23, 23, 1]])
        u0 = np.array([0.1, 0.1, tspan[0]])
        self.maxtime = tspan[1]
        super().__init__('Hopf', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u, maxtime):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(-u[1]+u[0]*((u[2]/maxtime)-u[0]**2-u[1]**2))
        out = out.at[1].set(u[0]+u[1]*((u[2]/maxtime)-u[0]**2-u[1]**2))
        out = out.at[2].set(1)
        return out
    

    @staticmethod
    def _f_np(t, u, maxtime):
        out = np.zeros(u.shape)
        out[0] = -u[1]+u[0]*((u[2]/maxtime)-u[0]**2-u[1]**2)
        out[1] = u[0]+u[1]*((u[2]/maxtime)-u[0]**2-u[1]**2)
        out[2] = 1
        return out
    

    def _get_f(self, use_jax):
        if use_jax and is_jax_installed:
            f_jit = jax.jit(self._f_jax)
            f = lambda t, u: f_jit(t, u, self.maxtime)
        else:
            f = lambda t, u: self._f_np(t, u, self.maxtime)
        return f
    

class DblPend(ODE):
    def __init__(self, **kwargs):
        mn, mx = jnp.array([[-2, -2.5, -17, -3.5], [2, 2.5, 1, 3.5]])
        u0 = np.array([-0.5,0,0,0]) 
        super().__init__('DblPend', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(u[1])
        out = out.at[1].set((-1/(2-jnp.cos(u[0]-u[2])**2))*((u[1]**2)*jnp.cos(u[0]-u[2])*jnp.sin(u[0]-u[2])+(u[3]**2)*jnp.sin(u[0]-u[2])+2*jnp.sin(u[0])-jnp.cos(u[0]-u[2])*jnp.sin(u[2])))
        out = out.at[2].set(u[3])
        out = out.at[3].set((-1/(2-jnp.cos(u[0]-u[2])**2))*(-2*(u[1]**2)*jnp.sin(u[0]-u[2])-(u[3]**2)*jnp.sin(u[0]-u[2])*jnp.cos(u[0]-u[2])-2*jnp.cos(u[0]-u[2])*jnp.sin(u[0])+2*jnp.sin(u[2])))
        return out
    

    @staticmethod
    def _f_np(t, u):
        out = np.zeros(u.shape)
        out[0] = u[1]
        out[1] = (-1/(2-np.cos(u[0]-u[2])**2))*((u[1]**2)*np.cos(u[0]-u[2])*np.sin(u[0]-u[2])+(u[3]**2)*np.sin(u[0]-u[2])+2*np.sin(u[0])-np.cos(u[0]-u[2])*np.sin(u[2]))
        out[2] = u[3]
        out[3] = (-1/(2-np.cos(u[0]-u[2])**2))*(-2*(u[1]**2)*np.sin(u[0]-u[2])-(u[3]**2)*np.sin(u[0]-u[2])*np.cos(u[0]-u[2])-2*np.cos(u[0]-u[2])*np.sin(u[0])+2*np.sin(u[2]))
        return out
    

class Brusselator(ODE):
    def __init__(self, **kwargs):
        mn, mx = jnp.array([[0.4, 0.9], [4, 5]])
        u0 = np.array([1, 3.07])   
        super().__init__('Brusselator', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(1+(u[0]**2)*u[1]-(3+1)*u[0])
        out = out.at[1].set(3*u[0]-(u[0]**2)*u[1])
        return out
    

    @staticmethod
    def _f_np(t, u):
        out = np.zeros(u.shape)
        out[0] = 1+(u[0]**2)*u[1]-(3+1)*u[0]
        out[1] = 3*u[0]-(u[0]**2)*u[1]
        return out
    

class Lorenz(ODE):
    def __init__(self, **kwargs):
        mn, mx = jnp.array([[-17.1, -23, 6], [18.1, 25, 45]])
        u0 = np.array([-15,-15,20])
        super().__init__('Lorenz', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(10*(u[1]-u[0]))
        out = out.at[1].set(28*u[0]-u[1]-u[0]*u[2])
        out = out.at[2].set(u[0]*u[1]-(8/3)*u[2])
        return out
    

    @staticmethod
    def _f_np(t, u):
        out = np.zeros(u.shape)
        out[0] = 10*(u[1]-u[0])
        out[1] = 28*u[0]-u[1]-u[0]*u[2]
        out[2] = u[0]*u[1]-(8/3)*u[2]
        return out
    

class ThomasLabyrinth(ODE):
    def __init__(self, **kwargs):
        mn, mx = jnp.array([[-12, -12, -12], [12, 12, 12]])
        u0 = np.array([4.6722764,5.2437205e-10,-6.4444208e-10])
        super().__init__('ThomasLabyrinth', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        a = 0.5
        b = 10.0
        out = jnp.zeros(u.shape)
        x = u[0]
        y = u[1]
        z = u[2]
        xdot = -a * x + b * jnp.sin(y)
        ydot = -a * y + b * jnp.sin(z)
        zdot = -a * z + b * jnp.sin(x)
        out = out.at[0].set(xdot)
        out = out.at[1].set(ydot)
        out = out.at[2].set(zdot)
        return out
    

    @staticmethod
    def _f_np(t, u):
        a = 0.5
        b = 10.0
        out = np.zeros(u.shape)
        x = u[0]
        y = u[1]
        z = u[2]
        xdot = -a * x + b * np.sin(y)
        ydot = -a * y + b * np.sin(z)
        zdot = -a * z + b * np.sin(x)
        out[0] = xdot
        out[1] = ydot
        out[2] = zdot
        return out

    
class FHN_PDE(ODE):
    def __init__(self, d_x, seed=45, **kwargs):
        self.d_x = d_x
        d_y = d_x
        self.d_y = d_y

        d = 2 * (d_x * d_y)  
        self.d = d
        self.DXX, self.DYY = self._calc_matrices(d_x, d_y)
        
        mn, mx = np.array([[-1] * d, [1] * d])
        # this is for backward compatibility
        np.random.seed(seed)
        if hasattr(np.random, 'get_bit_generator'):
            bitgen = np.random.get_bit_generator()
            rng = np.random.Generator(bitgen)
        else:
            print('WARNING: different initial cond for FHN_PDE')
            rng = np.random.default_rng(seed)
        
        # rng = np.random.default_rng(seed)
        u0 = rng.uniform(size=self.d)

        # previous code used rand. That's an alias for np.random.random_sample
        # whose equivalent is rng.random. Seems like uniform with the setup above
        # does the same thing, so keep it as more readable

        super().__init__(f'FHN_PDE_{d_x}', mn, mx, u0, **kwargs)
        

    @staticmethod
    def _calc_matrices(d_x, d_y):                            
        xspan = [-1,1]                                                  

        dx = (xspan[1]-xspan[0])/(d_x-1)
        dy = (xspan[1]-xspan[0])/(d_y-1)

        z1 = np.ones(d_x)
        Txx = np.diag(-2*z1)
        idxs = np.arange(d_x-1)
        Txx[idxs, idxs+1] = z1[:d_x-1]
        Txx[idxs+1, idxs] = z1[:d_x-1]
        Dxx = (1/(dx**2))*Txx

        z1 = np.ones(d_y)
        Tyy = np.diag(-2*z1)
        idxs = np.arange(d_y-1)
        Tyy[idxs, idxs+1] = z1[:d_y-1]
        Tyy[idxs+1, idxs] = z1[:d_y-1]
        Dyy = (1/(dy**2))*Tyy


        # boundary conditions (periodic)
        Dxx[0,-1] = 1/(dx**2)
        Dxx[-1,0] = 1/(dx**2)
        Dyy[0, -1] = 1/(dy**2)
        Dyy[-1,0] = 1/(dy**2)

        # construct differentiation matrices (using kronecker products)
        DXX = np.kron(np.eye(d_y,d_y),Dxx)
        DYY = np.kron(Dyy,np.eye(d_x,d_x))
        
        return DXX, DYY
    
    @staticmethod
    def _f_jax(t,u,DXX,DYY):
        d = int(u.shape[0]/2)
        u1 = u[:d]
        u2 = u[d:]
        
        a = 2.8E-4
        b = 5E-3
        k = -5E-3
        tau = 0.1
        U = a*(DXX + DYY)@u1 + u1 - (u1**3) - u2 + k*jnp.ones(d)
        V = (1/tau)*( b*(DXX + DYY)@u2 + u1 - u2 )

        return jnp.hstack([U, V])
    
    @staticmethod
    def _f_np(t,u,DXX,DYY):
        d = int(u.shape[0]/2)
        u1 = u[:d]
        u2 = u[d:]
        
        a = 2.8E-4
        b = 5E-3
        k = -5E-3
        tau = 0.1
        U = a*(DXX + DYY)@u1 + u1 - (u1**3) - u2 + k*np.ones(d)
        V = (1/tau)*( b*(DXX + DYY)@u2 + u1 - u2 )

        return np.hstack([U, V])
    
    
    def _get_f(self, use_jax):
        if use_jax and is_jax_installed:
            f_jit = jax.jit(self._f_jax)
            f = lambda t, u: f_jit(t, u, self.DXX, self.DYY)
        else:
            f = lambda t, u: self._f_np(t, u, self.DXX, self.DYY)
        return f
    
    def _get_jax_func(self):
        if self.use_jax and is_jax_installed:
            return jax.jit(self._f_jax)
        else:
            raise NotImplementedError('This method is only available when using jax')
    


class Burgers(ODE):
    def __init__(self, d_x, nu=1/100, **kwargs):
        self.d_x = d_x
        self.nu = nu

        d = d_x 
        self.d = d
        self.Dxx, self.Dx = self._calc_matrices(d_x, nu)
        
        mn, mx = np.array([[0] * d, [1] * d])

        # init condition
        xspan = [-1,1]   
        x_fine = np.linspace(xspan[0], xspan[-1], num=(d-1)+1)
        u0 = 0.5*(np.cos(4.5*np.pi*x_fine) + 1)

        super().__init__(f'Burgers_{d_x}', mn, mx, u0, **kwargs)
        

    @staticmethod
    def _calc_matrices(d, nu):                            
        xspan = [-1,1]                                                      

        dx = (xspan[1]-xspan[0])/(d-1)
        z0 = np.zeros(d)
        z1 = np.ones(d)
        Txx = np.diag(-2*z1)
        idxs = np.arange(d-1)
        Txx[idxs, idxs+1] = z1[:d-1]
        Txx[idxs+1, idxs] = z1[:d-1]
        Dxx = (nu/(dx**2))*Txx
        Tx = np.diag(z0)
        Tx[idxs, idxs+1] = z1[:d-1]
        Tx[idxs+1, idxs] = -z1[:d-1]
        Dx = (1/(2*dx))*Tx
        Dxx[0,-1] = 1*(nu/(dx**2))
        Dxx[-1,0] = 1*(nu/(dx**2))
        Dx[0, -1] = -1*(1/(2*dx))
        Dx[-1,0] = 1*(1/(2*dx))
        
        return Dxx, Dx
    
    @staticmethod
    def _f_jax(t,u,Dxx,Dx):
        return  Dxx@u - u*(Dx@u)
    
    @staticmethod
    def _f_np(t,u,Dxx,Dx):
        return  Dxx@u - u*(Dx@u)
    
    
    def _get_f(self, use_jax):
        if use_jax and is_jax_installed:
            f_jit = jax.jit(self._f_jax)
            f = lambda t, u: f_jit(t, u, self.Dxx, self.Dx)
        else:
            f = lambda t, u: self._f_np(t, u, self.Dxx, self.Dx)
        return f
    


class DiffReact(ODE):
    '''
    Code adapted from PDEBench, https://github.com/pdebench/PDEBench
    '''
    def __init__(self, d_x: int, Du: float = 1e-3, Dv: float = 5E-3,
                 k: float = 5E-3, seed=45, **kwargs):
        self.d_x = d_x
        d_y = d_x
        self.d_y = d_y
        self.Du = Du
        self.Dv = Dv
        self.k = k

        d = 2 * (d_x * d_y)  
        self.d = d
        self.lap = self._calc_laplacian(d_x, d_y)
        
        mn, mx = np.array([[-4] * d, [4] * d])

        # init condition
        rng = np.random.default_rng(seed)
        u0 = rng.uniform(size=self.d)

        super().__init__(f'DiffReact2D_{d_x}', mn, mx, u0, **kwargs)
        

    @staticmethod
    def _calc_laplacian(Nx, Ny):   
        '''
        Nx: Number of spatial points in the x direction
        Ny: Number of spatial points in the y direction
        '''

        xspan = [-1,1]
        yspan = [-1,1]
        d_x = (xspan[1]-xspan[0])/(Nx) 
        d_y = (yspan[1]-yspan[0])/(Ny)

        main_diag = -2*np.ones(Nx)/d_x**2 -2*np.ones(Nx)/d_y**2
        main_diag[0] = -1/d_x**2 -2/d_y**2
        main_diag[-1] = -1/d_x**2 -2/d_y**2
        main_diag = np.tile(main_diag, Ny)
        main_diag[:Nx] = -2/d_x**2 -1/d_y**2
        main_diag[Nx*(Ny-1):] = -2/d_x**2 -1/d_y**2
        main_diag[0] = -1/d_x**2 -1/d_y**2
        main_diag[Nx-1] = -1/d_x**2 -1/d_y**2
        main_diag[Nx*(Ny-1)] = -1/d_x**2 -1/d_y**2
        main_diag[-1] = -1/d_x**2 -1/d_y**2
        
        left_diag = np.ones(Nx)
        left_diag[0] = 0
        left_diag = np.tile(left_diag, Ny)
        left_diag = left_diag[1:]/d_x**2
        
        right_diag = np.ones(Nx)
        right_diag[-1] = 0
        right_diag = np.tile(right_diag, Ny)
        right_diag = right_diag[:-1]/d_x**2
        
        bottom_diag = np.ones(Nx*(Ny-1))/d_y**2
        
        top_diag = np.ones(Nx*(Ny-1))/d_y**2
        
        # Generate the Laplacian matrix
        # lap = np.diag(main_diag) + np.diag(left_diag, -1) + np.diag(right_diag, 1) + np.diag(bottom_diag, -Nx) + np.diag(top_diag, Nx)
        diagonals = [main_diag, left_diag, right_diag, bottom_diag, top_diag]
        offsets = [0, -1, 1, -Nx, Nx]
        lap = scipy.sparse.diags(diagonals, offsets)

        return lap
    
    @staticmethod
    def _f_jax(t,y,lap, Du, Dv, k):
        u = y[:lap.shape[0]]
        v = y[lap.shape[0]:]
       
        # Calculate reaction function for each unknown
        react_u = u - u**3 - k - v
        react_v = u - v
       
        # Calculate time derivative for each unknown
        u_t = react_u + Du * (lap @ u)
        v_t = react_v + Dv * (lap @ v)
        
        # Stack the time derivative into a single array y_t
        y_t = jnp.concatenate((u_t,v_t))
       
        return y_t
    
    @staticmethod
    def _f_np(t,y,lap, Du, Dv, k):
        u = y[:lap.shape[0]]
        v = y[lap.shape[0]:]
       
        # Calculate reaction function for each unknown
        react_u = u - u**3 - k - v
        react_v = u - v
       
        # Calculate time derivative for each unknown
        u_t = react_u + Du * (lap @ u)
        v_t = react_v + Dv * (lap @ v)
        
        # Stack the time derivative into a single array y_t
        y_t = np.concatenate((u_t,v_t))
       
        return y_t
    
    
    def _get_f(self, use_jax):
        if use_jax and is_jax_installed:
            f_jit = jax.jit(self._f_jax)
            f = lambda t, u: f_jit(t, u, self.lap.toarray(), self.Du, self.Dv, self.k)
        else:
            f = lambda t, u: self._f_np(t, u, self.lap, self.Du, self.Dv, self.k)
        return f
    
