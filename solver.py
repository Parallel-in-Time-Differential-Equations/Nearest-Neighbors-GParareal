from kiwisolver import Solver
import numpy as np
import time
from scipy.integrate import solve_ivp

try:
    import jax 
    import jax.numpy as jnp
    from jax import config
    config.update("jax_enable_x64", True)
    is_jax_installed = True
except ImportError as e:
    print('Jax not installed. Falling back to numpy')
    print(e)
    jax = None
    jnp = np
    is_jax_installed = False
    
from RK import RK

def calc_time(f):
    def wrapper(*args, **kwargs):
        s_time = time.time()
        ret = f(*args, **kwargs)
        el_time = time.time() - s_time
        return ret, el_time
    return wrapper

class SolverAbstr:

    '''
    Abstract class. All methods below are expected to return the ODE solution at time t1, given
    an initial condition u0 at time t0. F is expected to be slower but more accurate than G.
    '''

    def run_F(self, t0, t1, u0):
        raise NotImplementedError('run_F not implemented')
    

    @calc_time
    def run_F_timed(self, t0, t1, u0):
        return self.run_F(t0, t1, u0)
    

    def run_F_full(self, t0, t1, u0):
        raise NotImplementedError('run_F_full not implemented')
    

    @calc_time
    def run_F_full_timed(self, t0, t1, u0):
        return self.run_F_full(t0, t1, u0)
    

    def run_G(self, t0, t1, u0):
        raise NotImplementedError('run_G not implemented')
    

    @calc_time
    def run_G_timed(self, t0, t1, u0):
        return self.run_G(t0, t1, u0)
    

    def run_G_full(self, t0, t1, u0):
        raise NotImplementedError('run_G_full not implemented')
    

    @calc_time
    def run_G_full_timed(self, t0, t1, u0):
        return self.run_G_full(t0, t1, u0)
    

class SolverRK(SolverAbstr):

    def __init__(self, f, Ng, Nf, F, G, thresh=1e7, use_jax=True, **kwargs):
        self.f = f
        self.Ng = int(Ng)
        self.Nf = int(Nf)
        self.F = F
        self.G = G
        self.thresh = thresh

        self.RK_F = RK(f, F, use_jax)
        self.RK_G = RK(f, G, use_jax)


    def _run_RK_paged(self, t0, t1, u0, steps, solver):
        f = self.f
        thresh = self.thresh
        if steps > thresh:
            steps = steps - 1
            iters = [thresh]*int(steps/thresh) + [steps%thresh]* (steps%thresh != 0)
            step = (t1 - t0)/(steps)
            for temp_steps in iters:
                t1 = t0 + step*temp_steps
                u0 = solver.run_get_last(t0, t1, steps, u0)
                t0 = t1
        else:
            u0 = solver.run_get_last(t0, t1, steps, u0)
        return u0


    def run_F(self, t0, t1, u0):
        return self._run_RK_paged(t0, t1, u0, self.Nf, self.RK_F)
    

    def run_G(self, t0, t1, u0):
        return self._run_RK_paged(t0, t1, u0, self.Ng, self.RK_G)
    
    def run_F_full(self, t0, t1, u0):
        return self.RK_F.run(t0, t1, self.Nf, u0)
    
    def run_G_full(self, t0, t1, u0):
        return self.RK_G.run(t0, t1, self.Ng, u0)
        

class SolverScipy(SolverAbstr):
    def __init__(self, f, Ng, Nf, G, F= 'RK45', use_jax=True, **kwargs):
        '''
        Note: Nf is interpreted as a soft constaint only. The algorithm may do more steps.
        For the coarse solver, it will use my own RK implementation.'''
        self.f = f
        self.Ng = Ng
        self.Nf = Nf
        self.F = self._map_solver(F)
        self.G = G
        self.kwargs = kwargs
        self.rk_solver = SolverRK(f, Ng, Nf, F, G, use_jax=use_jax)


    def _map_solver(self, solver):
        map_dict = {'RK2': 'RK23', 'RK4': 'RK45', 'RK8': 'DOP853'}
        if solver in map_dict:
            return map_dict[solver]
        else:
            return solver


    def run_F(self, t0, t1, u0):
        f = self.f
        res = solve_ivp(f, [t0, t1], u0, method=self.F, 
                         t_eval=(t1, ), max_step=(t1-t0)/self.Nf, 
                         **self.kwargs)
        if res.nfev > self.Nf*1.5:
            print(f'Warning: F solver did {res.nfev/self.Nf:0.1f}x more steps than expected')
        return res.y.reshape(-1)
    
    def run_G(self, t0, t1, u0):
        return self.rk_solver.run_G(t0, t1, u0)
    
