import numpy as np
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    
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




class RKAbstr():
    def __init__(self, f, method, use_jax):
        if use_jax and is_jax_installed:
            self.use_jax = True
        else:
            self.use_jax = False

        if method == 'RK1':
            a = np.array([[0]])
            b = np.array([[1]]) 
            c = np.array([0])
        elif method == 'RK2':
            a = np.array([[0,0],[0.5,0]])
            b = np.array([[0,1]])
            c = np.array([0,0.5])
        elif method == 'RK4':  #classic fourth-order method
            a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
            b = np.array([[1/6,1/3,1/3,1/6]])
            c = np.array([0,0.5,0.5,1])
        elif method == 'RK8': #Cooper-Verner eigth-order method (again there are many)
            s = np.sqrt(21)
            a = np.array([[0,0,0,0,0,0,0,0,0,0,0],[1/2,0,0,0,0,0,0,0,0,0,0],[1/4,1/4,0,0,0,0,0,0,0,0,0],[1/7,(-7-3*s)/98,(21+5*s)/49,0,0,0,0,0,0,0,0],[(11+s)/84,0,(18+4*s)/63,(21-s)/252,0,0,0,0,0,0,0],[(5+s)/48,0,(9+s)/36,(-231+14*s)/360,(63-7*s)/80,0,0,0,0,0,0],[(10-s)/42,0,(-432+92*s)/315,(633-145*s)/90,(-504+115*s)/70,(63-13*s)/35,0,0,0,0,0],[1/14,0,0,0,(14-3*s)/126,(13-3*s)/63,1/9,0,0,0,0],[1/32,0,0,0,(91-21*s)/576,11/72,(-385-75*s)/1152,(63+13*s)/128,0,0,0],[1/14,0,0,0,1/9,(-733-147*s)/2205,(515+111*s)/504,(-51-11*s)/56,(132+28*s)/245,0,0],[0,0,0,0,(-42+7*s)/18,(-18+28*s)/45,(-273-53*s)/72,(301+53*s)/72,(28-28*s)/45,(49-7*s)/18,0]])
            b = np.array([[1/20,0,0,0,0,0,0,49/180,16/45,49/180,1/20]])
            c = np.array([0,1/2,1/2,(7+s)/14,(7+s)/14,1/2,(7-s)/14,(7-s)/14,1/2,(7+s)/14,1])
        else:
            raise NotImplementedError('Only RK1, RK2, RK4 and RK8 are implemented')
        
        self.a = a
        self.b = b
        self.c = c
        self.f = f

    def run(self, *args, **kwargs):
        raise NotImplementedError('run not implemented')
    

    def run_timed(self, *args, **kwargs):
        s_time = time.time()
        ret = self.run(*args, **kwargs)
        el_time = time.time() - s_time
        return ret, el_time
    

    def run_get_last(self, *args, **kwargs):
        raise NotImplementedError('run_get_last not implemented')
    
    def run_get_last_timed(self, *args, **kwargs):
        s_time = time.time()
        ret = self.run_get_last(*args, **kwargs)
        el_time = time.time() - s_time
        return ret, el_time

    



class RK(RKAbstr):
    '''
    Class to obtain the solution of an ODE using a Runge-Kutta method. 
    run() returns the whole trajectory, at a relatively high memory footprint.
    run_get_last() returns only the last point of the trajectory - optimized.
    '''

    def __init__(self, f, method, use_jax=True):
        super().__init__(f, method, use_jax)
        
    

    def run(self, t0, t1, steps, u0):
        steps = int(steps)
        t = np.linspace(t0, t1, num=steps+1)
        if self.use_jax:
            res = _RK_jax_all(jnp.array(t), u0, self.f, 
                                jnp.array(self.a), jnp.array(self.b), jnp.array(self.c))
        else:
            res = self._RK_numpy_(t, u0, self.f, self.a, self.b, self.c)
        return np.array(res)

    def run_get_last(self, t0, t1, steps, u0):
        steps = int(steps)
        dt = (t1 - t0)/(steps)
        if self.use_jax:
            res = _RK_jax_last(t0, dt, steps, jnp.array(u0), self.f, 
                                jnp.array(self.a), jnp.array(self.b), jnp.array(self.c))
        else:
            return self.run(t0, t1, steps, u0)[-1, :]
        return np.array(res)
        


    @staticmethod
    def _RK_numpy_(t, u0, f, a, b, c):        
        u = np.zeros((len(u0), len(t)))
        u[:,0] = u0
        
        for n in range(len(t)-1):
            
            # iterate over runge kutta 
            h = t[n+1] - t[n]
            dim = len(u0)
            S = b.shape[-1]
            k = np.zeros((dim,S))
            k[:,0] = h*f(t[n], u[:,n])
            
            # calculate the coefficients k
            for i in range(1,S):
                temp = np.zeros(dim)
                for j in range(0, i):
                    temp = temp + a[i,j] * k[:,j]
                k[:,i] = h*f(t[n]+c[i]*h, u[:,n]+temp)
                
            # calculate the final solution
            u[:,n+1] = u[:,n] + np.sum(b*k, 1)
            
        return u.T



'''
Note: JAX JITed functions need to be jitteed outside classes to be pickled by MPI4py library
for parallel computing. Otherwise some _cpp_pjit.<locals>.cache_miss object will not be picklable.
'''

def _RK_jax_last(t0, dt, steps, u0, f, a, b, c):
    u = u0
    t = t0

    dim = u0.shape[0]
    S = b.shape[-1]
    
    def inner_inn_loop(j, carry):
        temp, i, k = carry
        return [temp + a[i,j] * k[:,j], i, k]
    
    def inner_loop(i, carry):
        k, u, t = carry
        h = dt
        temp = jnp.zeros(dim)
        temp, _, _ = jax.lax.fori_loop(0, i, inner_inn_loop, [temp, i, k])
        return [k.at[:,i].set(h*f(t+c[i]*h, u+temp)), u, t]
    
    def outer_loop(n, carry):
        u, t = carry
        h = dt
        k = jnp.zeros((dim,S))
        k = k.at[:,0].set(h*f(t, u))
        k, _, _ = jax.lax.fori_loop(1, S, inner_loop, [k, u, t])
        return [u + jnp.sum(b*k, 1), t+dt]
        
    u, t = jax.lax.fori_loop(0, steps, outer_loop, [u, t])
    return u.T    
_RK_jax_last = jax.jit(_RK_jax_last, static_argnums=(4,))



def _RK_jax_all(t, u0, f, a, b, c):
        u = jnp.zeros((u0.shape[0], t.shape[0]))
        u = u.at[:,0].set(u0)
        dim = u0.shape[0]
        S = b.shape[-1]
        
        def inner_inn_loop(j, carry):
            temp, i, k = carry
            return [temp + a[i,j] * k[:,j], i, k]
        
        def inner_loop(i, carry):
            n, k, u, h = carry
            temp = jnp.zeros(dim)
            temp, _, _ = jax.lax.fori_loop(0, i, inner_inn_loop, [temp, i, k])
            return [n, k.at[:,i].set(h*f(t[n]+c[i]*h, u[:,n]+temp)), u, h]
        
        def outer_loop(n, u):
            h = t[n+1] - t[n]
            k = jnp.zeros((dim,S))
            k = k.at[:,0].set(h*f(t[n], u[:,n]))
            _, k, _, _ = jax.lax.fori_loop(1, S, inner_loop, [n, k, u, h])
            return u.at[:, n+1].set(u[:,n] + jnp.sum(b*k, 1))
            
        u = jax.lax.fori_loop(0, t.shape[0]-1, outer_loop, u)
        return u.T  
_RK_jax_all = jax.jit(_RK_jax_all, static_argnums=(2,)) 