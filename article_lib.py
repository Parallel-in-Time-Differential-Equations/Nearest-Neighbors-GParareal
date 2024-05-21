import os
# os.chdir('/mnt/c/Users/u2133517/OneDrive - University of Warwick/lyudmila_project/massi/python')
import numpy as np

import jax 
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import scipy
from joblib import Parallel, delayed

#%% Changelog

# 2023_07_28:
#     - add support for normalized system in Systems class (note: this is a global normalization, as is specific to the
#       system and not the technique used. So all models ran on the system [parareap, GP, etc] will be normalized without
#       the option to un-normalize). You should be able to combine different runs objects for some of the visualizations though.
#     - Remove previous normalization implementation (model based, with global option)
#     - add support to alter system setting (e.g. starting condition, epsylon) in Systems
#     - add support for (partially) custom title in parareal plot() (for instance, adding info on ODE or hyper params)
        
# 2023_08_28:
#     - Add NNGP class
#     - Add data_x, data_D as input to fit
#     - Add i as input to predict (may require adding kwargs to model specific predict)
#     - Add args kwargs to GP and NNGP predict

# 2023_10_04:
#     - Add support for pool parameter in Parareal run
#     - Introduce parallel NNGP. Note the kernel is hardcoded for now.
#     - Re-use pool workers for RK computation as well in Parareal class
#     - In Parareal, recast all calls to RK that only keep last values to RK_last
#     - Introduce RK buffer, that only computes the last through subsequent applications (big numbers)


# New: 
#     - Fixed all bugs, incorporated kernel in model.
#     - Removed duplicated models NNGP and GPjax and GParareal. Rename *_p to *
#     - Made original *_p serial if no pool is given. A number of processor can also
#       be specified, no need of remembering the import.
#     - Add intermediate run to create and shutdown the pool
#     - Drastically change Systems so the normalized function can be pickled

### TODO:
    # There is a lot of overlapping code between GParareal, GPjax, NNGP, NNGP_p, GPjax_p. You might want to 
    # - create a base class that contains the kernel, and all the loss/predict functions
    # - find a basic version of the init and put it in the base class
    # - for non-parallel code, allow for the user to give its own kernel (else use a default from the base)
    # - for the parallels, leave the kernel hard coded.
    # - Ideally you'd also merge parallel and non-parallel versions
    

#%% Computing speedups

def calc_exp_gp_cost(run_obj, n_cores, n_jitter=9, *args, **kwargs):
    Tm = run_obj['timings']['avg_serial_train_time']
    d = run_obj['d']
    exp_train_time = np.sum(Tm * max(n_jitter * d / n_cores, 1))
    return run_obj['timings']['mdl_pred_t'] + exp_train_time

def get_act_mdl_cost(run_obj):
    return run_obj['timings']['mdl_tot_t']

def get_act_cost(run_obj):
    return run_obj['timings']['runtime']


def calc_exp_nngp_cost_rough(run_obj, n_cores, N, n_jitter=9, n_restarts=None, *args, **kwargs):
    k = run_obj['k']
    Tm = run_obj['timings']['avg_serial_train_time']
    if n_restarts is None:
        n_restarts = run_obj['mdl'].n_restarts
    d = run_obj['d']
    
    exp_c_rough = k * (Tm*max(((n_jitter*n_restarts*d)/n_cores),1)) * (N-(k+1)/2)
    return exp_c_rough
    
# This uses the average serial training time
def calc_exp_nngp_cost_precise_v1(run_obj, n_cores, N, n_jitter=9, n_restarts=None, *args, **kwargs):
    k = run_obj['k']
    Tm = run_obj['timings']['avg_serial_train_time']
    if n_restarts is None:
        n_restarts = run_obj['mdl'].n_restarts
    d = run_obj['d']
    
    conv_int = np.array([0]+run_obj['conv_int'][:-1])
    exp_cost_precise = ((N-conv_int)*(Tm*max(((n_jitter*n_restarts*d)/n_cores),1))).sum()
    return exp_cost_precise
    
    
def calc_exp_para_mdl_cost(run_obj, *args, **kwargs):
    return 0
    
    
def est_serial(run_obj, N):
    return run_obj['timings']['F_time_serial_avg']*N
    
def calc_speedup(run_obj, serial=None, N=None):
    if N is None:
        raise Exception('Cannot compute speedup without either N or serial.')
    else:
        serial = est_serial(run_obj, N)
    return serial/get_act_cost(run_obj)

def calc_exp_speedup(run_obj, mdl_cost_fn, *args, **kwargs):
    if 'N' not in kwargs:
        raise Exception('Cannot compute speedup without either N or serial.')
    else:
        serial = est_serial(run_obj, kwargs['N'])
    Tf = run_obj['timings']['F_time_serial_avg']*run_obj['k']
    Tg = run_obj['timings']['G_time']
    return serial/(Tf + Tg + mdl_cost_fn(run_obj, *args, **kwargs))
    

#%% RK stuff


def RK_last(ins, thresh):
    t_s, t_end, t_steps, x_init, f, F = ins
    if t_steps > thresh:
        t_steps = t_steps - 1
        iters = [thresh]*int(t_steps/thresh) + [t_steps%thresh]* (t_steps%thresh != 0)
        step = (t_end - t_s)/(t_steps)
        for temp_t_steps in iters:
            t_end = t_s + step*temp_t_steps
            x_init = RK(np.linspace(t_s, t_end, num=t_steps), x_init, f, F)[-1, :]
            t_s = t_end
    else:
        x_init = RK(np.linspace(t_s, t_end, num=t_steps), x_init, f, F)[-1, :]
    return x_init

# def RK_last(ins):
#     t_s, t_end, t_steps, x_init, f, F = ins
#     return RK(np.linspace(t_s, t_end, num=t_steps), x_init, f, F)[-1, :]

def RK_last_t(*args, **kwargs):
    s_time = time.time()
    ret = RK_last(*args, **kwargs)
    el_time = time.time() - s_time
    return ret, el_time

def RK_t(*args, **kwargs):
    s_time = time.time()
    ret = RK(*args, **kwargs)
    el_time = time.time() - s_time
    return ret, el_time

def RK(t, u0, f, method):
    if method == 'RK1':
        a = np.array([[0]]);
        b = np.array([[1]]); 
        c = np.array([[0]]);
    elif method == 'RK2':
        a = np.array([[0,0],[0.5,0]])
        b = np.array([[0,1]])
        c = np.array([0,0.5])
    elif method == 'RK4':  #classic fourth-order method
        a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
        b = np.array([[1/6,1/3,1/3,1/6]])
        c = np.array([0,0.5,0.5,1])
    elif method == 'RK8': #Cooper-Verner eigth-order method (again there are many)
        s = np.sqrt(21);
        a = np.array([[0,0,0,0,0,0,0,0,0,0,0],[1/2,0,0,0,0,0,0,0,0,0,0],[1/4,1/4,0,0,0,0,0,0,0,0,0],[1/7,(-7-3*s)/98,(21+5*s)/49,0,0,0,0,0,0,0,0],[(11+s)/84,0,(18+4*s)/63,(21-s)/252,0,0,0,0,0,0,0],[(5+s)/48,0,(9+s)/36,(-231+14*s)/360,(63-7*s)/80,0,0,0,0,0,0],[(10-s)/42,0,(-432+92*s)/315,(633-145*s)/90,(-504+115*s)/70,(63-13*s)/35,0,0,0,0,0],[1/14,0,0,0,(14-3*s)/126,(13-3*s)/63,1/9,0,0,0,0],[1/32,0,0,0,(91-21*s)/576,11/72,(-385-75*s)/1152,(63+13*s)/128,0,0,0],[1/14,0,0,0,1/9,(-733-147*s)/2205,(515+111*s)/504,(-51-11*s)/56,(132+28*s)/245,0,0],[0,0,0,0,(-42+7*s)/18,(-18+28*s)/45,(-273-53*s)/72,(301+53*s)/72,(28-28*s)/45,(49-7*s)/18,0]])
        b = np.array([[1/20,0,0,0,0,0,0,49/180,16/45,49/180,1/20]])
        c = np.array([[0,1/2,1/2,(7+s)/14,(7+s)/14,1/2,(7-s)/14,(7-s)/14,1/2,(7+s)/14,1]])
    else:
        raise Exception('exp_1')
        
    return np.array(RK_jax_(jnp.array(t), u0, f, jnp.array(a), jnp.array(b), jnp.array(c)) )
    # return RK_numpy_(t, u0, f, a, b, c)

def RK_jax_(t, u0, f, a, b, c):
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
    # return temp, k, f1
RK_jax_ = jax.jit(RK_jax_, static_argnums=(2,)) 

def RK_numpy_(t, u0, f, a, b, c):        
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

    
#%% Models

#%%% Mainstream

    
class ModelAbstr():
    
    def __init__(self, **kwargs):
        self.train_time = 0
        self.pred_time = 0
    
    
    def fit_timed(self, x, y, *args, **kwargs):
        s_time = time.time()
        ret = self.fit(x, y, *args, **kwargs)
        elap_time = time.time() - s_time 
        self.train_time += elap_time
        return ret
    
    def predict_timed(self, new_x, *args, **kwargs):
        s_time = time.time()
        ret = self.predict(new_x, *args, **kwargs)
        elap_time = time.time() - s_time 
        self.pred_time += elap_time
        return ret
    
    def get_times(self):
        return {'mdl_train_t':self.train_time, 'mdl_pred_t':self.pred_time, 'mdl_tot_t':self.train_time + self.pred_time}
    
    def fit(self, x, y, *args, **kwargs):
        self.x, self.y = x, y
        raise Exception('Not implemented')
    
    def predict(self, new_x, prev_F, prev_G):
        preds = None
        raise Exception('Not implemented')
        return preds
    
    def _print_cond(self, K, jitted=False):
        e_vals = np.abs(np.linalg.eig(K)[0])
        if jitted:
            print(f'--- Jitted: max |eig|: {e_vals.max():0.2e}, min |eig|: {e_vals.min():0.2e}, ratio: {e_vals.max()/e_vals.min():0.2e}, truth: {np.linalg.cond(K):0.2e}')
        else:
            print(f'-- max |eig|: {e_vals.max():0.2e}, min |eig|: {e_vals.min():0.2e}, ratio: {e_vals.max()/e_vals.min():0.2e}, truth: {np.linalg.cond(K):0.2e}')

        
        
class BareParareal(ModelAbstr):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Parareal'
    
    def fit(self, *args, **kwargs):
        pass
    
    def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
        return prev_F - prev_G
    
    
def _fit_gp_jit(x, y, theta, kernel, jitter):
    N = x.shape[0]
    K = kernel(x, x, theta) +  jnp.eye(N)*10**jitter
    L = jnp.linalg.cholesky(K)
    alph = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, y, lower=True), lower=False)
    return L, alph
_fit_gp_jit = jax.jit(_fit_gp_jit, static_argnums=(3,))
   
    
from itertools import repeat
from itertools import product
   
class NNGP_p(ModelAbstr):
    
    def __init__(self, n, N, worker_pool, theta=None, fatol=None, xatol=None, **kwargs):
        super().__init__(**kwargs)
        # theta = np.ones(n) if theta is None else theta
        if theta is None:
            theta=[1,1]
        self.theta = np.array(theta)
        self.name = 'NNGP'
        self.kernel = self.kernel_jit
        self.fatol = 1e-1 if fatol is None else fatol
        self.xatol = 1e-1 if xatol is None else xatol
        self.n = n
        self.n_restarts = kwargs.get('n_restarts', 1)
        self.nn = kwargs.get('nn', 'adaptive')
        self.seed = kwargs.get('seed', 45)
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        self.pool = worker_pool
        
        
    @staticmethod
    def k_gauss(xi, xj, params):
        sigma_x, sigma_y = params 
        return 10**(sigma_y) * jnp.exp(-0.5 * (1/(10**sigma_x)) * jnp.sum((xi-xj)**2))

    def kernel_jit(x, y, kernel_params):
        kernel = NNGP_p.k_gauss
        map_x = jax.vmap(kernel, in_axes=(None, 0, None)) 
        map_y = jax.vmap(map_x, in_axes=(0, None, None))
        return map_y(x, y, kernel_params)
    kernel_jit = staticmethod(jax.jit(kernel_jit))
    
    def fit(self, x, y, k, *args, **kwargs):
        self.k = k
        self.x, self.y = x, y
        
    def _predict(x, y, theta, kernel, jitter, new_x):
        N = x.shape[0]
        L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
        K_star = kernel(x, new_x, theta)
        post_mean = K_star.T @ alph
        return post_mean
    _predict = staticmethod(jax.jit(_predict, static_argnums=(3,)))
    
    
    def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
        if self.nn == 'adaptive':
            nn = max(10, self.k + 2)
        else:
            nn = self.nn
            
        s_idx = np.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
        xm = self.x[s_idx[:nn], :]
        ym = self.y[s_idx[:nn], :]
        
        n = self.n
        preds = self.get_preds(xm, ym, n, new_x)
        
        return preds
    
    def get_preds(self, xm, ym, n, new_x):
        jitter = np.arange(-20, -11, dtype=float)
        restarts = range(self.n_restarts)
        mdls = range(n)
        n_pars = self.theta.shape[0]
        ins = list(product(mdls, jitter, restarts))
        static_ins = (xm, ym, self.fatol, self.xatol)
        rnd = [self.rng.integers(-8, 0, (n_pars)) for i in range(len(ins))]
        out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins, rnd))
        
        preds = np.empty(n)
        for j in range(n):
            res = [i for i in out_res if i[-1] == j]
            res = np.array(res)
            mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.9
            if mask.sum()==0:
                mask[:] = True
            *opt_params, opt_fval, opt_jitter,_ = min(res[mask,:], key=lambda x: x[n_pars])
            
            y_mean = self._predict(xm, ym[:, j], opt_params, self.kernel, opt_jitter, new_x) 
            preds[j] = np.squeeze(y_mean)
            
        return preds
        
    @staticmethod
    def _get_opt_par(static_ins, ins, rnd):
        xm, ym, fatol, xatol = static_ins
        j, jitter, n_restarts = ins
        
        # x = jax.random.uniform(jax.random.PRNGKey(0), (1000,), dtype=jnp.float64)
        # print(x.dtype)
        
        kernel = NNGP_p.kernel_jit
        opt_params, opt_fval = NNGP_p.opt_theta(xm, ym[:,j], rnd, jitter, fatol, xatol, kernel)
        return (*opt_params, opt_fval, jitter, j)
    
    def _log_lik(x, y, theta, kernel, jitter):
        N = x.shape[0]
        L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
        res = -(-0.5 * y.T @ alph - jnp.sum(jnp.log(jnp.diag(L))) - (N/2)*jnp.log(2*jnp.pi))
        return res
    _log_lik = staticmethod(jax.jit(_log_lik, static_argnums=(3,)))
    
    @staticmethod
    def log_lik(x, y, theta, jitter, kernel):
        res = NNGP_p._log_lik(x, y, theta, kernel, jitter)
        if np.isnan(res):
            return np.inf
        return res
    
    @staticmethod
    def opt_theta(x, y, old_theta, jitter, fatol, xatol, kernel):
        _log_lik = lambda theta: NNGP_p.log_lik(x, y, theta, jitter, kernel)
        res =  minimize(_log_lik, old_theta, 
                        method='Nelder-Mead', 
                        options={'fatol':fatol, 'xatol':xatol})
        return tuple(res.x), res.fun
    
    
class GPjax_p(ModelAbstr):
    def __init__(self, n, N, worker_pool, theta=None, jitter=None, fatol=None, xatol=None, **kwargs):
        super().__init__(**kwargs)
        if theta is None:
            theta=[1,1]
        theta = np.array(theta)
        self.name = 'GP'
        self.kernel = GPjax_p.kernel_jit
        self.hyp = np.ones((n, theta.shape[0], N))
        self.thetas = [theta for i in range(n)] 
        self.jitters = [None for i in range(n)] 
        self.fatol = 1e-4 if fatol is None else fatol
        self.xatol = 1e-4 if xatol is None else xatol
        self.theta = theta
        # self.N = N
        self.n = n
        self.mem = {}
        self.pool = worker_pool
        
    @staticmethod
    def k_gauss(xi, xj, params):
        sigma_x, sigma_y = params 
        return (sigma_y**2) * jnp.exp(-0.5 * (1/(sigma_x**2)) * jnp.sum((xi-xj)**2))

    def kernel_jit(x, y, kernel_params):
        kernel = GPjax_p.k_gauss
        map_x = jax.vmap(kernel, in_axes=(None, 0, None)) 
        map_y = jax.vmap(map_x, in_axes=(0, None, None))
        return map_y(x, y, kernel_params)
    kernel_jit = staticmethod(jax.jit(kernel_jit))
        
    def _log_lik(x, y, theta, kernel, jitter):
        N = x.shape[0]
        L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
        res = -(-0.5 * y.T @ alph - jnp.sum(jnp.log(jnp.diag(L))) - (N/2)*jnp.log(2*jnp.pi))
        return res
    _log_lik = staticmethod(jax.jit(_log_lik, static_argnums=(3,)))
    
    @staticmethod
    def log_lik(x, y, theta, jitter):
        res = GPjax_p._log_lik(x, y, theta, GPjax_p.kernel_jit, jitter)
        if np.isnan(res):
            return np.inf
        return res    
    
    @staticmethod
    def opt_theta(x, y, old_theta, jitter, fatol, xatol):
        _log_lik = lambda theta: GPjax_p.log_lik(x, y, theta, jitter)
        res =  minimize(_log_lik, old_theta, 
                        method='Nelder-Mead', 
                        options={'fatol':fatol, 'xatol':xatol})
        return tuple(res.x), res.fun
    
    @staticmethod
    def _get_opt_par(static_ins, ins):
        x, y, old_thetas, fatol, xatol = static_ins
        j, jitter = ins
        opt_params, opt_fval = GPjax_p.opt_theta(x, y[:,j], old_thetas[j], jitter, fatol, xatol)
        return (*opt_params, opt_fval, jitter, j)
    
    def _train(self, x, y, old_thetas):
        jitter = np.arange(-20, -11, dtype=float)
        mdls = range(self.n)
        n_pars = self.theta.shape[0]
        ins = list(product(mdls, jitter))
        static_ins = (x, y, old_thetas, self.fatol, self.xatol)
        out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins))
        
        temp = np.zeros((self.n, n_pars))
        for j in range(self.n):
            res = [i for i in out_res if i[-1] == j]
            res = np.array(res)
            mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.9
            if mask.sum()==0:
                mask[:] = True
            *opt_params, opt_fval, opt_jitter,_ = min(res[mask,:], key=lambda x: x[n_pars])
            
            if np.isinf(opt_fval):
                raise Exception('Optimal loss should not be inf')
            
            self.thetas[j] = opt_params
            self.jitters[j] = opt_jitter
            temp[j,:] = opt_params
            
        return temp
    

    def fit(self, x, y, k, *args, **kwargs):
        
        new_hyp = self._train(x, y, self.thetas)
        self.hyp[...,k+1] = new_hyp
        
        self.x, self.y = x, y
        self.k = k
        

    def _predict(self, x, y, new_x, theta, jitter):
        N = x.shape[0]
        L, alph = self.mem.get(tuple(theta), (None, None))
        if L is None or L.shape[0] != x.shape[0]:
            K = self.kernel(x, x, theta)
            L = np.linalg.cholesky(K + np.eye(N)*10**jitter)
            alph = np.linalg.solve(L.T, np.linalg.solve(L,y))
            self.mem[tuple(theta)] = (L, alph)
        K_star = self.kernel(x, new_x, theta)
        v = np.linalg.solve(L, K_star)
        post_mean = K_star.T @ alph
        return post_mean
    
    
    def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
        n = self.n
        preds = np.empty(n)
        for j in range(n):
            y_mean = self._predict(self.x, self.y[:,j], new_x, self.thetas[j], self.jitters[j])
            preds[j] = np.squeeze(y_mean)
        return preds




#%% Parareal

from itertools import repeat
import concurrent.futures
from cycler import cycler

class MyPool():
    
    @staticmethod
    def map(*args, **kwargs):
        return map(*args, **kwargs)
    
    @staticmethod
    def shutdown(*args, **kwargs):
        pass

class Parareal():
    
    ### NOTE ###
    # To run parareal in parallel make sure you set the environment variable for jax
    # JAX_ENABLE_X64=True
    # so that it's 64 bit precision by default, otherwise new processes spawned by
    # joblib will have lower accuracy, yielding different results between parallel 
    # and serial
    
    def __init__(self, f=None, tspan=None, u0=None, N=None, Ng=None, Nf=None, epsilon=None, 
                 F=None, G=None, ode_name='No-Name', 
                 normalization='-11', RK_thresh=1e7):
        
        if sum(map(lambda x: x is None, [f, tspan, u0, N, Ng, Nf, epsilon, F, G])):
            f, tspan, u0, epsilon, N, Ng, Nf, G, F, data_tr, data_tr_inv = Systems(ode_name, normalization=normalization, u0=u0,
                                                                                   epsilon=epsilon).fetch()
        else:
            data_tr = lambda x: x
            data_tr_inv = lambda x: x
            
        u0 = np.array(u0)
        N = int(N)
        Ng = int(Ng)
        Nf = int(Nf)
        
        self.f = f
        self.tspan = tspan
        self.u0 = u0
        self.n = u0.shape[0] 
        self.N = N
        self.Ng = Ng
        self.Nf = Nf
        self.epsilon = epsilon
        self.F = F
        self.G = G
        self.runs = dict()
        self.fine = None
        self.ode_name = ode_name
        
        self.data_tr = data_tr
        self.data_tr_inv = data_tr_inv
        
        self.RK_thresh = RK_thresh  
        
    def run(self, *args, **kwargs):
        pool = kwargs.get('pool', None)
        if isinstance(pool, int):
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=pool)
        elif isinstance(pool, type(None)):
            pool = MyPool()
        kwargs['pool'] = pool
        
        try:
            out = self._run(*args, **kwargs)
        except Exception as e:
            pool.shutdown()
            raise
            
        pool.shutdown()
        return out
        
    def _run(self, model='parareal', cstm_mdl_name=None, add_model=False, **kwargs):
            
        if model.lower() == 'parareal':
            mdl = BareParareal(**kwargs)
        elif model.lower() == 'gpjax':
            if 'pool' not in kwargs:
                raise Exception('A worker pool must be provided to run NNGP in parallel')
            mdl = GPjax_p(n=self.n, N=self.N, worker_pool=kwargs['pool'], **kwargs)
        elif model.lower() == 'nngp':
            if 'pool' not in kwargs:
                raise Exception('A worker pool must be provided to run NNGP in parallel')
            mdl = NNGP_p(n=self.n, N=self.N, worker_pool=kwargs['pool'], **kwargs)
        else:
            raise Exception('Not implemented')
        
        
        s_time = time.time()
        out = self._parareal(mdl, **kwargs)
        elap_time = time.time() - s_time
        out['timings']['runtime'] = elap_time
        print(f'Elapsed Parareal time: {elap_time:0.2f}s')
        
        if add_model:
            out['mdl'] = mdl
        if cstm_mdl_name is None:
            cstm_mdl_name = mdl.name
        self.runs[cstm_mdl_name] = out
        return out
        
    def _parareal(self, model, early_stop=None, parall='Serial', **kwargs):
        f, tspan, u0, N, Ng = self.f, self.tspan, self.u0, self.N, self.Ng
        Nf, epsilon, F, G, n = self.Nf, self.epsilon, self.F, self.G, self.n
        
        L = tspan[1] - tspan[0]           #length of interval
        L_sub = L/N                       #length of sub-interval
        dT = L/Ng                         #coarse time step
        dt = L/Nf                         #fine time step
        t = np.linspace(tspan[0], tspan[1], num=N+1)     #time sub-intervals (the mesh)  
        t_shift = t[1:]               #shifted mesh for parfor loops below
        I = 0                             #counter for how many intervals have converged
        
        if (Ng % N != 0) or (Nf%Ng != 0):
            raise Exception('Nf must be a multiple of Ng and Ng must be a multiple of N - change time steps!')
            
        parall = parall.lower()
        if parall == 'mpi':
            if 'pool' not in kwargs:
                raise Exception('MPI parallel backend requested but no pool of worker provided')
            pool = kwargs['pool']
        
        u = np.empty((N+1, n, N+1))
        uG = np.empty((N+1, n, N+1))
        uF = np.empty((N+1, n, N+1))
        err = np.empty((N+1, N))
        u.fill(np.nan)
        uG.fill(np.nan)
        uF.fill(np.nan)
        err.fill(np.nan)
        
        x = np.zeros((0, n))
        D = np.zeros((0,n))
        data_x = np.empty((N, n, N))
        data_x.fill(np.nan)
        data_D = np.empty((N, n, N))
        data_D.fill(np.nan)
        
        G_time = 0
        F_time = 0
            
        u[0,:,:] = u0[:, np.newaxis]
        uG[0,:,:] = u[0,:,:]
        uF[0,:,:] = u[0,:,:]
        
        temp, temp_t = RK_t(np.linspace(t[0], t[-1], num=Ng+1), u0, f, G)
        G_time += temp_t
        uG[:,:,0] = temp[0::int(Ng/N), :]
        del temp, temp_t
        u[:,:,0] = uG[:,:,0]        
        
        #Step 2: integrate using F (fine solver) in parallel with the current best initial
        # values
        for k in range(N):
            # if k == 0:
            #     print(f'{model.name} iteration number (out of {N}): {k+1} ', end='')
            # else:
            #     print(k+1, end=' ')
            print(f'{self.ode_name} {model.name} iteration number (out of {N}): {k+1} ')
                
            s_time = time.time()
            if parall == 'mpi':
                ins = [(t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F) for i in range(I,N)]
                out = list(pool.map(RK_last, ins, repeat(self.RK_thresh)))
                uF[I+1:N+1,:,k] = np.array(out)
            elif parall == 'joblib':
                out = Parallel(-1)(delayed(lambda i: RK_last((t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F), self.RK_thresh))(i) for i in range(I,N))
                uF[I+1:N+1,:,k] = np.array(out)
            else:
                for i in range(I, N):
                    temp = RK_last((t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F), self.RK_thresh)
                    uF[i+1,:,k] = temp
            F_time += time.time() - s_time
            del s_time
            
            # save values forward (as solution at time I+1 is now converged)
            uG[I+1,:,(k+1):] = uG[I+1,:,k].reshape(-1,1)
            uF[I+1,:,(k+1):] = uF[I+1,:,k].reshape(-1,1)
            u[I+1,:,(k+1):] = uF[I+1,:,k].reshape(-1,1)
            I = I + 1
            
            # collect training data
            x = np.vstack([x, u[I-1:N+1-1,:,k]])
            D = np.vstack([D, uF[I:N+1,:,k] - uG[I:N+1,:,k]])
            data_x[I-1:N+1-1,:,k] = u[I-1:N+1-1,:,k]
            data_D[I-1:N+1-1,:,k] = uF[I:N+1,:,k] - uG[I:N+1,:,k]
            
            # early stop if only one interval was missing
            if I == N:
                print('WARNING: early stopping')
                err[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
                err[-1,k] = np.nextafter(epsilon, 0)
                break
            
            
            model.fit_timed(x, D, k, data_x=data_x, data_y=data_D)
            
            
            for i in range(I, N):
                # run G solver on best initial value
                temp, temp_t = RK_last_t((t[i], t[i+1], int(Ng/N)+1, u[i,:,k+1], f, G), self.RK_thresh)
                G_time += temp_t
                uG[i+1,:,k+1] = temp
                del temp, temp_t
                
                preds = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                                           uF[i+1,:,k], uG[i+1,:,k], i=i)
                
                # do predictor-corrector update
                # u[i+1,:,k+1] = uF[i+1,:,k] + uG[i+1,:,k+1] - uG[i+1,:,k]
                u[i+1,:,k+1] = preds + uG[i+1,:,k+1]
                
            # error catch
            a = 0
            if np.any(np.isnan(uG[:,:, k+1])):
                raise Exception("NaN values in initial coarse solve - increase Ng!")
                           
            # Step 4: Converence check
            # checks whether difference between solutions at successive iterations
            # are small, if so then that time slice is considered converged.               
            err[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
            err[I,k] = 0
            
            II = I;
            for p in range(II+1, N+1):
                if err[p, k] < epsilon:
                    u[p,:,k+2:] = u[p,:,k+1].reshape(-1,1)
                    uG[p,:,k+2:] = uG[p,:,k+1].reshape(-1,1)
                    uF[p,:,k+1:] = uF[p,:,k].reshape(-1,1)
                    I = I + 1
                else:
                    break
                
            print('--> Converged:', I)
            if I == N:
                break
            if (early_stop is not None) and k == (early_stop-1):
                print('Early stopping due to user condition.')
                break            
            
        timings = {'F_time':F_time, 'G_time': G_time}
        timings.update(model.get_times())
        
        return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 'data_x':data_x, 
                'data_D':data_D, 'timings':timings, 'converged':I==N}

    def _build_plot_data(self, t, u, err, **kwargs):
        Nf, N, u0, f, F = self.Nf, self.N, self.u0, self.f, self.F
        u_par = np.empty((int(Nf/N)*(N-1) + int(Nf/N)+1, u0.shape[0]))
        u_par.fill(np.nan)
        for i in range(N):
            temp = RK(np.linspace(t[i], t[i+1], num=int(Nf/N)+1), u[i, :, -1], f, F)
            u_par[i*int(Nf/N):(i+1)*int(Nf/N),:] = temp[:-1,:]
        u_par[-1] = temp[-1,:]
        
        u_interval = u
        u_continuous = u_par
        return {'u_int':u_interval, 'u_cont': u_continuous, 'err':err, 't':t}
        
    def clear_plot_obj(self):
        self.runs = dict()
    
    def plot(self, skip = [], add_name=True, add_title='', cstm_title=None):
        runs, tspan, Nf, u0 = self.runs, self.tspan, self.Nf, self.u0
        f, F, epsilon = self.f, self.F, self.epsilon
        
        if cstm_title is None:
            cstm_title = f'{self.ode_name}'
        
        if len(add_title) != 0:
            add_title = add_title + ' - '
        
        if self.fine is None:
            fine, fine_t = RK_t(np.linspace(tspan[0], tspan[-1], num=Nf+1), u0, f, F)
            self.fine, self.fine_t = fine, fine_t
        else:
            fine = self.fine
        
        plot_data = {key : self._build_plot_data(**runs[key]) for key in runs}
         
        cols = ['gray','green','blue','red', 'm', 'y', 'k']

        fig1, ax = plt.subplots(u0.shape[0],1)
        x_plot = np.linspace(tspan[0], tspan[-1], num=Nf+1)
        for i in range(u0.shape[0]):
            for _idx, mdl_name in enumerate(plot_data):
                y_plot = np.log10(np.abs(fine - plot_data[mdl_name]['u_cont']))
                ax[i].plot(x_plot, y_plot[:,i], linewidth=0.5, label=mdl_name, color=cols[_idx])
            ax[i].set_ylabel(f'$u_{{{i+1}}}$ log error')
            ax[i].axhline(np.log10(epsilon), linestyle='dashed', color='gray', linewidth=1, label='Tolerance')
        # ax[i].legend()
        ax[i].set_xlabel('$t$')
        if add_name:
            fig1.suptitle(f'{cstm_title} - {add_title}Algorithm error wrt fine solver')
        else:
            fig1.suptitle('Algorithm error wrt fine solver')
        fig1.tight_layout()
        
       
        
        styles = ['solid', 'dotted', 'dashed', 'dashdot']
        fig2, ax = plt.subplots()
        cycl = cycler(linestyle=styles, lw=[0.5, 1, 1, 1]) * cycler('color', cols)
        ax.set_prop_cycle(cycl)
        for mdl_name in plot_data:
            err = plot_data[mdl_name]['err']
            x_plot = np.arange(1, err.shape[-1]+1)
            
            idx = 1
            out = np.empty(err.shape[1])
            out.fill(np.nan)
            one_step_err = np.empty(err.shape[1])
            one_step_err.fill(np.nan)
            for i in range(err.shape[1]):
                one_step_err[i] = err[np.argmax(err[:, i] > 0),i]
                if not np.any(err[idx:, i] >= epsilon):
                    n_conv = err.shape[0]-idx
                else:
                    n_conv = np.argmax(err[idx:, i] >= epsilon) -1 + 1
                    n_conv = n_conv if err[idx+n_conv, i] else err.shape[0]-idx
                    idx += n_conv
                out[i] = n_conv
                
                
            
            y_plot = out
            line2d1, = ax.plot(x_plot, np.cumsum(y_plot),  label=mdl_name[:18])
            ax.scatter(x_plot, np.cumsum(y_plot), s=1, color=line2d1.get_color())
            
        ax.axhline(err.shape[0]-1, linestyle='dashed', color='gray', linewidth=1)
        leg = ax.legend(loc='upper left', bbox_to_anchor= (1, 1), fontsize='small')
        
        if add_name:
            ax.set_title(f'{cstm_title}')
        else:
            ax.set_title(f'# Converged Intervals')
        ax.set_xlabel('k')
        ax.set_ylabel('# Converged Intervals')
        fig2.tight_layout()
        
        fig1.savefig(os.path.join('img', f'{self.ode_name}_{str(self.epsilon)[-1]}_prec.pdf'))
        fig1.savefig(os.path.join('img', f'{self.ode_name}_{str(self.epsilon)[-1]}_prec'))
        
        fig2.savefig(os.path.join('img', f'{self.ode_name}_{str(self.epsilon)[-1]}_conv.pdf'))
        fig2.savefig(os.path.join('img', f'{self.ode_name}_{str(self.epsilon)[-1]}_conv'))
        
    def print_speedup(self, mdls=None, md=True, fine_t=None, F_t=None, mdl_title=''):
        out = []
        if md:
            beg = '|'
            end = '|'
            sep = ' | '
            F = 'F'
            G = 'G'
        else:
            beg = ''
            end = '\\\\'
            sep = ' & '
            F = '$T_{\\f}$'
            G = '$T_{\\g}$'
        
        str_format = lambda x: f'{x:.2e}'
        out.append([ 'Model', 'K', G,F, 'Model', 'Total', 'Speed-up'])
        n = len(out[0])
        if F_t is not None:
            fine_t = F_t*self.N
        if md:
            out.append(['---']*n )
        else:
            out.append([r'\hline'])
        if fine_t is None:
            fine_t = self.fine_t
        if fine_t is None:
            raise Exception('Running time of fine solver unknown/not provided')
            
        mdl_map = {'GP':'GParareal', 'NNGP':'NN-GParareal'}
        out.append([ 'Fine', '-', '-','-', '-', str_format(fine_t), '1'])
        if mdls is None:
            mdls = {i:i for i in list(self.runs.keys())}        
        for k, v in mdls.items():
            if k not in self.runs:
                raise Exception('Unknown model', k)
            mdl = self.runs[k]
            if F_t is not None:
                tot_spd = F_t * mdl['k'] + mdl['timings']['mdl_tot_t']
                speedup = f'{fine_t/tot_spd:.2f}'
            else:
                speedup = f'{fine_t/mdl["timings"]["runtime"]:.2f}'
            temp = [mdl_map.get(v,v), mdl['k'], str_format(mdl['timings']['G_time']/mdl['k']), str_format(mdl['timings']['F_time']/mdl['k']), 
                    str_format(mdl['timings']['mdl_tot_t']), str_format(mdl['timings']['runtime']), 
                    speedup]
            out.append(temp)
            
        out = [[str(j)for j in i] for i in out]
        out = [beg+sep.join(i)+end for i in out]
        if not md:
            temp = [r'\caption*{' + mdl_title + r', $N=' + f'{self.N}' +r'$}']
            temp.append(r'\begin{tabular}{lcccccc}')
            temp.extend(out)
            temp.append(r'\end{tabular}\\    \bigskip'+'\n')
            out = temp
        else:
            temp = [f'$N={self.N}$\n']
            temp.extend(out)
            out=temp
        out = '\n'.join(out)
        print(out)
        return out
            
        
        
#%% Systems
   
class Systems:
    # This is just a container basically, not really using the idea of objects properly but 
    # just a simple way to store things neatly
    
    avail_odes = ['rossler_long', 'non_aut', 'fhn', 'dbl_pend', 'brus_2d', 'lorenz']
    
    def __init__(self, ode_name, normalization='-11', *args, **kwargs):
        
        if sum(map(lambda x: x in ode_name.lower(), self.avail_odes)) != 1:
            raise Exception(f'Unknown ode {ode_name}')
        
        if normalization not in ['-11', 'identity']:
            raise Exception('Unknown value of normalizaiton')
            
        if ode_name.lower()[-2:] == '_n':
            self.normalization = normalization
            ode_name = ode_name[:-2]
            self.normalize = True
        else:
            self.normalization = 'identity'
            self.normalize = False
            
        if ode_name.lower() == 'rossler_long':
            self.ft = self.get_rossler_long
        elif ode_name.lower() == 'fhn':
            self.ft = self.get_fhn
        elif 'non_aut' in ode_name.lower():
            try:
                N = int(ode_name[7:])
            except ValueError as e:
                raise Exception(f'Invalid interval number for non aut system: {ode_name}, {ode_name[7:]}') from None
            kwargs['N'] = N
            self.ft = self.get_non_aut
        elif ode_name.lower() == 'dbl_pend':
            self.ft = self.get_dbl_pend
        elif ode_name.lower() == 'brus_2d':
            self.ft = self.get_brus_2d
        elif ode_name.lower() == 'lorenz':
            self.ft = self.get_lorenz
            
        self.f = lambda: self.ft(*args, **kwargs)
        self.u0 = kwargs.get('u0', None)
            
    @staticmethod
    def _tr(x, mn, mx):
        return 2*(x-mn)/(mx-mn)-1
    
    @staticmethod
    def _tr_inv(x, mn, mx):
        return (x+1)/2 * (mx-mn) + mn
    
    @staticmethod
    def _scale(mn, mx):
        return 2/(mx-mn)
    
    def gen_transform(self, bounds):
        if self.normalization == '-11':
            tr = lambda x: self._tr(x, *bounds)
            tr_inv = lambda x: self._tr_inv(x, *bounds)
            scale = self._scale(*bounds)
        elif self.normalization == 'identity':
            tr = lambda x: x
            tr_inv = lambda x: x
            scale = 1
        else:
            raise Exception(f'Normalization mode {self.normalization} unknown.')
        return tr, tr_inv, scale
            
            
    def fetch(self):
        f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds = self.f()
        data_tr, data_tr_inv, scale = self.gen_transform(bounds)
        if self.u0 is not None:
            u0 = self.u0
        if self.normalize:
            return f, tspan, data_tr(u0), epsilon, N, Ng, Nf, G, F, data_tr, data_tr_inv
        else:
            return f, tspan, u0, epsilon, N, Ng, Nf, G, F, data_tr, data_tr_inv
        
            
    @staticmethod
    def f_fhn(t, u):
        a = 0.2 
        b = 0.2
        c = 3
        out = jnp.zeros(u.shape)
        out = out.at[0].set(c*(u[0] - ((u[0]**3)/3) + u[1]))
        out = out.at[1].set(-(1/c)*(u[0] - a + b*u[1]))
        return out
    
    @staticmethod
    def f_fhn_n(t,u):
        mn, mx = jnp.array([[-2, -1], [2.1, 1.2]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_fhn(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_fhn(self, *args, **kwargs):
        if self.normalize:
            f = self.f_fhn_n
        else:
            f = self.f_fhn
        
        
        tspan = [0,40]                    
        u0 = np.array([-1,1]) if kwargs.get('u0', None) is None  else kwargs['u0']
        epsilon = 10**(-6) if kwargs.get('epsilon', None) is None  else kwargs['epsilon']                 
        N = 40;                            
        Ng = N*4;                          
        Nf = int(160000/160*Ng); 
        G = 'RK2';                         
        F = 'RK4';    
        bounds = np.array([[-2, -1], [2.1, 1.2]]) #mins, maxs

        return f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds
    
    
    @staticmethod
    def f_ross(t, u):
        a = 0.2
        b = 0.2
        c = 5.7
        out = jnp.zeros(u.shape)
        out = out.at[0].set(-u[1]-u[2])
        out = out.at[1].set(u[0]+(a*u[1]))
        out = out.at[2].set(b+u[2]*(u[0]-c))
        return out
    
    @staticmethod
    def f_ross_n(t,u):
        mn, mx = jnp.array([[-10, -11, 0], [12, 8, 23]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_ross(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_rossler_long(self, *args, **kwargs):
        if self.normalize:
            f = self.f_ross_n
        else:
            f = self.f_ross

        tspan = [0,170]                   #time interval
        u0 = np.array([0,-6.78,0.02])               #intial conditions
        N = 20                            #no. of time sub-intervals steps
        Ng = 45000                        #no. of coarse steps
        Nf = 2250000                    #no. of fine steps
        epsilon = 10**(-6) if kwargs.get('epsilon', None) is None  else kwargs['epsilon'] 
        G = 'RK1'                         #coarse solver (see RK.m file)
        F = 'RK4'                         #fine solver   (see RK.m file)


        tspan2 = [0, tspan[-1]*2]
        N2 = N*2
        Ng2 = Ng*2
        Nf2 = Nf*2
        bounds = np.array([[-10, -11, 0], [12, 8, 23]]) #mins, maxs

        return f, tspan2, u0, epsilon, N2, Ng2, Nf2, G, F, bounds
    
    
    @staticmethod
    def f_nonaut(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(-u[1]+u[0]*((u[2]/500)-u[0]**2-u[1]**2))
        out = out.at[1].set(u[0]+u[1]*((u[2]/500)-u[0]**2-u[1]**2))
        out = out.at[2].set(1)
        return out
    
    @staticmethod
    def f_nonaut_n(t,u):
        mn, mx = jnp.array([[-23, -23, 0], [23, 23, 1]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_nonaut(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_non_aut(self, N=32, *args, **kwargs):
        if self.normalize:
            f = self.f_nonaut_n
        else:
            f = self.f_nonaut

        tspan = [-20, 500]                   #time interval
        u0 = np.array([0.1,0.1,tspan[0]])               #intial conditions
        
        Ng = 2*1024                        #no. of coarse steps
        Nf = Ng*85                    #no. of fine steps
        epsilon = 10**(-6) if kwargs.get('epsilon', None) is None  else kwargs['epsilon'] 
        G = 'RK1'                         #coarse solver (see RK.m file)
        F = 'RK8' 
        bounds = np.array([[-23, -23, 0], [23, 23, 1]]) #mins, maxs

        return f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds
    
    
    
    
    @staticmethod
    def f_pend(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(u[1])
        out = out.at[1].set((-1/(2-jnp.cos(u[0]-u[2])**2))*((u[1]**2)*jnp.cos(u[0]-u[2])*jnp.sin(u[0]-u[2])+(u[3]**2)*jnp.sin(u[0]-u[2])+2*jnp.sin(u[0])-jnp.cos(u[0]-u[2])*jnp.sin(u[2])))
        out = out.at[2].set(u[3])
        out = out.at[3].set((-1/(2-jnp.cos(u[0]-u[2])**2))*(-2*(u[1]**2)*jnp.sin(u[0]-u[2])-(u[3]**2)*jnp.sin(u[0]-u[2])*jnp.cos(u[0]-u[2])-2*jnp.cos(u[0]-u[2])*jnp.sin(u[0])+2*jnp.sin(u[2])))
        return out
    
    @staticmethod
    def f_pend_n(t,u):
        mn, mx = jnp.array([[-2, -2.5, -17, -3.5], [2, 2.5, 1, 3.5]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_pend(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_dbl_pend(self, *args, **kwargs):
        if self.normalize:
            f = self.f_pend_n
        else:
            f = self.f_pend


        tspan = [0,80]                    
        u0 = np.array([-0.5,0,0,0])                    
        N = 32                       
        Ng = 3072+N                    
        Nf = Ng*70                         
        epsilon = 10**(-6) if kwargs.get('epsilon', None) is None  else kwargs['epsilon']                  
        G = 'RK1'                           
        F = 'RK8'   
        bounds = np.array([[-2, -2.5, -17, -3.5], [2, 2.5, 1, 3.5]]) #mins, maxs

        return f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds
    
    
    
    
    @staticmethod
    def f_brus(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(1+(u[0]**2)*u[1]-(3+1)*u[0])
        out = out.at[1].set(3*u[0]-(u[0]**2)*u[1])
        return out
    
    @staticmethod
    def f_brus_n(t,u):
        mn, mx = jnp.array([[0.4, 0.9], [4, 5]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_brus(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_brus_2d(self, *args, **kwargs):
        if self.normalize:
            f = self.f_brus_n
        else:
            f = self.f_brus


        tspan = [0,100]                      
        u0 = np.array([1,3.07])                         
        N = 25                                
        Ng = N*10                               
        Nf = Ng*100                           
        epsilon = 10**(-6)  if kwargs.get('epsilon', None) is None  else kwargs['epsilon']                  
        G = 'RK4'                           
        F = 'RK4'  
        bounds = np.array([[0.4, 0.9], [4, 5]]) #mins, maxs

        return f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds
    
    
    
    @staticmethod
    def f_lorenz(t, u):
        out = jnp.zeros(u.shape)
        out = out.at[0].set(10*(u[1]-u[0]))
        out = out.at[1].set(28*u[0]-u[1]-u[0]*u[2])
        out = out.at[2].set(u[0]*u[1]-(8/3)*u[2])
        return out
    
    @staticmethod
    def f_lorenz_n(t,u):
        mn, mx = jnp.array([[-17.1, -23, 6], [18.1, 25, 45]])
        u = Systems._tr_inv(u, mn, mx)
        out = Systems.f_lorenz(t, u)
        out = out * Systems._scale(mn, mx)
        return out
    
    def get_lorenz(self, *args, **kwargs):
        if self.normalize:
            f = self.f_lorenz_n
        else:
            f = self.f_lorenz


        tspan = [0,18]
        u0 = np.array([-15,-15,20])
        N = 50
        Ng = N*6                     
        Nf = Ng*75      
        epsilon = 10**(-8) if kwargs.get('epsilon', None) is None  else kwargs['epsilon']                
        G = 'RK4'                            
        F = 'RK4' 
        bounds = np.array([[-17.1, -23, 6], [18.1, 25, 45]]) #mins, maxs

        return f, tspan, u0, epsilon, N, Ng, Nf, G, F, bounds
    
  