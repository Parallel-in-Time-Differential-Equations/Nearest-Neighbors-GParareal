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
import pickle
import copy

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


# 2023_10_24: 
#     - Fixed all bugs, incorporated kernel in model.
#     - Removed duplicated models NNGP and GPjax and GParareal. Rename *_p to *
#     - Made original *_p serial if no pool is given. A number of processor can also
#       be specified, no need of remembering the import.
#     - Add intermediate run to create and shutdown the pool
#     - Drastically change Systems so the normalized function can be pickled

# new:
    # new_lib just became equal to article_lib, except for parareal, to revert 2023_10_24
    

   
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
        N = kwargs['N']
        self.pred_times = np.zeros(N)
    
    
    def fit_timed(self, x, y, *args, **kwargs):
        self.time_k = kwargs['k']
        s_time = time.time()
        ret = self.fit(x, y, *args, **kwargs)
        elap_time = time.time() - s_time 
        self.train_time += elap_time
        self.pred_times[self.time_k] += elap_time
        return ret
    
    def predict_timed(self, new_x, *args, **kwargs):
        s_time = time.time()
        ret = self.predict(new_x, *args, **kwargs)
        elap_time = time.time() - s_time 
        self.pred_time += elap_time
        self.pred_times[self.time_k] += elap_time
        return ret
    
    def get_times(self):
        return {'mdl_train_t':self.train_time, 'mdl_pred_t':self.pred_time, 'mdl_tot_t':self.train_time + self.pred_time, 'by_iter':self.pred_times[:self.time_k+1]}
    
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

    def store(self):
        if hasattr(self, 'pool'):
            pool = self.pool
            self.pool = None
            new = copy.deepcopy(self)
            self.pool = pool
        else:
            new = copy.deepcopy(self)
        return new
        
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
        super().__init__(N=N, **kwargs)
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
        
        # compute average time for hyper_param opt across
        self.tot_train_t = 0
        self.train_count = 0
        self.calc_detail_avg = kwargs.get('calc_detail_avg', False)
        self.calc_parall_overhead = kwargs.get('calc_parall_overhead', False)
        if self.calc_detail_avg:
            self.detail_avg = np.zeros((N,N))
        if self.calc_parall_overhead:
            self.overhead = np.zeros((N,N))
        
    def get_times(self):
        out = super().get_times()
        if self.calc_detail_avg:
            detail_avg = self.detail_avg[:self.k+1,:]
        else:
            detail_avg = None
        if self.calc_parall_overhead:
            overhead = self.overhead[:self.k+1,:]
        else:
            overhead = None
            
        out.update({'serial_train_time': self.tot_train_t,
                    'calc_detail_avg': detail_avg, 
                    'overhead': overhead, 
                    'avg_serial_train_time': self.tot_train_t/self.train_count})
        return out
        
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
        preds = self.get_preds(xm, ym, n, new_x, kwargs['i'])
        return preds
    
    def get_preds(self, xm, ym, n, new_x, intrvl_i):
        jitter = np.arange(-20, -11, dtype=float)
        restarts = range(self.n_restarts)
        mdls = range(n)
        n_pars = self.theta.shape[0]
        ins = list(product(mdls, jitter, restarts))
        static_ins = (xm, ym, self.fatol, self.xatol)
        rnd = [self.rng.integers(-8, 0, (n_pars)) for i in range(len(ins))]
        
        # you might want to compute the parallel overhead, for that
        if self.calc_parall_overhead:
            _overhead = time.time()
            out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins, rnd))
            _overhead = time.time() - _overhead
            _overhead = _overhead - np.array(out_res)[:,-1].sum()
            self.overhead[self.k, intrvl_i] = _overhead
        else:
            out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins, rnd))      
        
        tot_time = 0
        time_c = 0
        preds = np.empty(n)
        for j in range(n):
            res = [i for i in out_res if i[-2] == j]
            res = np.array(res)
            tot_time += res[:, -1].sum()
            time_c += res.shape[0]
            mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.9
            if mask.sum()==0:
                mask[:] = True
            *opt_params, opt_fval, opt_jitter,_,_ = min(res[mask,:], key=lambda x: x[n_pars])
            
            y_mean = self._predict(xm, ym[:, j], opt_params, self.kernel, opt_jitter, new_x) 
            preds[j] = np.squeeze(y_mean)
        
        assert time_c == len(out_res)
        self.tot_train_t += tot_time
        self.train_count += len(out_res)
        if self.calc_detail_avg:
            self.detail_avg[self.k, intrvl_i] = tot_time/len(out_res)
            
        return preds
        
    @staticmethod
    def _get_opt_par(static_ins, ins, rnd):
        st_ = time.time()
        xm, ym, fatol, xatol = static_ins
        j, jitter, n_restarts = ins
        
        kernel = NNGP_p.kernel_jit
        opt_params, opt_fval = NNGP_p.opt_theta(xm, ym[:,j], rnd, jitter, fatol, xatol, kernel)
        elap_t = time.time() - st_
        return (*opt_params, opt_fval, jitter, j, elap_t)
    
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
    
    def store(self):
        new = super().store()
        new.kernel = None
        new.pool = None
        return new
    
    def restore_attrs(self, pool):
        self.pool = pool
        self.kernel = NNGP_p.kernel_jit
    
    
class GPjax_p(ModelAbstr):
    def __init__(self, n, N, worker_pool, theta=None, jitter=None, fatol=None, xatol=None, **kwargs):
        super().__init__(N=N, **kwargs)
        if theta is None:
            theta=[1,1]
        theta = np.array(theta)
        self.name = 'GP'
        self.kernel = GPjax_p.kernel_np
        self.hyp = np.ones((n, theta.shape[0], N))
        self.thetas = [theta for i in range(n)] 
        self.jitters = [None for i in range(n)] 
        self.fatol = 1e-4 if fatol is None else fatol
        self.xatol = 1e-4 if xatol is None else xatol
        self.theta = theta
        self.N = N
        self.n = n
        self.mem = {}
        self.pool = worker_pool
        self.rng = np.random.default_rng(45)
        
        # computes serial training times
        self.tot_train_t = np.zeros(N)
        self.train_count = np.zeros(N)
        
    def get_times(self):
        out = super().get_times()
        out.update({'serial_train_time': self.tot_train_t[:self.k+1], 
                    'avg_serial_train_time': (self.tot_train_t/self.train_count)[:self.k+1]})
        return out
        
    @staticmethod
    def kernel_np(x, y, kernel_params):
        sigma_x, sigma_y = kernel_params 
        dist = scipy.spatial.distance.cdist(x, y, metric='sqeuclidean')
        return (sigma_y**2) * jnp.exp(-0.5 * (1/(sigma_x**2)) * dist)
    
    @staticmethod
    def _fit_gp_np(x, y, theta, kernel, jitter):
        N = x.shape[0]
        K = kernel(x, x, theta) +  np.eye(N)*10**jitter
        L = np.linalg.cholesky(K)
        alph = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y, lower=True), lower=False)
        return L, alph

    @staticmethod
    def _log_lik_np(x, y, theta, kernel, jitter):
        N = x.shape[0]
        L, alph = GPjax_p._fit_gp_np(x, y, theta, kernel, jitter)
        res = -(-0.5 * y.T @ alph - np.sum(jnp.log(jnp.diag(L))) - (N/2)*np.log(2*np.pi))
        return res
    
    @staticmethod
    def log_lik(x, y, theta, jitter):
        try:
            res = GPjax_p._log_lik_np(x, y, theta, GPjax_p.kernel_np, jitter)
        except np.linalg.LinAlgError as e:
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
        st_ = time.time()
        x, y, old_thetas, fatol, xatol = static_ins
        j, jitter = ins
        opt_params, opt_fval = GPjax_p.opt_theta(x, y[:,j], old_thetas[j], jitter, fatol, xatol)
        elap_t = time.time() - st_
        return (*opt_params, opt_fval, jitter, j, elap_t)
    
    @staticmethod
    def _get_opt_par_rnd(static_ins, ins, theta):
        st_ = time.time()
        x, y, fatol, xatol = static_ins
        j, jitter = ins
        opt_params, opt_fval = GPjax_p.opt_theta(x, y[:,j], theta, jitter, fatol, xatol)
        elap_t = time.time() - st
        return (*opt_params, opt_fval, jitter, j, elap_t)
    
    def _train_coord_rnd(self, x, y, coord):
        jitter = np.arange(-20, -11, dtype=float)
        n_pars = self.theta.shape[0]
        tot_rnd = max(3, int(self.N/9))
        ins = list(product([coord for i in range(tot_rnd)], jitter))
        thetas = [10**self.rng.uniform(-4, 1, (n_pars)) for i in range(len(ins))]
        static_ins = (x, y, self.fatol, self.xatol)
        out_res = list(self.pool.map(self._get_opt_par_rnd, repeat(static_ins), ins, thetas))
        
        res = np.array(out_res)
        tot_time = res[:, -1].sum()
        time_c = res.shape[0]
        mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.9
        if mask.sum()==0:
            mask[:] = True
        *opt_params, opt_fval, opt_jitter,_,_ = min(res[mask,:], key=lambda x: x[n_pars])
        
        self.tot_train_t[self.k] += tot_time
        self.train_count[self.k] += len(out_res)
            
        if np.isinf(opt_fval):
            print('random restart failed')
            print(res)
            print('x shape, y shape, coord', x.shape, y.shape, coord)
            opt_params, opt_fval, opt_jitter = self._train_coord_rnd(x, y, coord)
            
        return opt_params, opt_fval, opt_jitter
    
    def _train(self, x, y, old_thetas):
        jitter = np.arange(-20, -11, dtype=float)
        mdls = range(self.n)
        n_pars = self.theta.shape[0]
        ins = list(product(mdls, jitter))
        static_ins = (x, y, old_thetas, self.fatol, self.xatol)
        out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins))
        
        tot_time = 0
        time_c = 0
        temp = np.zeros((self.n, n_pars))
        for j in range(self.n):
            res = [i for i in out_res if i[-2] == j]
            res = np.array(res)
            tot_time += res[:, -1].sum()
            time_c += res.shape[0]
            mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.9
            if mask.sum()==0:
                mask[:] = True
            *opt_params, opt_fval, opt_jitter,_,_ = min(res[mask,:], key=lambda x: x[n_pars])

            if np.isinf(opt_fval):
                print('------> GP trainign failed for coordinate', j)
                print(res)
                print('x shape, y shape, coord', x.shape, y.shape, j)
                print(old_thetas[j])
                opt_params, opt_fval, opt_jitter = self._train_coord_rnd(x, y, j)
                
            self.thetas[j] = opt_params
            self.jitters[j] = opt_jitter
            temp[j,:] = opt_params
            
        self.tot_train_t[self.k] += tot_time
        self.train_count[self.k] += len(out_res)
            
        return temp
    

    def fit(self, x, y, k, *args, **kwargs):
        self.mem = {}
        self.k = k
        new_hyp = self._train(x, y, self.thetas)
        self.hyp[...,k+1] = new_hyp
        
        self.x, self.y = x, y
        
        

    def _predict(self, x, y, new_x, theta, jitter):
        N = x.shape[0]
        L, alph = self.mem.get(tuple(theta), (None, None))
        if L is None or L.shape[0] != x.shape[0]:
            K = self.kernel(x, x, theta)
            L = np.linalg.cholesky(K + np.eye(N)*10**jitter)
            alph = np.linalg.solve(L.T, np.linalg.solve(L,y))
            self.mem[tuple(theta)] = (L, alph)
        K_star = self.kernel(x, new_x, theta)
        # v = np.linalg.solve(L, K_star)
        post_mean = K_star.T @ alph
        return post_mean
    
    
    def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
        n = self.n
        preds = np.empty(n)
        for j in range(n):
            y_mean = self._predict(self.x, self.y[:,j], new_x, self.thetas[j], self.jitters[j])
            preds[j] = np.squeeze(y_mean)
        return preds
    
    def store(self):
        new = super().store()
        new.kernel = None
        new.pool = None
        new.mem = None
        new.hyp = new.hyp[...,:self.k+3]
        return new

    def restore_attrs(self, pool):
        self.pool = pool
        self.kernel = GPjax_p.kernel_np
        self.mem = {}
        hyp = np.ones((self.n, self.theta.shape[0], self.N))
        hyp[..., :self.hyp.shape[-1]:1] = self.hyp
        self.hyp = hyp


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
                 normalization='-11', RK_thresh=1e7, verbose='v'):
        
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
        self.verbose = verbose
        
    def _get_pool(self, *args, **kwargs):
        pool = kwargs.get('pool', None)
        if isinstance(pool, int):
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=pool)
        elif isinstance(pool, type(None)):
            pool = MyPool()
        return pool
        
        
    def run(self, *args, **kwargs):
        pool = self._get_pool(*args, **kwargs)
        kwargs['pool'] = pool
        try:
            if kwargs.get('_run_from_int', False):
                out = self._run_from_int(*args, **kwargs)
            else:
                out = self._run(*args, **kwargs)
        except Exception as e:
            pool.shutdown()
            raise
            
        pool.shutdown()
        return out
        
    def _run(self, model='parareal', cstm_mdl_name=None, add_model=False, **kwargs):
            
        if model.lower() == 'parareal':
            mdl = BareParareal(N=self.N, **kwargs)
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
        if self.verbose == 'v':
            print(f'Elapsed Parareal time: {elap_time:0.2f}s')
        
        if add_model:
            out['mdl'] = mdl.store()
        if cstm_mdl_name is None:
            cstm_mdl_name = mdl.name
        self.runs[cstm_mdl_name] = out
        return out
    
    def store(self, name, path='', mdl=None, objs=None):
        # if path doesn't exist, create it
        if not os.path.exists(path) and len(path) > 0:
            os.makedirs(path)
        
        data_tr = self.data_tr
        data_tr_inv = self.data_tr_inv
        f = self.f
        self.data_tr = None
        self.data_tr_inv = None
        self.f = None
        if objs is not None:
            pool = objs['kwargs']['pool']
            objs['kwargs']['pool'] = None
            self.objs = objs
        if mdl is not None:
            self.mdl = mdl.store()
            
        with open(os.path.join(path, name), 'wb') as _file:
            pickle.dump(self, _file, pickle.HIGHEST_PROTOCOL)
            
        self.data_tr = data_tr
        self.data_tr_inv = data_tr_inv
        self.f = f
        if objs is not None:
            self.objs = None
            objs['kwargs']['pool'] = pool
        if mdl is not None:
            self.mdl = None
            
    def load_int_dump(self, other, f, cstm_mdl_name=None, add_model=False, **kwargs):
        self.f = f
        self.tspan = other.tspan
        self.u0 = other.u0
        self.n = other.u0.shape[0] 
        self.N = other.N
        self.Ng = other.Ng
        self.Nf = other.Nf
        self.epsilon = other.epsilon
        self.F = other.F
        self.G = other.G
        self.runs = other.runs
        self.fine = other.fine
        self.ode_name = other.ode_name
        
        self.data_tr = kwargs.get('data_tr', other.data_tr)
        self.data_tr_inv = kwargs.get('data_tr_inv', other.data_tr_inv)
        
        self.RK_thresh = other.RK_thresh  
        self.verbose = other.verbose
        
        # get pool and prepping to run
        objs = other.objs
        other_run_kwargs = objs['kwargs']
        other_run_kwargs['_run_from_int'] = True
        other_run_kwargs.update(kwargs)
        mdl = other.mdl
        
        base_time = objs['F_time']+objs['G_time']+mdl.get_times()['mdl_tot_t']
        
        err = objs['err']
        idx = 1
        out = np.empty(err.shape[1])
        out.fill(np.nan)
        one_step_err = np.empty(err.shape[1])
        one_step_err.fill(np.nan)
        for i in range(err.shape[1]):
            one_step_err[i] = err[np.argmax(err[:, i] > 0),i]
            if not np.any(err[idx:, i] >= other.epsilon):
                n_conv = err.shape[0]-idx
            else:
                n_conv = np.argmax(err[idx:, i] >= other.epsilon) -1 + 1
                n_conv = n_conv if err[idx+n_conv, i] else err.shape[0]-idx
                idx += n_conv
            out[i] = n_conv
        
        objs['conv_int'] = objs.get('conv_int', list(np.cumsum(out)))
        obj_names = ['L', 'L_sub', 'dT', 'dt', 't', 't_shift', 'I', 'verbose', 
                     'conv_int', 'u', 'uG', 'uF', 'err', 'x', 'D', 'data_x', 
                     'data_D', 'G_time', 'F_time', 'k']
        reload_objs = [objs[nm] for nm in obj_names]
        
        return self.run(mdl, base_time, cstm_mdl_name, add_model, _reload_objs=reload_objs, **other_run_kwargs)
        
        
    def _run_from_int(self, mdl, base_time, cstm_mdl_name, add_model, **kwargs):
        
        # add pool to mdl
        mdl.restore_attrs(kwargs['pool']) 
        
        s_time = time.time()
        out = self._parareal(mdl, _load_mdl=True, **kwargs) 
        elap_time = time.time() - s_time + base_time
        out['timings']['runtime'] = elap_time
        if self.verbose == 'v':
            print(f'Elapsed Parareal time: {elap_time:0.2f}s')
        
        if add_model:
            out['mdl'] = mdl.store()
        if cstm_mdl_name is None:
            cstm_mdl_name = mdl.name
        self.runs[cstm_mdl_name] = out
        return out
                        
        
    def _parareal(self, model, debug=False, early_stop=None, lag_k=None, parall='Serial', store_int=False, _load_mdl=False, **kwargs):
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
            
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = self.verbose
            
        conv_int = []
        
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
        F_time_serial = 0
        
            
        u[0,:,:] = u0[:, np.newaxis]
        uG[0,:,:] = u[0,:,:]
        uF[0,:,:] = u[0,:,:]
        
        if debug:
            mean_errs = []
            max_errs = []
            one_step_error = []
            all_pred_err = []
        
        
        temp, temp_t = RK_t(np.linspace(t[0], t[-1], num=Ng+1), u0, f, G)
        G_time += temp_t
        uG[:,:,0] = temp[0::int(Ng/N), :]
        del temp, temp_t
        u[:,:,0] = uG[:,:,0]

        if _load_mdl:
            L, L_sub, dT, dt, t, t_shift, I, verbose, conv_int, _u, _uG, _uF, _err, x, D, _data_x, _data_D, G_time, F_time, _k = kwargs['_reload_objs']
            u[..., :_k+2] = _u
            uG[..., :_k+2] = _uG
            uF[..., :_k+2] = _uF
            err[..., :_k+2] = _err
            data_x[..., :_k+2] = _data_x
            data_D[..., :_k+2] = _data_D
            
            for p in range(u.shape[0]):
                u[p,:,_k+2:] = u[p,:,_k+1].reshape(-1,1)
                uG[p,:,_k+2:] = uG[p,:,_k+1].reshape(-1,1)
                uF[p,:,_k+1:] = uF[p,:,_k].reshape(-1,1)
            
            _loop_range = range(_k+1, N)
            if I == N:
                raise Exception('System has already converged')
        else:
            _loop_range = range(N)
        
        #Step 2: integrate using F (fine solver) in parallel with the current best initial
        # values
        for k in _loop_range:
            # if k == 0:
            #     print(f'{model.name} iteration number (out of {N}): {k+1} ', end='')
            # else:
            #     print(k+1, end=' ')
            if verbose == 'v':
                print(f'{self.ode_name} {model.name} iteration number (out of {N}): {k+1} ')
                
            s_time = time.time()
            if parall == 'mpi':
                ins = [(t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F) for i in range(I,N)]
                out = list(pool.map(RK_last_t, ins, repeat(self.RK_thresh)))
                _temp_uFs = np.array([i[0] for i in out])
                uF[I+1:N+1,:,k] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
                del _temp_uFs
            elif parall == 'joblib':
                out = Parallel(-1)(delayed(lambda i: RK_last_t((t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F), self.RK_thresh))(i) for i in range(I,N))
                _temp_uFs = np.array([i[0] for i in out])
                uF[I+1:N+1,:,k] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
            else:
                temp_t = 0
                for i in range(I, N):
                    temp = RK_last_t((t[i], t_shift[i], int(Nf/N)+1, u[i,:,k], f, F), self.RK_thresh)
                    uF[i+1,:,k] = temp[0]
                    temp_t =+ temp[1]
                F_time_serial += temp_t/(N-I)
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
                if verbose == 'v':
                    print('WARNING: early stopping')
                err[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
                err[-1,k] = np.nextafter(epsilon, 0)
                break
            
            if lag_k is None:
                model.fit_timed(x, D, k=k, data_x=data_x, data_y=data_D)
            else:
                tr_x = np.moveaxis(data_x[I:,:,max(k+1-lag_k, 0):k+1], 1, -1).reshape(-1, n)
                tr_y = np.moveaxis(data_D[I:,:,max(k+1-lag_k, 0):k+1], 1, -1).reshape(-1, n)
                model.fit_timed(tr_x, tr_y, k=k)
                if verbose == 'vv':
                    print('k', k, 'train shape', tr_x.shape, tr_y.shape)
            
            if debug:
                preds_t = np.empty((N-I, n))
                truth_t = np.empty((N-I, n))
                preds_t.fill(np.nan)
                truth_t.fill(np.nan)
                
            for i in range(I, N):
                # run G solver on best initial value
                temp, temp_t = RK_last_t((t[i], t[i+1], int(Ng/N)+1, u[i,:,k+1], f, G), self.RK_thresh)
                G_time += temp_t
                uG[i+1,:,k+1] = temp
                del temp, temp_t
                
                if not debug:
                    preds = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                                               uF[i+1,:,k], uG[i+1,:,k], i=i)
                
                if debug:
                    temp = RK_last((t[i], t[i+1], int(Nf/N)+1, u[i,:,k+1], f, F), self.RK_thresh)
                    opt_pred = temp
                    del temp
                    truth_t[i-I,:] =  opt_pred - uG[i+1,:,k+1] 
                    preds = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                                               uF[i+1,:,k], uG[i+1,:,k], i=i, truth=opt_pred - uG[i+1,:,k+1])
                    preds_t[i-I,:] = preds
                    
                
                # do predictor-corrector update
                # u[i+1,:,k+1] = uF[i+1,:,k] + uG[i+1,:,k+1] - uG[i+1,:,k]
                u[i+1,:,k+1] = preds + uG[i+1,:,k+1]
                
            
            # print(uG[:20,0, 3])
            
            if debug:
                pred_err = np.abs(truth_t - preds_t)
                mean_errs.append(np.mean(pred_err,0))
                max_errs.append(np.max(pred_err,0))
                if verbose == 'v':
                    print(f'Avg error {np.mean(pred_err,0)}, Max. error {np.max(pred_err,0)}')
                all_pred_err.append(pred_err)
            # error catch
            a = 0
            if np.any(np.isnan(uG[:,:, k+1])):
                raise Exception("NaN values in initial coarse solve - increase Ng!")
                           
            # Step 4: Converence check
            # checks whether difference between solutions at successive iterations
            # are small, if so then that time slice is considered converged.               
            err[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
            err[I,k] = 0
            
            if debug:
                one_step_error.append([err[I+1,k], pred_err.max()])
            
            II = I;
            for p in range(II+1, N+1):
                if err[p, k] < epsilon:
                    u[p,:,k+2:] = u[p,:,k+1].reshape(-1,1)
                    uG[p,:,k+2:] = uG[p,:,k+1].reshape(-1,1)
                    uF[p,:,k+1:] = uF[p,:,k].reshape(-1,1)
                    I = I + 1
                else:
                    break
            if verbose == 'v':    
                print('--> Converged:', I)
            conv_int.append(I)
            if store_int:
                name_base = f'{self.ode_name}_{self.N}_{model.name}_int'
                int_dir = kwargs.get('int_dir', '')
                name_base = kwargs.get('int_name', name_base)
                int_name = f'{name_base}_{k}'
                _objs = {'L':L, 'L_sub':L_sub, 'dT':dT, 'dt':dt, 't':t, 't_shift':t_shift, 'I':I, 'verbose':verbose,
                     'u':u[...,:k+2], 'uG':uG[...,:k+2], 'uF':uF[...,:k+2], 'err':err[...,:k+2], 'x':x, 'D':D, 
                     'data_x':data_x[...,:k+2], 'data_D':data_D[...,:k+2], 'G_time':G_time, 'F_time':F_time,
                     'debug':debug, 'early_stop':early_stop, 'lag_k':lag_k, 'parall':parall, 'store_int':store_int, 'kwargs':kwargs,
                     'k':k, 'conv_int': conv_int}
                
                self.store(path=os.path.join(int_dir, name_base), name=int_name, mdl=model, objs=_objs)
                
                
            if I == N:
                break
            if (early_stop is not None) and k == (early_stop-1):
                if verbose == 'v':
                    print('Early stopping due to user condition.')
                break
        
        debug_dict = {}
        
        if debug:
            # plot prediction errors per iteration
            # mean_errs = np.array(mean_errs)
            # max_errs = np.array(max_errs)
            # fig,ax = plt.subplots(2,1)
            # for i in range(n):
            #     ax[0].plot(np.log10(mean_errs[:,i]), label=f'Coord {i}')
            #     ax[1].plot(np.log10(max_errs[:,i]), label=f'Coord {i}')
            # ax[0].set_title('Avg. error per iteration (log)')
            # ax[1].set_title('Max. error per iteration (log)')
            # fig.suptitle(self.ode_name)
            # fig.tight_layout()
            
            one_step_error = np.array(one_step_error)
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, one_step_error[:,0].shape[0]+1), np.log10(one_step_error[:,0]), label='$U_{k-1}-U_k$')
            ax.plot(np.arange(1, one_step_error[:,0].shape[0]+1), np.log10(one_step_error[:,1]), label='$(F-G)$ err')
            ax.set_title('Model vs Convergence error')
            ax.legend()
            debug_dict['one_step_error'] = one_step_error
            debug_dict['all_pred_err'] = all_pred_err
            
            
        timings = {'F_time':F_time, 'G_time': G_time, 'F_time_serial_avg': F_time_serial}
        timings.update(model.get_times())
        
        return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 'data_x':data_x[...,:k+1], 
                'data_D':data_D[...,:k+1], 'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
                'conv_int':conv_int}

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
    
    def plot(self, skip = [], add_name=True, add_title=''):
        runs, tspan, Nf, u0 = self.runs, self.tspan, self.Nf, self.u0
        f, F, epsilon = self.f, self.F, self.epsilon
        
        if len(add_title) != 0:
            add_title = add_title + ' - '
        
        if self.fine is None:
            fine, fine_t = RK_t(np.linspace(tspan[0], tspan[-1], num=Nf+1), u0, f, F)
            self.fine, self.fine_t = fine, fine_t
        else:
            fine = self.fine
        
        plot_data = {key : self._build_plot_data(**runs[key]) for key in runs}
        
        if 0 not in skip:
            fig, ax = plt.subplots(u0.shape[0],1)
            x_plot = np.linspace(tspan[0], tspan[-1], num=Nf+1)
            for i in range(u0.shape[0]):
                ax[i].plot(x_plot, fine[:,i], linewidth=0.5, label='Fine')
                for mdl_name in plot_data:
                    line2d, = ax[i].plot(x_plot, plot_data[mdl_name]['u_cont'][:,i], 
                                         linewidth=0.5, label=mdl_name)
                    ax[i].scatter(plot_data[mdl_name]['t'], plot_data[mdl_name]['u_int'][:,i,-1], 
                                  marker='x', s=2, color=line2d.get_color())
                ax[i].set_ylabel(f'$u_{{{i+1}}}(t)$')
            ax[i].legend()
            ax[i].set_xlabel('$t$')
            if add_name:
                fig.suptitle(f'{self.ode_name} - {add_title}Comparison of trajectories')
            else:
                fig.suptitle('Comparison of trajectories')
            fig.tight_layout()
        
        if 1 not in skip:
            fig, ax = plt.subplots(u0.shape[0],1)
            x_plot = np.linspace(tspan[0], tspan[-1], num=Nf+1)
            for i in range(u0.shape[0]):
                for mdl_name in plot_data:
                    y_plot = np.log10(np.abs(fine - plot_data[mdl_name]['u_cont']))
                    ax[i].plot(x_plot, y_plot[:,i], linewidth=0.5, label=mdl_name)
                ax[i].set_ylabel(f'$u_{{{i+1}}}$ log error')
                ax[i].axhline(np.log10(epsilon), linestyle='dashed', color='gray', linewidth=1, label='Tolerance')
            ax[i].legend()
            ax[i].set_xlabel('$t$')
            if add_name:
                fig.suptitle(f'{self.ode_name} - {add_title}Algorithm error wrt fine solver')
            else:
                fig.suptitle('Algorithm error wrt fine solver')
            fig.tight_layout()
        
        if 2 not in skip:
            fig, ax = plt.subplots()
            for mdl_name in plot_data:
                err = plot_data[mdl_name]['err']
                x_plot = np.arange(1, err.shape[-1]+1)
                y_plot = np.log10(np.max(err, axis=0))
                line2d, = ax.plot(x_plot, y_plot, linewidth=0.5, label=mdl_name)
                ax.scatter(x_plot, y_plot, s=1, color=line2d.get_color())
            ax.set_ylabel('Max. absolute error (log)')
            ax.axhline(np.log10(epsilon), linestyle='dashed', color='gray', linewidth=1, label='Tolerance')
            ax.legend()
            ax.set_xlabel('$k$')
            if add_name:
                fig.suptitle(f'{self.ode_name} - {add_title}Max. abs. error over parareal iterations')
            else:
                fig.suptitle('Max. abs. error over parareal iterations')
            fig.tight_layout()
            
        if 3 not in skip:
            cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            styles = ['solid', 'dotted', 'dashed', 'dashdot']
            fig, ax = plt.subplot_mosaic('AAA.;BBCC', constrained_layout=True)
            cycl = cycler(linestyle=styles, lw=[0.5, 1, 1, 1]) * cycler('color', cols)
            ax['A'].set_prop_cycle(cycl)
            ax['B'].set_prop_cycle(cycl)
            ax['C'].set_prop_cycle(cycl)
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
                line2d, = ax['B'].plot(x_plot, y_plot, label=mdl_name)
                ax['B'].scatter(x_plot, y_plot, s=1, color=line2d.get_color())
                line2d1, = ax['A'].plot(x_plot, np.cumsum(y_plot),  label=mdl_name[:18])
                ax['A'].scatter(x_plot, np.cumsum(y_plot), s=1, color=line2d1.get_color())
                line2d2, = ax['C'].plot(x_plot, np.log10(one_step_err), label=mdl_name)
                ax['C'].scatter(x_plot, np.log10(one_step_err), s=1, color=line2d2.get_color())

            ax['B'].set_title('# Converged Intervals per iteration')
            ax['C'].set_title('Error on 1st interval')

            ax['A'].axhline(err.shape[0]-1, linestyle='dashed', color='gray', linewidth=1)
            ax['C'].axhline(np.log10(epsilon), linestyle='dashed', color='gray', linewidth=1)
            leg = ax['A'].legend(loc='upper left', bbox_to_anchor= (1, 1), fontsize='small')
            leg.set_in_layout(False)
            ax['B'].set_xlabel('$k$')
            ax['C'].set_xlabel('$k$')
            if add_name:
                ax['A'].set_title(f'{self.ode_name} - {add_title}# Converged Intervals')
            else:
                ax['A'].set_title(f'# Converged Intervals')
    
    def print_times(self, mdl_speedup=None, expected_fine=None):
        if mdl_speedup is None:
            if self.fine is None:
                fine, fine_t = RK_t(np.linspace(self.tspan[0], self.tspan[-1], num=self.Nf+1), self.u0, self.f, self.F)
                self.fine, self.fine_t = fine, fine_t
            else:
                fine, fine_t = self.fine, self.fine_t
            
        if mdl_speedup is None:
            mdl_speedup = False
        elif mdl_speedup in self.runs:
            s_ref = self.runs[mdl_speedup]['timings']['mdl_tot_t']
            mdl_speedup = True
        else:
            mdl_speedup = False
            
        cols = ['Model', 'K', 'G','F','Train','Pred','Mdl Tot', 'Overall', 'Speedup']
        if mdl_speedup:
            cols[-1] = 'Mdl Speedup'
        str_format = lambda x: f'{x:.2e}'
        max_col_len = []
        max_col_len.append(max(len(cols[0]), 4, max(map(len, self.runs.keys()))))
        max_col_len.append(max(map(lambda x: len(str(x)), [v['k'] for k,v in self.runs.items()])))
        _attrs = ['G_time', 'F_time', 'mdl_train_t', 'mdl_pred_t', 'mdl_tot_t', 'runtime']
        max_col_len.extend([max(map(lambda x: len(str_format(x)), [v['timings'][k] for _,v in self.runs.items()])) for k in _attrs])
        max_col_len.append(len(cols[-1]))
        
        if expected_fine is not None:
            expected_speedup = True
            cols.append('E[Speedup]')
            max_col_len.append(len(cols[-1]))
        else:
            expected_speedup = False
        res = []
        res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(cols)])+'|')
        res.append('|'+'|'.join([f'{"-"*max_col_len[i]}' for i in range(len(cols))])+'|')
        if mdl_speedup:
            res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(['Fine','-','-','-','-','-','-', '-','-'])])+'|')
        else:
            res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(['Fine','-','-','-','-','-','-', str_format(self.fine_t),1])])+'|')
        if expected_speedup:
            res[-1] = res[-1] + f'{1:^{max_col_len[-1]}}|'
        
        for mdl_name,v in self.runs.items():
            temp = []
            temp.append(f'{mdl_name:^{max_col_len[0]}}')
            temp.append(f'{v["k"]:^{max_col_len[1]}}')
            temp.append('|'.join([f'{str_format(v["timings"][k]):^{max_col_len[i+2]}}'for i,k in enumerate(_attrs)]))
            if mdl_speedup:
                temp.append(f'{s_ref/v["timings"]["mdl_tot_t"]:^{max_col_len[8]}.2f}')
            else:
                temp.append(f'{self.fine_t/v["timings"]["runtime"]:^{max_col_len[8]}.2f}')
            if expected_speedup:
                exp_cost = (expected_fine/self.N*v['k']) + v["timings"]["mdl_tot_t"]
                temp.append(f'{expected_fine/exp_cost:^{max_col_len[-1]}.2f}')
            res.append('|'+'|'.join(temp)+'|')
        print('\n'.join(res))
        return '\n'.join(res)
        
    
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
            F = '$\\f$'
            G = '$\\g$'
        
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
            
        
        
    
    def plot_all_err(self, key):
        if key not in self.runs:
            return None
        if len(self.runs[key]['debug_dict']) == 0:
            return None
        
        for idx, pred_err in enumerate(self.runs[key]['debug_dict']['all_pred_err']):
            fig,ax = plt.subplots()
            ax.plot(np.max(np.log10(pred_err), axis=1), label='true err comp')
            l = self.runs[key]['err'][:, idx]
            start = (l != 0).argmax()
            ax.plot(np.log10(l[start:]), label='conv err')
            ax.axhline(-6,ls='dashed', lw=1, color='gray')
            ax.axhline(-8,ls='dashed', lw=1, color='black')
            ax.axhline(-10,ls='dashed', lw=1, color='gray')
            ax.set_title(idx+1)
            ax.legend()
        
    
    def print_times(self, mdl_speedup=None, expected_fine=None):
        if mdl_speedup is None:
            if self.fine is None:
                fine, fine_t = RK_t(np.linspace(self.tspan[0], self.tspan[-1], num=self.Nf+1), self.u0, self.f, self.F)
                self.fine, self.fine_t = fine, fine_t
            else:
                fine, fine_t = self.fine, self.fine_t
            
        if mdl_speedup is None:
            mdl_speedup = False
        elif mdl_speedup in self.runs:
            s_ref = self.runs[mdl_speedup]['timings']['mdl_tot_t']
            mdl_speedup = True
        else:
            mdl_speedup = False
            
        cols = ['Model', 'K', 'G','F','Mdl Tot', 'Overall', 'Speedup']
        if mdl_speedup:
            cols[-1] = 'Mdl Speedup'
        str_format = lambda x: f'{x:.2e}'
        max_col_len = []
        max_col_len.append(max(len(cols[0]), 4, max(map(len, self.runs.keys()))))
        max_col_len.append(max(map(lambda x: len(str(x)), [v['k'] for k,v in self.runs.items()])))
        _attrs = ['G_time', 'F_time', 'mdl_tot_t', 'runtime']
        max_col_len.extend([max(map(lambda x: len(str_format(x)), [v['timings'][k] for _,v in self.runs.items()])) for k in _attrs])
        max_col_len.append(len(cols[-1]))
        
        if expected_fine is not None:
            expected_speedup = True
            cols.append('E[Speedup]')
            max_col_len.append(len(cols[-1]))
        else:
            expected_speedup = False
        res = []
        res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(cols)])+'|')
        res.append('|'+'|'.join([f'{"-"*max_col_len[i]}' for i in range(len(cols))])+'|')
        if mdl_speedup:
            res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(['Fine','-','-','-','-', '-','-'])])+'|')
        else:
            res.append('|'+'|'.join([f'{x:^{max_col_len[i]}}' for i,x in enumerate(['Fine','-','-','-','-', str_format(self.fine_t),1])])+'|')
        if expected_speedup:
            res[-1] = res[-1] + f'{1:^{max_col_len[-1]}}|'
        
        for mdl_name,v in self.runs.items():
            temp = []
            temp.append(f'{mdl_name:^{max_col_len[0]}}')
            temp.append(f'{v["k"]:^{max_col_len[1]}}')
            temp.append('|'.join([f'{str_format(v["timings"][k]):^{max_col_len[i+2]}}'for i,k in enumerate(_attrs)]))
            if mdl_speedup:
                temp.append(f'{s_ref/v["timings"]["mdl_tot_t"]:^{max_col_len[6]}.2f}')
            else:
                temp.append(f'{self.fine_t/v["timings"]["runtime"]:^{max_col_len[6]}.2f}')
            if expected_speedup:
                exp_cost = (expected_fine/self.N*v['k']) + v["timings"]["mdl_tot_t"]
                temp.append(f'{expected_fine/exp_cost:^{max_col_len[-1]}.2f}')
            res.append('|'+'|'.join(temp)+'|')
        print('\n'.join(res))
        return '\n'.join(res)
    
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
    
  