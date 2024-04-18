import numpy as np
import time
import copy    

import jax 
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import scipy
from scipy.optimize import minimize
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
except:
    sklearn = None


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

    def kernel_jit_(x, y, kernel_params):
        kernel = NNGP_p.k_gauss
        map_x = jax.vmap(kernel, in_axes=(None, 0, None)) 
        map_y = jax.vmap(map_x, in_axes=(0, None, None))
        return map_y(x, y, kernel_params)
    kernel_jit = staticmethod(jax.jit(kernel_jit_))
    
    def fit(self, x, y, k, *args, **kwargs):
        self.k = k
        self.x, self.y = x, y
        
    # @staticmethod
    def _predict_(x, y, theta, kernel, jitter, new_x):
        N = y.shape[0]
        L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
        K_star = kernel(x, new_x, theta)
        post_mean = K_star.T @ alph 
        return post_mean
    _predict = staticmethod(jax.jit(_predict_, static_argnums=(3,)))
    
    
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
    

    def _log_lik_(x, y, theta, kernel, jitter):
        N = y.shape[0]
        L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
        res = -(-0.5 * y.T @ alph - jnp.sum(jnp.log(jnp.diag(L))) - (N/2)*jnp.log(2*jnp.pi))
        return res
    _log_lik = staticmethod(jax.jit(_log_lik_, static_argnums=(3,)))
    
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
        elap_t = time.time() - st_
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
        mem = self.mem
        self.mem = None
        new = super().store()
        self.mem = mem
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


class ELM_base():
    
    def __init__(self, d, seed=47, res_size=500, loss='relu', M=1, R=1, alpha=0, degree=2, m=5):
        self.d = d
        self.N = res_size
        self.rng = np.random.default_rng(seed)
        self.m = m
        
        radbas = lambda x: np.exp(-x**2)
        relu = lambda x: np.maximum(x, 0)
        tanh = lambda x: np.tanh(x)
        losses = {'radbad': radbas, 'relu':relu,'tanh':tanh}
        self.loss = losses[loss]
        self.M = M
        self.R = R
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree=degree)
        self.poly.fit(np.zeros((1, d)))
        self.degree = self.poly.n_output_features_
        
        bias, C = self._init_obj()
        self.bias, self.C = bias, C

        
    def _init_obj(self):
        N, rng = self.N, self.rng
        bias = rng.uniform(-1, 1, (N, 1))
        C = rng.uniform(-1, 1, (N, self.degree))
        return bias, C
    
    def _fit(self, x, y, bias, C):
        x = self.poly.fit_transform(x)
        X = self.loss(bias + C @ x.T) # activation
        X = X.T #first col is intercept
        mdl = Ridge(alpha=self.alpha)
        mdl.fit(X, y)
        return mdl
        

    def fit(self, x, y, k):
        self.x = x
        self.y = y
        self.k = k
        

    def predict(self, new_x):
        bias = self.M * self.R * self.bias
        bias = self.bias
        C = self.R * self.C

        s_idx = np.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
        xm = self.x[s_idx[:self.m], :]
        ym = self.y[s_idx[:self.m], :]
        
        new_X = self.poly.fit_transform(new_x)
        new_X = self.loss(bias + C @ new_X.T)
        
        mdl = self._fit(xm, ym, bias, C)
        preds = np.squeeze(mdl.predict(new_X.T))
        return preds
        
    def fit_predict(self, x, y, new_x, k):
        self.fit(x, y, k)
        return self.predict(new_x)
    
    
class ELM(ModelAbstr):
    def __init__(self, d, N, seed=47, res_size=20, loss='relu', M=1, R=1, alpha=0, degree=2, m=4, **kwargs):
        super().__init__(N=N, **kwargs)
        self.ELM = ELM_base(d=d, seed=seed, res_size=res_size, loss=loss, M=M, R=R, alpha=alpha, degree=degree, m=m)
        self.name = 'ELM'


    def fit(self, x, y, k, *args, **kwargs):
        self.ELM.fit(x, y, k)


    def predict(self, new_x, *args, **kwargs):
        return self.ELM.predict(new_x)