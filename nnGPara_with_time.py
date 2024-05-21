from globals import *

############ All NN-GParareal with time results ############

class ParaMod(Parareal):
    pass
        
#################### Heuristics simulations #########################

#%% Code to run the simulations 

# prevents accidental runs
if False:

    #%%% Code
    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product, repeat
    from new_lib import *

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)

    class NNGP_alt(NNGP_p):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = 'NNGP'+str(kwargs['nntype'])
            self.nntype = kwargs['nntype']
            self.rng2 = np.random.default_rng(self.seed)
            self.show_mtx = kwargs.get('show_mtx', False)
            
        def fit(self, x, y, k, *args, **kwargs):
            self.k = k
            self.x, self.y = x, y
            self.data_x = kwargs['data_x']
            self.data_y = kwargs['data_y']
        
        def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
            self.i = kwargs['i']
            if self.nn == 'adaptive':
                nn = max(10, self.k + 2)
            else:
                nn = self.nn
            data_x, data_y = self.data_x, self.data_y
                
            if self.nntype == 'nn':
                s_idx = np.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
                xm = self.x[s_idx[:nn], :]
                ym = self.y[s_idx[:nn], :]
                
            elif self.nntype == 'col+rnd':
                on_col = min(nn, self.k+1)
                on_near = nn - on_col
                
                x_col = self.data_x[self.i, :, (self.k+1)-on_col:self.k+1 ].T
                y_col = self.data_y[self.i, :, (self.k+1)-on_col:self.k+1 ].T
                
                idx_rem = list(map(lambda xx: np.argmax(np.any(self.x == xx.reshape(1,-1), axis=1)), x_col))
                s_idx = self.rng2.permutation(np.arange(self.x.shape[0])) # random
                cands = s_idx[:nn]
                cands_mask = np.array(list(map(lambda x: x not in idx_rem, cands)))
                x_near = self.x[cands[cands_mask][:on_near],:]
                y_near = self.y[cands[cands_mask][:on_near],:]
                xm = np.vstack([x_col, x_near])
                ym = np.vstack([y_col, y_near])
                assert xm.shape[0] == nn
                
            elif self.nntype == 'col_only':
                xm = self.data_x[self.i, :, 0:self.k+1 ].T
                ym = self.data_y[self.i, :, 0:self.k+1 ].T
                
            elif self.nntype == 'row_col':
                # expand radially
                def mygen(mtx, idx_row, idx_col):
                    for row, col in zip(idx_row, idx_col):
                        if np.any(np.isnan(mtx[row, :, col])):
                            continue
                        yield mtx[row, :, col]
                nn = self.nn
                data_x = data_x[:,:,:self.k+1]
                data_y = data_y[:,:,:self.k+1]
                iters = np.arange(data_x.shape[0]).reshape(-1,1,1) + np.zeros(data_x.shape[2]).reshape(1,1,-1)
                intrvl = np.arange(data_x.shape[2]).reshape(1,1,-1) + np.zeros(data_x.shape[0]).reshape(-1,1,1)
                flat_idxs = np.argsort(np.squeeze((np.abs(intrvl - self.k) + np.abs(iters - self.i))), axis=None)
                idx_row, idx_col = np.array(flat_idxs/data_x.shape[-1], dtype=int), flat_idxs%data_x.shape[-1]

                my_gen = mygen(data_x, idx_row, idx_col)
                xm = np.array([next(my_gen) for i in range(nn)])
                my_gen = mygen(data_y, idx_row, idx_col)
                ym = np.array([next(my_gen) for i in range(nn)])
                
            elif self.nntype == 'row':
                def cstm_iter(mtx, i, j):
                    def my_cycler(a,b):
                        exc1 = False
                        c = 0
                        while True:
                            try:
                                yield next(a)
                            except StopIteration:
                                exc1=True
                            
                            try:
                                yield next(b)
                            except StopIteration:
                                if exc1:
                                    break
                            if c == 100:
                                print('aaaarg')
                                break
                            c += 1
                                
                    for row in range(i, -1, -1):
                        left_col_gen = (i for i in range(j, -1,-1))
                        right_col_gen = (i for i in range(j+1, mtx.shape[0]))
                        for col in my_cycler(left_col_gen, right_col_gen):
                            if np.any(np.isnan(mtx[col, :, row])):
                                continue
                            yield mtx[col, :, row]
                            
                data_x = data_x[:,:,:self.k+1]
                data_y = data_y[:,:,:self.k+1]
                nn = self.nn
                my_gen = cstm_iter(data_x, self.k, self.i)
                xm = np.array([next(my_gen) for i in range(nn)])
                
                my_gen = cstm_iter(data_y, self.k, self.i)
                ym = np.array([next(my_gen) for i in range(nn)])
                
            elif self.nntype == 'col_full':
                def cstm_iter(mtx, i, j):
                    def my_cycler(a,b):
                        exc1 = False
                        c = 0
                        while True:
                            try:
                                yield next(a)
                            except StopIteration:
                                exc1=True
                            
                            try:
                                yield next(b)
                            except StopIteration:
                                if exc1:
                                    break
                            if c == 100:
                                print('aaaarg')
                                break
                            c += 1
                                
                    
                    left_col_gen = (i for i in range(j, -1,-1))
                    right_col_gen = (i for i in range(j+1, mtx.shape[0]))
                    for col in my_cycler(left_col_gen, right_col_gen):
                        for row in range(i, -1, -1):
                            if np.any(np.isnan(mtx[col, :, row])):
                                continue
                            yield mtx[col, :, row]
                            
                data_x = data_x[:,:,:self.k+1]
                data_y = data_y[:,:,:self.k+1]
                nn = self.nn
                my_gen = cstm_iter(data_x, self.k, self.i)
                xm = np.array([next(my_gen) for i in range(nn)])
                
                my_gen = cstm_iter(data_y, self.k, self.i)
                ym = np.array([next(my_gen) for i in range(nn)])
                
            if self.show_mtx:
                if self.i % 3 == 0 and self.k> 1:
                    show_mtx(self.data_x, new_x, self.k, self.i, nn=nn, other_points=xm)
                    
                    # with np.printoptions(edgeitems=50, linewidth=300):
                    #     print(np.array(np.squeeze((np.abs(intrvl - self.k) + np.abs(iters - self.i))), int))
                    # raise
            
            n = self.n
            preds = self.get_preds(xm, ym, n, new_x)
            
            return preds
        
        
    class ParaMod(Parareal):
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
                mdl = NNGP_alt(n=self.n, N=self.N, worker_pool=kwargs['pool'], **kwargs)
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
        
        

    def do(mdl, nn, epsilon):
        solver = ParaMod(ode_name=mdl, epsilon=epsilon)
        types = ['nn', 'col+rnd', 'col_only', 'row_col', 'row', 'col_full']
        for t in types:
            try:
                res = solver.run(model='nngp', nntype=t, show_mtx=False, nn=16)
            except Exception as e:
                solver.runs['NNGP'+t] = {'error':str(e)}
        solver.data_tr = None
        solver.data_tr_inv = None
        return [solver, mdl, nn, epsilon]

    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 2
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        
        mdls = ['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n','dbl_pend_n']
        nns = [16]*6 + [13, 13, 12, 12, 13, 14]
        epsilons = [5e-7]*6 
        
        res = list(p.map(do, mdls, nns, epsilons))
        
        with open(os.path.join('nngptime_diff_subsets2'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

#%%% Analysis

res = read_pickle('nngptime_diff_subsets2')

# we can read data off the column K
# for solver, mdl, nn, epsilon in res:
#     print(f'mdl: {mdl}, epsilon: {epsilon}')
#     solver.print_times()
    
def build(data, eps=5e-7, do_plots=False):
    d = {'fhn_n':'FHN', 'lorenz_n':'Lorenz', 'rossler_long_n':'Rossler', 'non_aut32_n':'Hopf',
          'brus_2d_n':'Brusselator','dbl_pend_n':'Double Pendulum'}
    sep = ' & '
    end = ' \\\\'# '\\\\'
    start = ' ' #''
    res = [start+'System'+sep+sep.join(list(data[0][0].runs.keys()))+end]
    
    # res.append('|-'*(len(data[0][0].runs.keys())+1) + '|')
    res.append(r'\hline\\')
    for solver, system, nn, epsilon in data:
        if epsilon != eps:
            continue
        l = [d[system]]
        for mdl in solver.runs.keys():
            if 'error' in solver.runs[mdl]:
                print(solver.runs[mdl]['error'])
                l.append('Exc')
            else:
                l.append(str(solver.runs[mdl]['k']))
        l_str = f' {sep} '.join(l)
        res.append(start+l_str+end)
        if do_plots:
            solver.plot(cstm_title=d[system])
    print('\n'.join(res))
    
def find_model(data, mdl):
    for i in data:
        solver, system, nn, K = i
        if system == mdl:
            return i
        
build(res, eps=5e-7, do_plots=False)



##################### Kernel Learning #########################

#%% Code to run the simulations 

# prevents accidental runs
if False:

    #%%% Code Nelder-Mead
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

    import sys
    sys.path.append('/mnt/c/Users/u2133517/OneDrive - University of Warwick/lyudmila_project/massi/python')

    from new_lib import *

    import pickle

    def store_pickle(obj, name):
        with open(os.path.join('/mnt/c/Users/u2133517/OneDrive - University of Warwick/lyudmila_project/massi/python', name), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
    def read_pickle(name):
        with open(os.path.join('/mnt/c/Users/u2133517/OneDrive - University of Warwick/lyudmila_project/massi/python', name), 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
        return data

    def show_mtx(data_x, new_x, k, i, nn=0, other_points=None):
        
        dist_m = np.sqrt(np.mean((data_x[..., :k+1] - new_x.reshape(1,-1,1))**2, axis=(1)))
        
        ddata = np.log10(dist_m.T)                
        do = lambda x: 'n' if np.isnan(x) else str(int(x))
        fig, ax = plt.subplots(figsize=(16, 40))
        ax.matshow(ddata)
        
        # add white box to identify prediction interval
        for ii in range(ddata.shape[0]):
            z = ddata[ii, i]
            ax.text(i, ii, do(z), ha='center', va='center', fontsize='small', c='white',
                    bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
            
        # red color nearest neighbors
        if isinstance(nn, list) or isinstance(nn, np.ndarray):
            for idx, (ii, jj) in enumerate(map(lambda xx: np.unravel_index(np.equal(data_x, nn[[xx],:, np.newaxis])[:,0,:].argmax(), 
                                                        (data_x.shape[0], data_x.shape[-1])), range(len(nn)))):
                z = ddata[jj, ii]
                if ii == i:
                    ax.text(ii, jj, do(z), ha='center', va='center', fontsize='small', c='red', fontweight='black',
                            bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
                else:
                    ax.text(ii, jj, do(z), ha='center', va='center', fontsize='small', c='red', fontweight='black')
        elif nn != 0:
            dist_m_f = dist_m.ravel()
            closest_idxs = dist_m_f.argsort(axis=None)[:nn]
            for ii, jj in zip(*np.unravel_index(closest_idxs, ddata.shape[::-1])):
                z = ddata[jj, ii]
                if ii == i:
                    ax.text(ii, jj, do(z), ha='center', va='center', fontsize='small', c='red', fontweight='black',
                            bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))
                else:
                    ax.text(ii, jj, do(z), ha='center', va='center', fontsize='small', c='red', fontweight='black')
            
        # Add more custom points, text color white with black bounding box
        if other_points is not None:
            for idx, (ii, jj) in enumerate(map(lambda xx: np.unravel_index(np.equal(data_x, other_points[[xx],:, np.newaxis])[:,0,:].argmax(), 
                                                        (data_x.shape[0], data_x.shape[-1])), range(len(other_points)))):
                z = ddata[jj, ii]    
                ax.text(ii, jj, do(z), ha='center', va='center', fontsize='small', c='white', fontweight='black',
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
        plt.pause(0.01)

    class MyExc(Exception):
        def __init__(self,*args, **kwargs):
            self.args = args
            self.d = kwargs

    def _fit_gp_jit(x, y, theta, kernel, jitter):
        N = x.shape[0]
        K = kernel(x, x, theta) +  jnp.eye(N)*10**jitter
        L = jnp.linalg.cholesky(K)
        alph = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, y, lower=True), lower=False)
        return L, alph
    _fit_gp_jit = jax.jit(_fit_gp_jit, static_argnums=(3,))

    class NNGP_p(ModelAbstr):
        
        def __init__(self, n, N, worker_pool, theta=None, fatol=None, xatol=None, **kwargs):
            super().__init__(**kwargs)
            # theta = np.ones(n) if theta is None else theta
            if theta is None:
                theta=[1,1]
            self.theta = np.array(theta)
            self.name = 'NNGPtime'
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
            
            self.start_rnd = kwargs.get('start_rnd', False)
            self.nn_iters = kwargs.get('nn_iters', 5)
            self.get_best = kwargs.get('get_best', False)
            self.data_store = {'full_data': {}, 'nn': {}}
            
            
        @staticmethod
        def k_gauss_mod(xi, xj, params):
            sigma_x, sigma_y, sigma_intrvl, sigma_iters = params 
            dims = xi.shape[0]
            ti = xi[(dims-2):]
            tj = xj[(dims-2):]
            xi = xi[:(dims-2)]
            xj = xj[:(dims-2)]
            space = 10**(sigma_y) * jnp.exp(-0.5 * (1/(10**sigma_x)) * jnp.sum((xi-xj)**2))
            intrvl = jnp.exp(-0.5 * (1/(10**sigma_intrvl)) * (ti[0]-tj[0])**2)
            iters = jnp.exp(-0.5 * (1/(10**sigma_iters)) * (ti[1]-tj[1])**2)
            return space * intrvl * iters

        def kernel_jit(x, y, kernel_params):
            kernel = NNGP_p.k_gauss_mod
            map_x = jax.vmap(kernel, in_axes=(None, 0, None)) 
            map_y = jax.vmap(map_x, in_axes=(0, None, None))
            return map_y(x, y, kernel_params)
        kernel_jit = staticmethod(jax.jit(kernel_jit))
        
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
        
        def fit(self, x, y, k, *args, **kwargs):
            self.k = k
            self.x, self.y = x, y
            
            data_x = kwargs['data_x'][...,:k+1]
            data_y = kwargs['data_y'][...,:k+1]
            self.data_x = data_x
            self.data_y = data_y
            
            data_x = self.data_x
            data_y = self.data_y
            
            
            def gen_norm(x):
                def nrm(z):
                    z = np.array(z)
                    if x.min() == x.max():
                        assert len(np.unique(x)) == 1
                        temp = z.copy()
                        temp.fill(1)
                        return temp
                    else:
                        return 2*(z-x.min())/(x.max()-x.min()) - 1
                        
                return nrm
            
            # enrich dataset adding times
            
            intrvl = np.arange(data_x.shape[0]).reshape(-1,1,1) + np.zeros(data_x.shape[2]).reshape(1,1,-1)
            iters = np.arange(data_x.shape[2]).reshape(1,1,-1) + np.zeros(data_x.shape[0]).reshape(-1,1,1)
            nrm_int = gen_norm(intrvl)
            intrvl = nrm_int(intrvl)
            nrm_iter = gen_norm(iters)
            iters = nrm_iter(iters)
            self.nrm_iter = nrm_iter
            self.nrm_int = nrm_int
            self._intrvl = intrvl
            self._iters = iters
            data_x = np.concatenate([data_x, intrvl, iters], axis=1)
            xt = np.moveaxis(data_x, 2, 1).reshape(-1, self.n+2)
            yt = np.moveaxis(data_y, 2, 1).reshape(-1, self.n)
            nan_mask = np.logical_not(np.isnan(xt[:,0]))
            self.xt = xt[nan_mask,:]
            self.yt = yt[nan_mask,:]
            
            self.data_store['full_data'][k] = (self.xt, self.yt, self.data_x, self.data_y)
            
        def _predict(x, y, theta, kernel, jitter, new_x):
            N = x.shape[0]
            L, alph = _fit_gp_jit(x, y, theta, kernel, jitter)
            K_star = kernel(x, new_x, theta)
            post_mean = K_star.T @ alph
            return post_mean
        _predict = staticmethod(jax.jit(_predict, static_argnums=(3,)))
        
        
        def get_preds(self, xm, ym, n, new_x, targj=None):
            jitter = np.arange(-20, -11, dtype=float)
            restarts = range(self.n_restarts+1)
            n=1
            mdls = [targj]
            n_pars = self.theta.shape[0]
            ins = list(product(mdls, jitter, restarts))
            static_ins = (xm, ym, self.fatol, self.xatol)
            rnd = [self.rng.integers(-8, 0, (n_pars)) for i in range(self.n_restarts*n*len(jitter))]+[[1]*n_pars for i in range(n*len(jitter))]
            assert len(rnd) == len(ins)
            out_res = list(self.pool.map(self._get_opt_par, repeat(static_ins), ins, rnd))
            
            j = targj
            res = [i for i in out_res if i[-1] == j]
            res = np.array(res)
            mask = res[:,(n_pars)] < res[:,(n_pars)].min()*0.99
            if mask.sum()==0:
                mask[:] = True
            *opt_params, opt_fval, opt_jitter,_ = min(res[mask,:], key=lambda x: x[n_pars])
            return opt_params, opt_jitter, opt_fval
        
        
        def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
            self.i = kwargs['i']
            xt, yt = self.xt, self.yt
            print(self.i, self.k)
            
            if self.nn == 'adaptive':
                nn = max(10, self.k + 2)
            else:
                nn = self.nn
                
            # enrich new_x
            new_x = np.r_[new_x[0,:], self.nrm_int(self.i), self.nrm_iter(self.k)].reshape(1,-1)
            assert new_x.shape[0] == 1
            
            truth = kwargs['truth']


            n = self.n
            preds = np.empty(n)
            
            stt = time.time()
            for j in range(n):
                res = []
                counter = 0
                self.data_store['nn'][(self.k, self.i, j)] = {}
                for _p in range(10):
                    for _l in range(self.nn_iters):
                        if _l == 0:
                            s_idx = self.rng.permutation(xt.shape[0])
                            xm = xt[s_idx[:nn], :]
                            ym = self.yt[s_idx[:nn], :]
                        else:
                            s_idx = np.argsort(self.kernel(xt, new_x, opt_params)[:,0])[::-1]
                            xm = xt[s_idx[:nn], :]
                            ym = self.yt[s_idx[:nn], :]
                            
                        opt_params, opt_jitter, opt_fval = self.get_preds(xm, ym, n, new_x, targj=j)
                        y_mean = np.squeeze(self._predict(xm, ym[:, j], opt_params, self.kernel, opt_jitter, new_x)) 
                        error = np.abs(truth[j]-y_mean)
                        self.data_store['nn'][(self.k, self.i, j)][counter] = [xm,ym,new_x,opt_params,opt_jitter,opt_fval, truth, y_mean, error]
                        counter = counter + 1
                        
                        if j == 0:
                            if self.i % 10 == 0:
                                print(opt_params, opt_fval, opt_jitter)
                        
                        res.append((xm, ym, opt_params, opt_fval, opt_jitter))
                    
                xm, ym, opt_params, opt_fval, opt_jitter = min(res, key=lambda x: x[-2])
                if j == 0:
                    if self.i % 10 == 0:
                        show_mtx(self.data_x, new_x[:,:-2], self.k, self.i, nn=nn, other_points=xm[:,:-2])
                        print('Opt:', opt_params, opt_fval, opt_jitter)
                    
                y_mean = np.squeeze(self._predict(xm, ym[:, j], opt_params, self.kernel, opt_jitter, new_x)) #x, y, theta, kernel, jitter, new_x
                preds[j] = y_mean
                
                error = np.abs(truth[j]-y_mean)
                self.data_store['nn'][(self.k, self.i, j)]['opt'] = [xm,ym,new_x,opt_params,opt_jitter,opt_fval, truth, y_mean, error]
                
                
                if np.any(np.isnan(y_mean)):
                    raise MyExc(mdl=self,new_x=new_x,kwargs=kwargs, opt_params=opt_params, opt_jitter=opt_jitter, 
                                opt_fval=opt_fval, xm=xm, ym=ym, res=res, s_idx=s_idx, preds=preds)
            print('->>>>>> Overall', time.time() - stt)
            
            return preds
        
    class ParaMod(Parareal):
        def _run(self, model='parareal', cstm_mdl_name=None, add_model=False, **kwargs):
                
            if model.lower() == 'parareal':
                mdl = BareParareal(**kwargs)
            elif model.lower() == 'gpjax':
                if 'pool' not in kwargs:
                    raise Exception('A worker pool must be provided to run NNGP in parallel')
                mdl = GPjax_p(n=self.n, N=self.N, worker_pool=kwargs['pool'], **kwargs)
            elif model.lower() == 'nngptime':
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
            
            mdl.data_store['n_nn'] = kwargs['nn']
            store_pickle(mdl.data_store, f"NNGP_time/lorenz_nngptime_sim_w_errors")
            
            

            out['mdl'] = mdl
            if cstm_mdl_name is None:
                cstm_mdl_name = mdl.name
            self.runs[cstm_mdl_name] = out
            return out

    #%

    solver = ParaMod(ode_name='lorenz_n', global_debug=False, normalization='-11', epsilon=5e-7)
    try:
        res = solver.run(model='NNGPtime', theta=[1,1,0,0], fatol=1e-1, xatol=1e-1, nn=11, n_restarts=20, seed=45,
                        nn_iters=20, start_rnd=True, pool=14, debug=True)
    except MyExc as e:
        print('Exception occurred')
        exc = e
        get = lambda x: exc.d.get(x, None)
        get_m = lambda *x: [get(i) for i in x]
        x, y, k, i, self = get_m('x', 'y','k','i', 'mdl')
        theta, xm, ym, truth, data_x, data_y = get_m('theta', 'xm', 'ym', 'truth', 'data_x', 'data_y')
        y_mean, y_var, j, new_x, preds = get_m('y_mean', 'y_var', 'j', 'new_x', 'preds')
        xt, yt,kwargs = get_m('xt','yt','kwargs')
        
        store_pickle(exc, 'todel')


#%% Analysis 

        
res = read_pickle('lorenz_nngptime_sim_w_errors')

def find_extr_i(keys, trgt_k, trgt_j):
    i_s = []
    for key in keys:
        k,i,j = key
        if k == trgt_k and j == trgt_j:
            i_s.append(i)
    return min(i_s), max(i_s)+1


#### Last subset
l = []

j = 0
nn = res['n_nn']

# k=10
# i = 27
# [xm,ym,new_x,opt_params,opt_jitter,opt_fval, truth, y_mean, error] = res['nn'][(k, i, j)][19]

def find_perc_nn(data_x, k, new_x, xm):
    dist_m = np.sqrt(np.mean((data_x[..., :k+1] - new_x.reshape(1,-1,1))**2, axis=(1)))
    dist_m_f = dist_m.ravel()
    closest_idxs = dist_m_f.argsort(axis=None)[:nn]
    idxs = np.unravel_index(closest_idxs, dist_m.shape)
    closest_obs = data_x[..., :k+1][idxs[0],:,idxs[1]]
    return np.mean([i in closest_obs for i in xm])
    
for k in sorted(res['full_data'].keys()):
    xt, yt, data_x, data_y = res['full_data'][k]
    for i in range(*find_extr_i(res['nn'].keys(), k, j)):
        for counter in range(200):
            int = res['nn'][(k, i, j)][counter]
            if counter % 19 == 0: #we just did the last one
                [xm,ym,new_x,opt_params,opt_jitter,opt_fval, truth, y_mean, error] = res['nn'][(k, i, j)][counter]
                l.append(find_perc_nn(data_x, k, new_x[:,:-2], xm[:,:-2]))
            
            
l = np.array(l)
lbls = np.histogram(l)[1]
fig, ax = plt.subplots()
lbl_text = [f'[{lbls[idx]:.1f}, {lbls[idx+1]:.1f})' if idx < lbls.shape[0]-2 else f'[{lbls[idx]:.1f}, {lbls[idx+1]:.1f}]' for idx in range(lbls.shape[0]-1)]
lbl_text2 = [f'$<{lbls[idx+1]:.1f}$' if idx < lbls.shape[0]-2 else f'$\leq{lbls[idx+1]:.1f}$' for idx in range(lbls.shape[0]-1)]
ax.bar(lbl_text, np.histogram(l)[0]/np.histogram(l)[0].sum(), label=['A']*10)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.set_xlabel('Percentage of nearest neighbors in final subset', fontsize=8)
ax.set_ylabel('Proportion', fontsize=8)
fig.tight_layout()           

fig, ax = plt.subplots()
ax.bar(range(nn+1), np.unique(l, return_counts=True)[1]/np.unique(l, return_counts=True)[1].sum())
ax.set_xlabel('Number of nearest neighbors in final subset')
ax.set_ylabel('Proportion')
fig.tight_layout()       
store_fig(fig, 'nngp_time_lorenz_final_subset')

