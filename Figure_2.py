from globals import *

#%% Prediction error of k-NN Parareal and GPara vs m-NNGP 
from solver import SolverRK
from systems import Rossler
from configs import Config
import scipy
from models import ModelAbstr, NNGP_p
from parareal import Parareal, MyPool
import numpy as np
import time
import matplotlib.pyplot as plt

class Paramod(Parareal):
    def _parareal(self, model, debug=False, early_stop=None, parall='Serial', store_int=False, _load_mdl=False, **kwargs):
        tspan, N, epsilon, n = self.tspan, self.N, self.epsilon, self.n
        f, u0 = self.f, self.u0
        solver: SolverAbstr = self.solver
                         
        t = np.linspace(tspan[0], tspan[1], num=N+1)           
        I = 0                             
            
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

        if (comp_mdls := kwargs.get('comp_mdls', None)) is not None:
            err_store_mdls = {mdl.name: dict() for mdl in comp_mdls}
            do_err_mdls = True
            err_store_mdls['para'] = dict()
        else:
            do_err_mdls = False
                
        
            
        u[0,:,:] = u0[:, np.newaxis]
        uG[0,:,:] = u[0,:,:]
        uF[0,:,:] = u[0,:,:]
        
        if debug:
            mean_errs = []
            max_errs = []
            one_step_error = []
            all_pred_err = []
        
        # Initialization: run G sequentially
        temp = u0
        for i in range(N):
            temp, temp_t = solver.run_G_timed(t[i], t[i+1], temp)
            G_time += temp_t
            uG[i+1,:,0] = temp

            

        # temp, temp_t = solver.run_G(t[0], t[-1], u0)
        # G_time += temp_t
        # uG[:,:,0] = temp[0::int(Ng/N), :]
        del temp, temp_t
        u[:,:,0] = uG[:,:,0]

        if _load_mdl:
            t, I, verbose, conv_int, _u, _uG, _uF, _err, x, D, _data_x, _data_D, G_time, F_time, _k = kwargs['_reload_objs']
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
                out = list(pool.map(solver.run_F_timed, t[I:N], t[I+1:N+1], [u[i,:,k] for i in range(I,N)]))
                _temp_uFs = np.array([i[0] for i in out])
                uF[I+1:N+1,:,k] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
                del _temp_uFs
            elif parall == 'joblib':
                out = Parallel(-1)(delayed(lambda i: solver.run_F_timed(t[i], t[i+1], u[i,:,k]))(i) for i in range(I,N))
                _temp_uFs = np.array([i[0] for i in out])
                uF[I+1:N+1,:,k] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
            else:
                temp_t = 0
                for i in range(I, N):
                    temp = solver.run_F_timed(t[i], t[i+1], u[i,:,k])
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
            
            
            model.fit_timed(x, D, k=k, data_x=data_x, data_y=data_D)

            if do_err_mdls:
                for mdl  in comp_mdls:
                    err_store_mdls[mdl.name][k] = list()
                    err_store_mdls['para'][k] = list()
                    mdl.fit_timed(x, D, k=k, data_x=data_x, data_y=data_D)

            
            if debug:
                preds_t = np.empty((N-I, n))
                truth_t = np.empty((N-I, n))
                preds_t.fill(np.nan)
                truth_t.fill(np.nan)
                
            for i in range(I, N):
                # run G solver on best initial value
                temp, temp_t = solver.run_G_timed(t[i], t[i+1], u[i,:,k+1])
                G_time += temp_t
                uG[i+1,:,k+1] = temp
                del temp, temp_t
                
                if not debug:
                    preds = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                                               uF[i+1,:,k], uG[i+1,:,k], i=i)
                
                if debug:
                    temp = solver.run_F(t[i], t[i+1], u[i,:,k+1])
                    opt_pred = temp
                    del temp
                    truth_t[i-I,:] =  opt_pred - uG[i+1,:,k+1] 
                    preds = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                                               uF[i+1,:,k], uG[i+1,:,k], i=i, truth=opt_pred - uG[i+1,:,k+1])
                    preds_t[i-I,:] = preds

                    if do_err_mdls:
                        for mdl  in comp_mdls:
                            mdl_pred = mdl.predict(u[i,:,k+1].reshape(1,-1), 
                                                uF[i+1,:,k], uG[i+1,:,k], i=i, truth=opt_pred - uG[i+1,:,k+1])
                            # np.max(np.log10(np.abs(truth_t[i-I,:]  - mdl_pred)))
                            err_store_mdls[mdl.name][k].append(np.max(np.log10(np.abs(truth_t[i-I,:]  - mdl_pred))))
                        err_store_mdls['para'][k].append(np.max(np.log10(np.abs(truth_t[i-I,:]  - preds))))
                        
                
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
            
            II = I
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
                _objs = {'t':t, 'I':I, 'verbose':verbose,
                     'u':u[...,:k+2], 'uG':uG[...,:k+2], 'uF':uF[...,:k+2], 'err':err[...,:k+2], 'x':x, 'D':D, 
                     'data_x':data_x[...,:k+2], 'data_D':data_D[...,:k+2], 'G_time':G_time, 'F_time':F_time,
                     'debug':debug, 'early_stop':early_stop, 'parall':parall, 'store_int':store_int, 'kwargs':kwargs,
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
        
        if do_err_mdls:
            return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 'data_x':data_x[...,:k+1], 
                'data_D':data_D[...,:k+1], 'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
                'conv_int':conv_int, 'err_store_mdls':err_store_mdls}
        else:
            return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 'data_x':data_x[...,:k+1], 
                    'data_D':data_D[...,:k+1], 'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
                    'conv_int':conv_int}
    
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
            
        
        n = self.n
        preds = self.get_preds(xm, ym, n, new_x, kwargs['i'])
        
        return preds
  
nngp_types = ['nn', 'col+rnd', 'col_only', 'row_col', 'row', 'col_full']

class NN(ModelAbstr):
    def __init__(self, *args, **kwargs):
        self.nn = kwargs.get('nn', 10)
        self.name = f'{self.nn}-NN'
        super().__init__(*args, **kwargs)

    def fit(self, x, y, k, *args, **kwargs):
        self.k = k
        self.x, self.y = x, y
        self.data_x = kwargs['data_x']
        self.data_y = kwargs['data_y']
     
    def predict(self, new_x, prev_F, prev_G, *args, **kwargs):
        self.i = kwargs['i']
        nn = self.nn
        data_x, data_y = self.data_x, self.data_y
            
        s_idx = np.argsort(scipy.spatial.distance.cdist(new_x, self.x, metric='sqeuclidean')[0,:])
        xm = self.x[s_idx[:nn], :]
        ym = self.y[s_idx[:nn], :]
        return ym.mean(axis=0)
    
ode = Rossler(normalization='-11')
config = Config(ode).get()
s = SolverRK(ode.get_vector_field(), **config)

n = ode.get_dim()
N = config['N']

comp_mdls_nngp = [NNGP_alt(n=n, N=N, worker_pool=MyPool(), nntype=nt, nn=15) for nt in nngp_types]
comp_mdls_nn = [NN(N=N, nn=nn) for nn in [1, 2, 3, 4, 5, 10, 15, 30]]

comp_mdls_gp = [NNGP_p(n=n, N=N, worker_pool=MyPool(), nn=nn) for nn in [10, 25, 40]]
for mdl in comp_mdls_gp:
    mdl.name = mdl.name + str(mdl.nn)

p = Paramod(ode, s, **config)

res_nn = p.run(comp_mdls=comp_mdls_nn, debug=True)
err_nn = res_nn['err_store_mdls']

# res_nngp = p.run(comp_mdls=comp_mdls_nngp, debug=True)
# err = res_nngp['err_store_mdls']

res_gp = p.run(model='gpjax',comp_mdls=comp_mdls_gp, debug=True)
err_gp = res_gp['err_store_mdls']




#%
err = err_nn
fig, axs = plt.subplots(1,3, figsize=(10,2))

def tr(lbl):
    if lbl == '1-NN':
        return '1-nn'
    elif lbl == '2-NN':
        return '2-nn'
    elif lbl == '3-NN':
        return '3-nn'
    elif lbl == '4-NN':
        return '4-nn'
    else:
        raise Exception(3)
for i, k in enumerate(range(4, 7)):

    x_plt = (np.arange(a:=len(err['para'][k])))+(N-a)
    ax = axs[i%3]
    # for mdl in ['para','1-NN', '2-NN', '3-NN', '4-NN', '5-NN', '10-NN', '15-NN', '30-NN'][:10]: #err:
    #     ax.plot(err[mdl][k], label=mdl)
    c = ['gray', 'green', 'red']
    for j, mdl in enumerate(['para','1-NN', '2-NN',  '4-NN'][:10]): #err:
        if j == 0:
            ax.plot(x_plt, err[mdl][k], ls=(0,(5,10)), label='Para', c='black')
        else:
            ax.plot(x_plt, err[mdl][k], label=tr(mdl), alpha=0.5, c=c[j-1])
    ax.axhline(-6,ls='dashed', lw=1, color='gray')
    ax.axhline(-8,ls='dashed', lw=1, color='black')
    ax.axhline(-10,ls='dashed', lw=1, color='gray')
    ax.set_title(f'Rossler - k={k+1}')
fig.supxlabel('Interval $i$')
fig.supylabel('Prediction Error')
fig.tight_layout()
ax.legend()
store_fig(fig, 'rossler_pred_err_para')

#%
err = err_gp.copy()
err['GPara'] = err.pop('para')
err['nnGPara (10)'] = err.pop('NNGP10')
err['nnGPara (25)'] = err.pop('NNGP25')
err['nnGPara (40)'] = err.pop('NNGP40')
# fig, axs = plt.subplots(1,3, figsize=(10,2))
fig, axs = plt.subplot_mosaic('1112223334', figsize=(12,2), constrained_layout=True)

for i, k in enumerate(range(4, 7)):

    x_plt = (np.arange(a:=len(err['GPara'][k])))+(N-a)
    print(str(i%3+1))
    ax = axs[str(i%3+1)]
    # for mdl in ['GPara','1-NN', '2-NN', '3-NN', '4-NN', '5-NN', '10-NN', '15-NN', '30-NN'][:10]: #err:
    #     ax.plot(err[mdl][k], label=mdl)
    c = ['gray', 'green', 'red']
    for j, mdl in enumerate(['GPara', 'nnGPara (10)', 'nnGPara (25)', 'nnGPara (40)']):
        if j == 0:
            ax.plot(x_plt, err[mdl][k], ls=(0,(5,10)), label='GPara', c='black')
        else:
            ax.plot(x_plt, err[mdl][k], label=mdl, alpha=0.5, c=c[j-1])
    ax.axhline(-6,ls='dashed', lw=1, color='gray')
    ax.axhline(-8,ls='dashed', lw=1, color='black')
    ax.axhline(-10,ls='dashed', lw=1, color='gray')
    ax.set_title(f'Rossler - k={k+1}')
fig.supxlabel('Interval $i$')
fig.supylabel('Prediction Error')
# fig.tight_layout()
axs['3'].legend(loc='upper left', bbox_to_anchor=(1, 1))
axs['4'].axis("off")
store_fig(fig, 'rossler_pred_err_nngp')

#%
err = err_nn
fig, axs = plt.subplot_mosaic('1112223334\n5556667778', figsize=(12,4), constrained_layout=True)
# axs = axss[0,:]
for i, k in enumerate(range(4, 7)):
    ax = axs[str(i%3+1)]
    x_plt = (np.arange(a:=len(err['para'][k])))+(N-a)
    # ax = axs[i%3]
    # for mdl in ['para','1-NN', '2-NN', '3-NN', '4-NN', '5-NN', '10-NN', '15-NN', '30-NN'][:10]: #err:
    #     ax.plot(err[mdl][k], label=mdl)
    c = ['gray', 'green', 'red']
    for j, mdl in enumerate(['para','1-NN', '2-NN',  '4-NN'][:10]): #err:
        if j == 0:
            ax.plot(x_plt, err[mdl][k], ls=(0,(5,10)), label='Para', c='black')
        else:
            ax.plot(x_plt, err[mdl][k], label=tr(mdl), alpha=0.5, c=c[j-1])
    ax.axhline(-6,ls='dashed', lw=1, color='gray')
    ax.axhline(-8,ls='dashed', lw=1, color='black')
    ax.axhline(-10,ls='dashed', lw=1, color='gray')
    ax.set_title(f'Rossler - k={k+1}')
fig.supxlabel('Interval $i$')
fig.supylabel('Prediction Error')
# fig.tight_layout()
axs['3'].legend(loc='upper left', bbox_to_anchor=(1, 1))
axs['4'].axis("off")

err = err_gp.copy()
err['GPara'] = err.pop('para')
err['nnGPara (10)'] = err.pop('NNGP10')
err['nnGPara (25)'] = err.pop('NNGP25')
err['nnGPara (40)'] = err.pop('NNGP40')
# fig, axs = plt.subplots(1,3, figsize=(10,2))
# fig, axs = plt.subplot_mosaic('1112223334', figsize=(12,2), constrained_layout=True)
# axs = axss[1,:]
for i, k in enumerate(range(4, 7)):

    x_plt = (np.arange(a:=len(err['GPara'][k])))+(N-a)
    print(str(i%3+1))
    ax = axs[str(i%3+1+4)]
    # for mdl in ['GPara','1-NN', '2-NN', '3-NN', '4-NN', '5-NN', '10-NN', '15-NN', '30-NN'][:10]: #err:
    #     ax.plot(err[mdl][k], label=mdl)
    c = ['gray', 'green', 'red']
    for j, mdl in enumerate(['GPara', 'nnGPara (10)', 'nnGPara (25)', 'nnGPara (40)']):
        if j == 0:
            ax.plot(x_plt, err[mdl][k], ls=(0,(5,10)), label='GPara', c='black')
        else:
            ax.plot(x_plt, err[mdl][k], label=mdl, alpha=0.5, c=c[j-1])
    ax.axhline(-6,ls='dashed', lw=1, color='gray')
    ax.axhline(-8,ls='dashed', lw=1, color='black')
    ax.axhline(-10,ls='dashed', lw=1, color='gray')
    ax.set_title(f'Rossler - k={k+1}')
fig.supxlabel('Interval $i$')
fig.supylabel('Prediction Error (log)')
fig.tight_layout()
axs['7'].legend(loc='upper left', bbox_to_anchor=(1, 1))
axs['8'].axis("off")
store_fig(fig, 'rossler_pred_err_both')