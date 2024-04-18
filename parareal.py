import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import concurrent.futures
from cycler import cycler
import os
import pickle
import time
from joblib import Parallel, delayed

from models import BareParareal, GPjax_p, NNGP_p, ELM
from systems import ODE
from solver import SolverAbstr


class MyPool():
    
    @staticmethod
    def map(*args, chunksize=None, **kwargs):
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
    
    def __init__(self, ode, solver, tspan, N, epsilon=5e-7, verbose='v', **kwargs):

        if not isinstance(ode, ODE):
            raise Exception('ode must be an instance of the ODE class, see systems.py file.')
        
        if not isinstance(solver, SolverAbstr):
            raise Exception('solver must be an instance of the SolverAbstr class, see solver.py file.')
        
        self.tspan = tspan
        self.N = N
        self.epsilon = epsilon
        self.runs = dict()
        self.fine = None
        self.ode_name = ode.name
        self.n = ode.get_dim()
        
        self.ode = ode
        self.solver = solver
        self.f = ode.get_vector_field()
        self.u0 = ode.get_init_cond()
        
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
        elif model.lower() == 'elm':
            mdl = ELM(d=self.n, N=self.N, **kwargs)
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
        
        ode = self.ode
        solver = self.solver
        self.ode = None
        self.solver = None
        if objs is not None:
            pool = objs['kwargs']['pool']
            objs['kwargs']['pool'] = None
            self.objs = objs
        if mdl is not None:
            self.mdl = mdl.store()
            
        with open(os.path.join(path, name), 'wb') as _file:
            pickle.dump(self, _file, pickle.HIGHEST_PROTOCOL)
            
        self.ode = ode
        self.solver: SolverAbstr = solver
        if objs is not None:
            self.objs = None
            objs['kwargs']['pool'] = pool
        if mdl is not None:
            self.mdl = None
            
    def load_int_dump(self, other, cstm_mdl_name=None, add_model=False, **kwargs):
        self.tspan = other.tspan
        self.n = other.n
        self.N = other.N
        self.epsilon = other.epsilon
        self.runs = other.runs
        self.fine = other.fine
        self.ode_name = other.ode_name
        
        self.ode = kwargs.get('ode', other.ode)
        self.solver = kwargs.get('solver', other.solver)

        if self.ode_name != self.ode.name or self.n != self.ode.get_dim():
            raise Exception('Input and previous ODEs do not match')
        
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
        obj_names = ['t', 'I', 'verbose', 
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
        
        return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 'data_x':data_x[...,:k+1], 
                'data_D':data_D[...,:k+1], 'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
                'conv_int':conv_int}

    def _build_plot_data(self, t, u, err, **kwargs):
        Nf, N, u0, f, F = self.Nf, self.N, self.u0, self.f, self.F
        # u_par = np.empty((int(Nf/N)*(N-1) + int(Nf/N)+1, u0.shape[0]))
        # u_par.fill(np.nan)
        # for i in range(N):
        #     temp = RK(np.linspace(t[i], t[i+1], num=int(Nf/N)+1), u[i, :, -1], f, F)
        #     u_par[i*int(Nf/N):(i+1)*int(Nf/N),:] = temp[:-1,:]
        # u_par[-1] = temp[-1,:]
        
        u_interval = u
        # u_continuous = u_par
        u_continuous = None
        return {'u_int':u_interval, 'u_cont': u_continuous, 'err':err, 't':t}
        
    def build_cont_traj(self, key=None):
        if key is None:
            if len(self.runs) != 1:
                raise Exception('Multiple runs, must specify key')
            key = list(self.runs.keys())[0]

        if isinstance(key, dict) and 't' in key and 'u' in key:
            t, u = key['t'], key['u']
        else:
            t, u = self.runs[key]['t'], self.runs[key]['u']

        return self._build_cont_traj(t, u)

    def _build_cont_traj(self, t, u):
        solver = self.solver
        u_full = []
        for i in range(self.N):
            temp = solver.run_F_full(t[i], t[i+1], u[i, :, -1])
            u_full.append(temp)
            temp = temp[-1,:]
        out = np.vstack(u_full)
        return out
    
    def clear_plot_obj(self):
        self.runs = dict()
    
    def plot(self, skip = [], add_name=True, add_title=''):
        runs, tspan, Nf, u0 = self.runs, self.tspan, self.Nf, self.u0
        f, F, epsilon = self.f, self.F, self.epsilon
        
        if len(add_title) != 0:
            add_title = add_title + ' - '
        
        if self.fine is None:
            # fine, fine_t = RK_t(np.linspace(tspan[0], tspan[-1], num=Nf+1), u0, f, F)
            # self.fine, self.fine_t = fine, fine_t
            pass
        else:
            fine = self.fine
        
        plot_data = {key : self._build_plot_data(**runs[key]) for key in runs}
        
        if 0 not in skip:
            print('Plot 0 is not implemented, code needs to be updated')
            # fig, ax = plt.subplots(u0.shape[0],1)
            # x_plot = np.linspace(tspan[0], tspan[-1], num=Nf+1)
            # for i in range(u0.shape[0]):
            #     ax[i].plot(x_plot, fine[:,i], linewidth=0.5, label='Fine')
            #     for mdl_name in plot_data:
            #         line2d, = ax[i].plot(x_plot, plot_data[mdl_name]['u_cont'][:,i], 
            #                              linewidth=0.5, label=mdl_name)
            #         ax[i].scatter(plot_data[mdl_name]['t'], plot_data[mdl_name]['u_int'][:,i,-1], 
            #                       marker='x', s=2, color=line2d.get_color())
            #     ax[i].set_ylabel(f'$u_{{{i+1}}}(t)$')
            # ax[i].legend()
            # ax[i].set_xlabel('$t$')
            # if add_name:
            #     fig.suptitle(f'{self.ode_name} - {add_title}Comparison of trajectories')
            # else:
            #     fig.suptitle('Comparison of trajectories')
            # fig.tight_layout()
        
        if 1 not in skip:
            print('Plot 1 is not implemented, code needs to be updated')
            # fig, ax = plt.subplots(u0.shape[0],1)
            # x_plot = np.linspace(tspan[0], tspan[-1], num=Nf+1)
            # for i in range(u0.shape[0]):
            #     for mdl_name in plot_data:
            #         y_plot = np.log10(np.abs(fine - plot_data[mdl_name]['u_cont']))
            #         ax[i].plot(x_plot, y_plot[:,i], linewidth=0.5, label=mdl_name)
            #     ax[i].set_ylabel(f'$u_{{{i+1}}}$ log error')
            #     ax[i].axhline(np.log10(epsilon), linestyle='dashed', color='gray', linewidth=1, label='Tolerance')
            # ax[i].legend()
            # ax[i].set_xlabel('$t$')
            # if add_name:
            #     fig.suptitle(f'{self.ode_name} - {add_title}Algorithm error wrt fine solver')
            # else:
            #     fig.suptitle('Algorithm error wrt fine solver')
            # fig.tight_layout()
        
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
            fig, ax = plt.subplot_mosaic('AAA.BBCC', constrained_layout=True)
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
                # fine, fine_t = RK_t(np.linspace(self.tspan[0], self.tspan[-1], num=self.Nf+1), self.u0, self.f, self.F)
                fine, fine_t = self.solver.run_F_timed(self.tspan[0], self.tspan[-1], self.u0)
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
        
   
class PararealLight(Parareal):

    def load_int_dump(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support loading from intermediate dumps')
    

    def _run_from_int(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support loading from intermediate dumps')
    

    def _build_plot_data(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support plotting')
    

    def plot(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support plotting')
    

    def print_times(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support printing times')
    

    def print_speedup(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support printing speedup')
    

    def plot_all_err(self, *args, **kwargs):
        raise NotImplementedError('PararealLight does not support plotting errors')
    

    def _parareal(self, model, early_stop=None, parall='Serial', store_int=False, **kwargs):
        tspan, N, epsilon, n = self.tspan, self.N, self.epsilon, self.n
        f, u0 = self.f, self.u0
        solver: SolverAbstr = self.solver

        if kwargs.get('debug', False):
            print('WARNING: PararealLight does not support debug mode')
                         
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

        if verbose and parall != 'serial':
            print(f'Running {model.name} with {parall} parallel backend')
            
        conv_int = []
        
        # u = np.empty((N+1, n, N+1))
        # uG = np.empty((N+1, n, N+1))
        # uF = np.empty((N+1, n, N+1))
        err = np.empty((N+1, N))
        # u.fill(np.nan)
        # uG.fill(np.nan)
        # uF.fill(np.nan)
        err.fill(np.nan)

        # err_old = np.empty((N+1, N))
        # err_old.fill(np.nan)

        u_curr = np.empty((N+1, n))
        u_next = np.empty((N+1, n))
        uG_curr = np.empty((N+1, n))
        uG_next = np.empty((N+1, n))
        uF_curr = np.empty((N+1, n))
        uF_next = np.empty((N+1, n))
        u_curr.fill(np.nan)
        u_next.fill(np.nan)
        uG_curr.fill(np.nan)
        uG_next.fill(np.nan)
        uF_curr.fill(np.nan)
        uF_next.fill(np.nan)
        
        # x_old = np.zeros((0, n))
        # D_old = np.zeros((0,n))
        x = np.zeros((0, n))
        D = np.zeros((0,n))
        
        G_time = 0
        F_time = 0
        F_time_serial = 0
        
            
        # u[0,:,:] = u0[:, np.newaxis]
        # uG[0,:,:] = u[0,:,:]
        # uF[0,:,:] = u[0,:,:]

        u_curr[0,:] = u0
        uG_curr[0,:] = u_curr[0,:]
        uF_curr[0,:] = u_curr[0,:]
        u_next[0,:] = u_curr[0,:]
        uG_next[0,:] = u_curr[0,:]
        uF_next[0,:] = u_curr[0,:]
        
        
        # Initialization: run G sequentially
        temp = u0
        for i in range(N):
            temp, temp_t = solver.run_G_timed(t[i], t[i+1], temp)
            G_time += temp_t
            # uG[i+1,:,0] = temp
            uG_curr[i+1,:] = temp

        del temp, temp_t
        # u[:,:,0] = uG[:,:,0]
        u_curr[:,:] = uG_curr[:,:]

        
        #Step 2: integrate using F (fine solver) in parallel with the current best initial
        # values
        for k in range(N):
            if verbose == 'v':
                print(f'{self.ode_name} {model.name} iteration number (out of {N}): {k+1} ')
                
            s_time = time.time()
            if parall == 'mpi':
                out = list(pool.map(solver.run_F_timed, t[I:N], t[I+1:N+1], [u_curr[i,:] for i in range(I,N)]))
                _temp_uFs = np.array([i[0] for i in out])
                uF_curr[I+1:N+1,:] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
                del _temp_uFs
            elif parall == 'joblib':
                out = Parallel(-1)(delayed(lambda i: solver.run_F_timed(t[i], t[i+1], u_curr[i,:]))(i) for i in range(I,N))
                _temp_uFs = np.array([i[0] for i in out])
                uF_curr[I+1:N+1,:] = _temp_uFs
                F_time_serial += np.array([i[1] for i in out]).mean()
            else:
                temp_t = 0
                for i in range(I, N):
                    # temp_old = solver.run_F_timed(t[i], t[i+1], u[i,:,k])
                    temp, _temp_t_int = solver.run_F_timed(t[i], t[i+1], u_curr[i,:])
                    # uF[i+1,:,k] = temp_old[0]
                    uF_curr[i+1,:] = temp
                    temp_t =+ _temp_t_int
                F_time_serial += temp_t/(N-I)
            F_time += time.time() - s_time
            del s_time
            # save values forward (as solution at time I+1 is now converged)
            # uG[I+1,:,(k+1):] = uG[I+1,:,k].reshape(-1,1)
            # uF[I+1,:,(k+1):] = uF[I+1,:,k].reshape(-1,1)
            # u[I+1,:,(k+1):] = uF[I+1,:,k].reshape(-1,1)

            uG_next[I+1,:] = uG_curr[I+1,:]
            uF_next[I+1,:] = uF_curr[I+1,:]
            u_next[I+1,:] = uF_curr[I+1,:]
            I = I + 1
            # collect training data
            # x_old = np.vstack([x_old, u[I-1:N+1-1,:,k]])
            # D_old = np.vstack([D_old, uF[I:N+1,:,k] - uG[I:N+1,:,k]])
            x = np.vstack([x, u_curr[I-1:N+1-1,:]])
            D = np.vstack([D, uF_curr[I:N+1,:] - uG_curr[I:N+1,:]])
            
            
            # early stop if only one interval was missing
            if I == N:
                if verbose == 'v':
                    print('WARNING: early stopping')
                # err_old[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
                err_old = np.nextafter(epsilon, 0)
                err[:,k] = np.linalg.norm(u_next[:,:] - u_curr[:,:], np.inf, 1)
                err[-1,k] = np.nextafter(epsilon, 0)
                break
            
            
            model.fit_timed(x, D, k=k)
                
            for i in range(I, N):
                # run G solver on best initial value
                # temp_old, _ = solver.run_G_timed(t[i], t[i+1], u[i,:,k+1])
                temp, temp_t = solver.run_G_timed(t[i], t[i+1], u_next[i,:])
                G_time += temp_t
                # uG[i+1,:,k+1] = temp_old
                uG_next[i+1,:] = temp
                del temp, temp_t
                
                # preds_old = model.predict_timed(u[i,:,k+1].reshape(1,-1), 
                #                             uF[i+1,:,k], uG[i+1,:,k], i=i)
                
                preds = model.predict_timed(u_next[i,:].reshape(1,-1), 
                                            uF_curr[i+1,:], uG_curr[i+1,:], i=i)
                
                
                # do predictor-corrector update
                # u[i+1,:,k+1] = uF[i+1,:,k] + uG[i+1,:,k+1] - uG[i+1,:,k]
                # u[i+1,:,k+1] = preds_old + uG[i+1,:,k+1]
                u_next[i+1,:] = preds + uG_next[i+1,:]
                
            
            # error catch
            a = 0
            if np.any(np.isnan(uG_next[:,:])):
                raise Exception("NaN values in initial coarse solve - increase Ng!")
            # if np.any(np.isnan(uG[:,:, k+1])):
            #     raise Exception("NaN values in initial coarse solve - increase Ng!")
                           
            # Step 4: Converence check
            # checks whether difference between solutions at successive iterations
            # are small, if so then that time slice is considered converged. 
                          
            # err_old[:,k] = np.linalg.norm(u[:,:,k+1] - u[:,:,k], np.inf, 1)
            # err_old[I,k] = 0

            err[:,k] = np.linalg.norm(u_next[:,:] - u_curr[:,:], np.inf, 1)
            err[I,k] = 0
            
            # I_old = I
            # II = I_old
            # for p in range(II+1, N+1):
            #     if err[p, k] < epsilon:
            #         u[p,:,k+2:] = u[p,:,k+1].reshape(-1,1)
            #         uG[p,:,k+2:] = uG[p,:,k+1].reshape(-1,1)
            #         uF[p,:,k+1:] = uF[p,:,k].reshape(-1,1)
            #         I_old = I_old + 1
            #     else:
            #         break

            u_curr[...] = u_next[...]
            uG_curr[...] = uG_next[...]
            II = I
            for p in range(II+1, N+1):
                if err[p, k] < epsilon:
                    u_next[p,:] = u_curr[p,:]
                    uG_next[p,:] = uG_curr[p,:]
                    uF_next[p,:] = uF_curr[p,:]
                    I += 1
                else:
                    break
            uF_curr[...] = uF_next[...]


            if verbose == 'v':    
                print('--> Converged:', I)
            conv_int.append(I)
            if store_int:
                raise NotImplementedError('PararealLight does not support storing intermediate results')
                # name_base = f'{self.ode_name}_{self.N}_{model.name}_int'
                # int_dir = kwargs.get('int_dir', '')
                # name_base = kwargs.get('int_name', name_base)
                # int_name = f'{name_base}_{k}'
                # _objs = {'t':t, 'I':I, 'verbose':verbose,
                #      'u':u[...,:k+2], 'uG':uG[...,:k+2], 'uF':uF[...,:k+2], 'err':err[...,:k+2], 'x':x, 'D':D, 
                #      'G_time':G_time, 'F_time':F_time,
                #      'early_stop':early_stop, 'parall':parall, 'store_int':store_int, 'kwargs':kwargs,
                #      'k':k, 'conv_int': conv_int}
                
                # self.store(path=os.path.join(int_dir, name_base), name=int_name, mdl=model, objs=_objs)
                
                
            if I == N:
                break
            if (early_stop is not None) and k == (early_stop-1):
                if verbose == 'v':
                    print('Early stopping due to user condition.')
                break
        
        debug_dict = {}
            
            
        timings = {'F_time':F_time, 'G_time': G_time, 'F_time_serial_avg': F_time_serial}
        timings.update(model.get_times())
        
        # return {'t':t, 'u':u[:,:,:k+1], 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 
        #         'u_curr':u_curr, 'uG_curr':uG_curr, 'uF_curr':uF_curr, 'err_old':err_old[:,:k+1],
        #         'uF':uF[:,:,:k+1], 'uG':uG[:,:,:k+1], 'x_old':x_old, 'D_old':D_old,
        #         'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
        #         'conv_int':conv_int}

        return {'t':t, 'u':u_curr, 'err':err[:, :k+1], 'x':x, 'D':D, 'k':k+1, 
                'timings':timings, 'debug_dict':debug_dict, 'converged':I==N, 
                'conv_int':conv_int}
    

    def _build_cont_traj(self, t, u):
        solver = self.solver
        u_full = []
        for i in range(self.N):
            temp = solver.run_F_full(t[i], t[i+1], u[i, :])
            u_full.append(temp)
            temp = temp[-1,:]
        out = np.vstack(u_full)
        return out
