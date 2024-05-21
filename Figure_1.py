from globals import *

# Parareal Evolution 
#%% 1D System evolution plot - Panel A

from parareal import Parareal
from models import BareParareal, NNGP_p
import time
import matplotlib
import matplotlib.animation as animation
from systems import Lorenz, FHN_ODE, Brusselator, ODE
from solver import SolverRK
import numpy as np
from configs import Config
from matplotlib import pyplot as plt

class ParaMod(Parareal):
    

    def _parareal(self, debug=False, early_stop=None, parall='Serial', store_int=False, _load_mdl=False, **kwargs):
        tspan, N, epsilon, n = self.tspan, self.N, self.epsilon, self.n
        # f, u0 = self.f, self.u0
        solver: SolverAbstr = self.solver
        model = BareParareal(N=self.N, **kwargs)
                         
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
                
                yield _objs

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
            
                
                # self.store(path=os.path.join(int_dir, name_base), name=int_name, mdl=model, objs=_objs)
                
                
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

class Ode1d(ODE):
    def __init__(self, **kwargs):
        mn, mx = np.array([[0.1], [14700]])
        u0 = np.array([0.1]) 
        super().__init__('OneDim', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        return (-(t-5)*u*0.3)
    

    @staticmethod
    def _f_np(t, u):
        return (-(t-5)*u*0.3)

class MySolverRK(SolverRK):
    def run_F_full(self, t0, t1, u0):
        noise = np.arange(1001).reshape(-1,1)
        noise = noise/noise[-1,0]
        out = super().run_F_full(t0, t1, u0)
        # if t0 in [1]:
        #     out = out + 0.8*noise
        if t0 in [2]:
            out = out + 0.35*noise
        if t0 in [3]:
            out = out - 0.3*noise
        if t0 in [4]:
            out = out - 0.3*noise
        if t0 == 5:
            out = out - 0.1*noise
        if t0 == 6:
            out = out - 0.1*noise
        return out
    
    def run_F(self, t0, t1, u0):
        return self.run_F_full(t0, t1, u0)[-1,:]
    
N = 10
T = 10
ode = Ode1d()
f = ode.get_vector_field()
s = MySolverRK(f, Ng=2, Nf=1e3, G='RK1', F='RK8')
p = ParaMod(ode, s, tspan=(0, T), N=N)


tspan = p.tspan
t0 = 0
N = p.N
u0 = p.u0

p_s = 20 #point size


def gen_full_traj(solver, t, init_cond):
        u_full = list()
        for i in range(t.shape[0] - 1):
            temp = solver(t[i], t[i+1], init_cond[i, :])
            u_full.append((np.linspace(t[i], t[i+1], num=temp.shape[0]), temp))
        # out = np.vstack(u_full)
        return u_full, ''

def do_one_plot(data, until_i, c, ax, k, no_points=False, col=None):
    artists = list()
    for i in range(until_i):
        artists.append(ax.plot(*data[i], c=c[i], lw=1)[0])
        if not no_points:
            if col is None:
                col = c[i]
            artists.append(ax.scatter(data[i][0][0], data[i][1][0,:], c=col, s=p_s))
            # artists.append(ax.text(data[i][0][0], data[i][1][0,:]-0.2, f'$U_{i}$', fontsize=10))
            artists.append(plot_one_init_cond_label(data[i][0][0], data[i][1][0,:], i, ax, k))
    return artists


def get_colors(t, fine):
    if fine:
        cols = matplotlib.cm.get_cmap('Blues')
    else:
        cols = matplotlib.cm.get_cmap('Reds')
    c = cols(np.linspace(0.3, 0.99, t.shape[0]-1))
    return c

def plot_evolution(init_cond, solver, t, ax, k, fine=False):
    container = []
    u_list, u_full = gen_full_traj(solver, t, init_cond)
    c = get_colors(t, fine)
    for i in range(1, t.shape[0]):
        if i == 1:
            artists = do_one_plot(u_list, i, c, ax, k, col='black')
            init_cond_artist = ax.scatter(t[i-1], init_cond[i-1,:], c='black', s=p_s)
        else:
            artists = do_one_plot(u_list, i, c, ax, k)
            init_cond_artist = ax.scatter(t[i-1], init_cond[i-1,:], c=c[i-1], s=p_s)
        artists.append(init_cond_artist)
        artists.append(get_interval_artist(i))
        container.append(artists)
    return container, u_list, c

def plot_evolution_oneshot(init_cond, solver, t, ax, k, fine=False):
    container = []
    u_list, u_full = gen_full_traj(solver, t, init_cond)
    c = get_colors(t, fine)
    i = t.shape[0]-1
    artists = do_one_plot(u_list, i, c, ax, k, no_points=True)
    # init_cond_artist = ax.scatter(*init_cond[i-1,:], c=c[i-1])
    # artists.append(init_cond_artist)
    container.append(artists)
    return container, u_list, c

def plot_init_cond(t, init_cond, ax, k: int, fine):
    c = get_colors(t, fine)
    artists = []
    artists.append(ax.scatter(t[:-1], init_cond[:-1], c=c, s=p_s))
    artists.extend(plot_all_init_cond_label(t, init_cond, range(t.shape[0]-1), ax, k=k))
    return artists

def plot_all_init_cond_label(t, init_cond, i_s, ax, k):
    artists = []
    for i in i_s:
        artists.append(plot_one_init_cond_label(t[i], init_cond[i], i, ax, k))
    return artists

def plot_one_init_cond_label(t, init_cond, i, ax, k):
    fts = 18
    if k is None:
        return ax.text(t, init_cond-0.3, f'$U_{i}$', fontsize=fts)
    else:
        if i == 0 or i == 9:
            return ax.text(t, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
        elif i < 6:
            return ax.text(t, init_cond-0.4, f'$U_{i}^{k}$', fontsize=fts)
        else:
            return ax.text(t-0.3, init_cond-0.48, f'$U_{i}^{k}$', fontsize=fts)

def show_one_update(base, t, uG, uF, u, I, k, ax):
    init_cond = u[I,:,k+1]
    artists = []
    artists.extend(base)
    G_new = s.run_G_full(t[I], t[I+1], init_cond)
    assert np.allclose(G_new[-1,:], uG[I+1,:,k+1])
    a1 = ax.scatter(t[I], init_cond, c='green', s=p_s)
    a2 = ax.plot(np.linspace(t[i], t[I+1], num=G_new.shape[0]), G_new, c='red', lw=1)[0]
    a3 = ax.scatter(t[I+1], G_new[-1,:], c='gray', s=p_s)
    a4 = ax.scatter(t[I+1], uG[I+1,:,k], c='red', s=p_s)
    a5 = ax.scatter(t[I+1], uF[I+1,:,k], c='blue', s=p_s)
    preds = uF[I+1,:,k] + uG[I+1,:,k+1] - uG[I+1,:,k]
    assert np.allclose(preds, u[I+1,:,k+1])
    a6 = ax.scatter(t[I+1], preds, c='green', s=p_s)
    a9 = plot_one_init_cond_label(t[I+1], preds, I+1, ax, k+1)
    a7 = ax.plot([t[I+1],t[I+1]], [uF[I+1,:,k], uG[I+1,:,k]], c='black')[0]
    a8 = ax.plot([t[I+1], t[I+1]], (preds, G_new[-1,:]), c='black')[0]
    artists.extend([a1, a2, a3, a4, a5, a6, a7, a8, a9])

    base_next = []
    base_next.extend(base)
    # base_next.extend([a1, a6, a9])
    base_next.extend([a1, a2, a3, a4, a5, a6, a7, a8, a9])

    return artists, base_next

def add_artist(base, artist):
    if isinstance(base[0], list):
        for frame in base:
            if isinstance(artist, list):
                frame.extend(artist)
            else:
                frame.append(artist)
    else:
        base.append(artist)

def get_interval_artist(i):
    return ax.text(0,0,'')
    return ax.text(3.5, 4.9, f'Interval: {i:2d}', fontsize=12)

def get_iteration_artist(k):
    return ax.text(0, 4.7, f'Iteration: {k:2d}', fontsize=16)

def get_parall_artist(parall=False):
    return ax.text(0,0,'')
    if parall:
        return ax.text(7, 4.9, 'Parallel, running $\mathscr{F}$', fontsize=12, color='green')
    else:
        return ax.text(7, 4.9, 'Sequential, running $\mathscr{G}$', fontsize=12, color='red')

def get_comment_artist(comment, font_size=12):
    return ax.text(0,0,'')
    return ax.text(6.3, 4.6, comment, fontsize=font_size, color='black',verticalalignment='top')


ss = SolverRK(f, Ng=2, Nf=1e3, G='RK1', F='RK8')
ss.Ng = s.Ng * p.N
ss.Nf = s.Nf * p.N
truth = ss.run_F_full(*tspan, p.u0)[::10,:]
truth_x = np.linspace(*tspan, num=truth.shape[0])
all_frames = []
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Time $t$ - Panel A', fontsize=16)
ax.set_ylabel('Solution $u(t)$', fontsize=16)
ax.tick_params(axis='both', labelsize=13)
for k, objs in enumerate(p._parareal(store_int=True)):
    uG = objs['uG']
    uF = objs['uF']
    u = objs['u']
    t = objs['t']
    I = objs['I']

    truth_artist = ax.plot(truth_x, truth, c='gray', lw=1, alpha=0.6)[0]


    iter_frames = []

    # explan_u = ax.text(-0.4, 4.6, '$U_{i}^{k}$: Initial condition at\n iteration $k$ for interval $i$.', fontsize=12, color='black',verticalalignment='top')
    explan_u = ax.text(0,0,'')

    # update_rule = ax.text(3.2, 0.25, '$U_{i}^{k+1}=\mathscr{G}(U_{i-1}^{k+1})+\mathscr{F}(U_{i-1}^{k})-\mathscr{G}(U_{i-1}^{k})$', fontsize=12, color='black',verticalalignment='top')
    update_rule = ax.text(0,0,'')

    # Initialization: show G
    if k == 0:
        frame = []
        frame.append(truth_artist)
        frame.append(ax.scatter(t0, u0, c='black'))
        frame.append(get_comment_artist(f'True solution for 1D ODE system.\nThe initial condition is given (black dot).\n$u_0={u0[0]}$'))
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)

        # showing the intervals
        frame = []
        F_full = []
        temp = u0
        for i in range(p.N):
            temp = s.run_F_full(t[i], t[i+1],temp)
            F_full.append(temp)
            temp = temp[-1,:]

        interval_bands = []
        for i, segment in enumerate(F_full):
            if i % 2 == 0:
                c = 'black'
            else:
                c = 'gray'
            frame.append(ax.plot(np.linspace(t[i], t[i+1], num=segment.shape[0]), segment, c=c, lw=2, alpha=0.6)[0])
            frame.append(ax.scatter(t[i], segment[0,:], c='black', s=10))
            interval_bands.append(ax.axvline(t[i+1], lw=1, alpha=0.4))
            add_artist(frame, plot_one_init_cond_label(t[i], segment[0,:], i, ax, k=None))
        interval_bands.append(ax.axvline(t[0], lw=1, alpha=0.4))
        
        frame.append(get_comment_artist(f'Divide the timespan into $N={N}$ intervals.\n The true intervals are shown.'))
        frame.append(ax.scatter(t0,u0, c='black'))
        frame.extend(interval_bands)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)


        container, *_ = plot_evolution(uG[...,0], s.run_G_full, t, ax, k, fine=False)
        iter_text = get_iteration_artist(0)
        add_artist(container, truth_artist)
        add_artist(container, iter_text)
        add_artist(container, interval_bands)
        add_artist(container, explan_u)
        add_artist(container, get_comment_artist('Initialization: building approximate \ninitial conditions for each\ninterval using $\mathscr{G}$.'))
        add_artist(container, get_parall_artist(parall=False))
        all_frames.extend(container)
    

    
    
    # Show Approximate initial conditions
    frame = []
    init_cond_artist = plot_init_cond(t, u[...,k], ax, k, fine=False)
    frame.extend(init_cond_artist)
    if k == 0:
        frame.append(get_comment_artist('Approximate initial conditions\n'+f'$U_{"i"}^{k+1}$'+' from $\mathscr{G}$'))
    else:
        frame.append(get_comment_artist('Updated initial conditions '+f'$U_{"i"}^{k+1}$'))
    iter_frames.append(frame)
    iter_frames.append(frame)
    iter_frames.append(frame)
    iter_frames.append(frame)


    base = []
    # Run F in parallel
    container, u_list, c = plot_evolution_oneshot(u[...,k], s.run_F_full, t, ax, k, fine=True)
    container[0].extend(init_cond_artist)
    base.extend(container[0])
    add_artist(container, get_parall_artist(parall=True))
    if k == 0:
        add_artist(container, get_comment_artist('Running $\mathscr{F}$ in parallel from the \napproximate initial conditions'+f' $U_{"i"}^{k+1}$'))
    else:
        add_artist(container, get_comment_artist('Running $\mathscr{F}$ in parallel from the \nupdated initial conditions'+f' $U_{"i"}^{k+1}$'))
    iter_frames.extend(container)
    iter_frames.extend(container)
    iter_frames.extend(container)
    iter_frames.extend(container)



    start = []
    # Add previously converged intervals
    # Add newly converged interval

    # base = []
    add_artist(base, update_rule)
    base.extend(init_cond_artist)
    for i in range(I):
        for j in range(0, I):
            base.append(ax.plot(*u_list[j], c='black')[0])
        base.append(ax.scatter(u_list[i][0][0], u_list[i][1][0], c='black', s=p_s))
        base.append(ax.scatter(u_list[i][0][-1], u_list[i][1][-1], c='black', s=p_s))

    if k != 7:
        frame = []
        frame.extend(base)
        add_artist(frame, get_comment_artist('Black line: converged intervals\nRed dots: current initial conditions'+f' $U_{"i"}^{k+1}$'))
        start.append(frame)  # this shows what has converged until now
        start.append(frame)
        start.append(frame)
        start.append(frame)
    
    # Catching early_stopping behavior, plot the converged ones together with the fine
    if k == 6:
        print(I, p.N)
    if I == p.N:
        print('here')
        iter_frames.extend(start)
        frame = []
        frame.extend(base)
        frame.append(ax.plot(*u_list[-1], c=c[-1], lw=1)[0])
        add_artist(frame, get_comment_artist('Algorithm has converged'))
        iter_frames.append(frame)
    else:
        # k=0 i=6 should be paused
        for i in range(I, p.N):
            curr_frame, base = show_one_update(base, t, uG, uF, u, i, k, ax)
            add_artist(curr_frame, get_interval_artist(i+1))
            add_artist(curr_frame, get_parall_artist(parall=False))
            if k != 0 or i != 5:
                add_artist(curr_frame, get_comment_artist('Updating initial conditions using $\mathscr{G}$'))
            else:
                add_artist(curr_frame, get_comment_artist('Explanation: run $\mathscr{G}$ (red line) from the updated \ninitial condition '+f' $U_{i}^{k+2}$'+' (green dot) and get an \napproximate initial condition for interval 6 \n  (gray dot,'+' $\mathscr{G}$' +f'$(U_{i}^{k+2})$'+'). Correct this using the difference\n    between '+'$\mathscr{F}$' +f'$(U_{i}^{k+1})$'+' (blue dot) and '+'$\mathscr{G}$' +f'$(U_{i}^{k+1})$'+'\n        (red dot) previously computed. This gives \n            the bottom green dot,'+f' $U_{i+1}^{k+2}$'+'. Now run $\mathscr{G}$ \n                again from this point. Repeat.',font_size=10))
                start.append(curr_frame)
                start.append(curr_frame)
                start.append(curr_frame)
            start.append(curr_frame)
        add_artist(base, get_comment_artist(f'End of iteration {k+1}. \nBlack line: converged intervals\nRed dots: previous initial conditions\n  Green dots: updated initial conditions', font_size=11))
        start.append(base)
        start.append(base)
        start.append(base)
        start.append(base)
        iter_frames.extend(start)


    iter_text = get_iteration_artist(k+1)


    for idx, frame in enumerate(iter_frames):
        # print(idx)
        frame.append(truth_artist)
        frame.append(iter_text)
        frame.append(explan_u)
    all_frames.extend(iter_frames)
    # all_frames.append([truth_artist])

    # # cols = matplotlib.cm.get_cmap('Blues')
    # # c = cols(np.linspace(0.3, 0.99, t.shape[0]-1))
    # # new_artist = ax.scatter(*uG[:-1,:,0].T, c=c)
    # # all_frames.append([truth_artist,new_artist])

    

    # plot_evolution(uG[...,0], s.run_G_full, t, ax, fine=False)
    if k == 1:
        break
last_frame = []
last_frame.append(truth_artist)
last_frame.append(iter_text)
last_frame.append(explan_u)
# F_full = []
# for i in range(p.N):
#     temp = s.run_F_full(t[i], t[i+1],u[i,:,-1])
#     F_full.append(temp)
#     temp = temp[-1,:]
for i, segment in enumerate(F_full):
    last_frame.append(ax.plot(np.linspace(t[i], t[i+1], num=segment.shape[0]), segment, c='black', lw=2, alpha=0.6)[0])
    last_frame.append(ax.scatter(t[i], segment[0,:], c='black', s=p_s))
    last_frame.append(plot_one_init_cond_label(t[i], segment[0,:], i, ax, k+1))
add_artist(last_frame, get_comment_artist('All initial conditions $U_i$ have stabilized.\nParareal has converged.'))
all_frames.append(last_frame)


# # all_frames = all_frames[:2]
# ani = animation.ArtistAnimation(fig=fig, artists=all_frames, interval=400)
# plt.show()
# ani.save('parareal_1D_nocomm.mp4')

# del ani

def save_single_frame(fig, arts, frame_number):
    # make sure everything is hidden
    for frame_arts in arts:
        for art in frame_arts:
            art.set_visible(False)
    # make the one artist we want visible
    for art in arts[frame_number]:
        art.set_visible(True)
    # arts[frame_number][artist_number].set_visible(True)
    fig.savefig(os.path.join('img',  f'paper_frame_{frame_number}.pdf'))

from matplotlib.lines import Line2D
lines = [Line2D([0], [0], color=c) for c in ['red','gray']]
labels = [r'$\mathscr{G}(U_i^0)$','Truth']
out = ax.legend(lines, labels, fontsize=18, loc='upper right')
all_frames[17].append(out)
# lines = [Line2D([0], [0], color=c) for c in ['gray','blue','red']]
# lines.append(Line2D([], [], color='#1f77b4', marker='|', linestyle='None',
#                           markersize=10, markeredgewidth=1.5, label='Vertical line', c='black'))
# labels = ['Truth', r'$\mathscr{F}(U_i^0)$', r'$\mathscr{G}(U_i^1)$', r'$(\mathscr{F}-\mathscr{G})(U_i^0)$']
# out = ax.legend(lines, labels, fontsize=15, loc='upper right')
# all_frames[36].append(out)
save_single_frame(fig, all_frames, 17)

# paper_frame_17.pdf
# save_single_frame(fig, all_frames, 36)

# for i in range(len(all_frames)):
#     save_single_frame(fig, all_frames, i)

#%% 1D System evolution plot - Panel B

from parareal import Parareal
from models import BareParareal, NNGP_p
import time
import matplotlib
import matplotlib.animation as animation
from systems import Lorenz, FHN_ODE, Brusselator, ODE
from solver import SolverRK
import numpy as np
from configs import Config
from matplotlib import pyplot as plt

class ParaMod(Parareal):
    

    def _parareal(self, debug=False, early_stop=None, parall='Serial', store_int=False, _load_mdl=False, **kwargs):
        tspan, N, epsilon, n = self.tspan, self.N, self.epsilon, self.n
        # f, u0 = self.f, self.u0
        solver: SolverAbstr = self.solver
        model = BareParareal(N=self.N, **kwargs)
                         
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
                
                yield _objs

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
            
                
                # self.store(path=os.path.join(int_dir, name_base), name=int_name, mdl=model, objs=_objs)
                
                
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

class Ode1d(ODE):
    def __init__(self, **kwargs):
        mn, mx = np.array([[0.1], [14700]])
        u0 = np.array([0.1]) 
        super().__init__('OneDim', mn, mx, u0, **kwargs)


    @staticmethod
    def _f_jax(t, u):
        return (-(t-5)*u*0.3)
    

    @staticmethod
    def _f_np(t, u):
        return (-(t-5)*u*0.3)

class MySolverRK(SolverRK):
    def run_F_full(self, t0, t1, u0):
        noise = np.arange(1001).reshape(-1,1)
        noise = noise/noise[-1,0]
        out = super().run_F_full(t0, t1, u0)
        # if t0 in [1]:
        #     out = out + 0.8*noise
        if t0 in [2]:
            out = out + 0.35*noise
        if t0 in [3]:
            out = out - 0.3*noise
        if t0 in [4]:
            out = out - 0.3*noise
        if t0 == 5:
            out = out - 0.1*noise
        if t0 == 6:
            out = out - 0.1*noise
        return out
    
    def run_F(self, t0, t1, u0):
        return self.run_F_full(t0, t1, u0)[-1,:]
    
N = 10
T = 10
ode = Ode1d()
f = ode.get_vector_field()
s = MySolverRK(f, Ng=2, Nf=1e3, G='RK1', F='RK8')
p = ParaMod(ode, s, tspan=(0, T), N=N)


tspan = p.tspan
t0 = 0
N = p.N
u0 = p.u0

p_s = 20 #point size


def gen_full_traj(solver, t, init_cond):
        u_full = list()
        for i in range(t.shape[0] - 1):
            temp = solver(t[i], t[i+1], init_cond[i, :])
            u_full.append((np.linspace(t[i], t[i+1], num=temp.shape[0]), temp))
        # out = np.vstack(u_full)
        return u_full, ''

def do_one_plot(data, until_i, c, ax, k, no_points=False, col=None):
    artists = list()
    for i in range(until_i):
        artists.append(ax.plot(*data[i], c=c[i], lw=1)[0])
        if not no_points:
            if col is None:
                col = c[i]
            artists.append(ax.scatter(data[i][0][0], data[i][1][0,:], c=col, s=p_s, zorder=1000))
            # artists.append(ax.text(data[i][0][0], data[i][1][0,:]-0.2, f'$U_{i}$', fontsize=10))
            artists.append(plot_one_init_cond_label(data[i][0][0], data[i][1][0,:], i, ax, k))
    return artists


def get_colors(t, fine):
    if fine:
        cols = matplotlib.cm.get_cmap('Blues')
        cols = ['blue' for i in range(t.shape[0]-1)]
    else:
        cols = matplotlib.cm.get_cmap('Reds')
        cols = ['red' for i in range(t.shape[0]-1)]
    # c = cols(np.linspace(0.3, 0.99, t.shape[0]-1))
    return cols
    # return c

def plot_evolution(init_cond, solver, t, ax, k, fine=False):
    container = []
    u_list, u_full = gen_full_traj(solver, t, init_cond)
    c = get_colors(t, fine)
    for i in range(1, t.shape[0]):
        if i == 1:
            artists = do_one_plot(u_list, i, c, ax, k, col='black')
            init_cond_artist = ax.scatter(t[i-1], init_cond[i-1,:], c='black', s=p_s, zorder=1000)
        else:
            artists = do_one_plot(u_list, i, c, ax, k)
            init_cond_artist = ax.scatter(t[i-1], init_cond[i-1,:], c=c[i-1], s=p_s, zorder=1000)
        artists.append(init_cond_artist)
        artists.append(get_interval_artist(i))
        container.append(artists)
    return container, u_list, c

def plot_evolution_oneshot(init_cond, solver, t, ax, k, fine=False):
    container = []
    u_list, u_full = gen_full_traj(solver, t, init_cond)
    c = get_colors(t, fine)
    i = t.shape[0]-1
    artists = do_one_plot(u_list, i, c, ax, k, no_points=True)
    # init_cond_artist = ax.scatter(*init_cond[i-1,:], c=c[i-1])
    # artists.append(init_cond_artist)
    container.append(artists)
    return container, u_list, c

def plot_init_cond(t, init_cond, ax, k: int, fine):
    c = get_colors(t, fine)
    artists = []
    artists.append(ax.scatter(t[:-1], init_cond[:-1], c=c, s=p_s, zorder=999))
    artists.extend(plot_all_init_cond_label(t, init_cond, range(t.shape[0]-1), ax, k=k))
    return artists

def plot_all_init_cond_label(t, init_cond, i_s, ax, k):
    artists = []
    for i in i_s:
        artists.append(plot_one_init_cond_label(t[i], init_cond[i], i, ax, k))
    return artists

def plot_one_init_cond_label(t, init_cond, i, ax, k):
    fts = 18
    if k is None:
        return ax.text(t, init_cond-0.3, f'$U_{i}$', fontsize=fts)
    if k == 1:
        if i == 2:
            return ax.text(t-0.3, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
        if i == 3:
            return ax.text(t-0.4, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
    if k == 0:
        if i == 4:
            return ax.text(t-0.3, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
        if i == 5:
            return ax.text(t-0.3, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
        if i == 6:
            return ax.text(t+0.1, init_cond, f'$U_{i}^{k}$', fontsize=fts)
    
    if i == 0 or i == 9:
        return ax.text(t, init_cond+0.2, f'$U_{i}^{k}$', fontsize=fts)
    elif i < 6:
        return ax.text(t, init_cond-0.4, f'$U_{i}^{k}$', fontsize=fts)
    else:
        return ax.text(t-0.3, init_cond-0.48, f'$U_{i}^{k}$', fontsize=fts)

def show_one_update(base, t, uG, uF, u, I, k, ax):
    init_cond = u[I,:,k+1]
    artists = []
    artists.extend(base)
    G_new = s.run_G_full(t[I], t[I+1], init_cond)
    assert np.allclose(G_new[-1,:], uG[I+1,:,k+1])
    preds = uF[I+1,:,k] + uG[I+1,:,k+1] - uG[I+1,:,k]
    if uF[I+1,:,k] > uG[I+1,:,k]:
        a7 = ax.plot([t[I+1],t[I+1]], [uF[I+1,:,k]-0.05, uG[I+1,:,k]+0.05], c='black')[0]
    else:
        a7 = ax.plot([t[I+1],t[I+1]], [uF[I+1,:,k]+0.05, uG[I+1,:,k]-0.05], c='black')[0]
    if preds > G_new[-1,:]:
        a8 = ax.plot([t[I+1], t[I+1]], (preds-0.05, G_new[-1,:]+0.05), c='black')[0]
    else:
        a8 = ax.plot([t[I+1], t[I+1]], (preds+0.05, G_new[-1,:]-0.05), c='black')[0]
    
    a1 = ax.scatter(t[I], init_cond, c='green', s=p_s, zorder=1000)
    a2 = ax.plot(np.linspace(t[i], t[I+1], num=G_new.shape[0]), G_new, c='red', lw=1)[0]
    a3 = ax.scatter(t[I+1], G_new[-1,:], c='gray', s=p_s, zorder=1000)
    a4 = ax.scatter(t[I+1], uG[I+1,:,k], c='red', s=p_s, zorder=1000)
    a5 = ax.scatter(t[I+1], uF[I+1,:,k], c='blue', s=p_s, zorder=1000)
    preds = uF[I+1,:,k] + uG[I+1,:,k+1] - uG[I+1,:,k]
    assert np.allclose(preds, u[I+1,:,k+1])
    a6 = ax.scatter(t[I+1], preds, c='green', s=p_s)
    a9 = plot_one_init_cond_label(t[I+1], preds, I+1, ax, k+1)
    
    artists.extend([a7, a8, a1, a2, a3, a4, a5, a6, a9])
    # artists.extend([ a1, a2, a3, a4, a5, a6, a9])

    base_next = []
    base_next.extend(base)
    # base_next.extend([a1, a6, a9])
    base_next.extend([a7, a8, a1, a2, a3, a4, a5, a6, a9])
    # base_next.extend([ a1, a2, a3, a4, a5, a6, a9])

    return artists, base_next

def add_artist(base, artist):
    if isinstance(base[0], list):
        for frame in base:
            if isinstance(artist, list):
                frame.extend(artist)
            else:
                frame.append(artist)
    else:
        base.append(artist)

def get_interval_artist(i):
    return ax.text(0,0,'')
    return ax.text(3.5, 4.9, f'Interval: {i:2d}', fontsize=12)

def get_iteration_artist(k):
    return ax.text(0, 4.7, f'Iteration: {k:2d}', fontsize=16)

def get_parall_artist(parall=False):
    return ax.text(0,0,'')
    if parall:
        return ax.text(7, 4.9, 'Parallel, running $\mathscr{F}$', fontsize=12, color='green')
    else:
        return ax.text(7, 4.9, 'Sequential, running $\mathscr{G}$', fontsize=12, color='red')

def get_comment_artist(comment, font_size=12):
    return ax.text(0,0,'')
    return ax.text(6.3, 4.6, comment, fontsize=font_size, color='black',verticalalignment='top')


ss = SolverRK(f, Ng=2, Nf=1e3, G='RK1', F='RK8')
ss.Ng = s.Ng * p.N
ss.Nf = s.Nf * p.N
truth = ss.run_F_full(*tspan, p.u0)[::10,:]
truth_x = np.linspace(*tspan, num=truth.shape[0])
all_frames = []
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Time $t$ - Panel B', fontsize=16)
ax.set_ylabel('Solution $u(t)$', fontsize=16)
ax.tick_params(axis='both', labelsize=13)
for k, objs in enumerate(p._parareal(store_int=True)):
    uG = objs['uG']
    uF = objs['uF']
    u = objs['u']
    t = objs['t']
    I = objs['I']

    truth_artist = ax.plot(truth_x, truth, c='gray', lw=1, alpha=0.6)[0]


    iter_frames = []

    # explan_u = ax.text(-0.4, 4.6, '$U_{i}^{k}$: Initial condition at\n iteration $k$ for interval $i$.', fontsize=12, color='black',verticalalignment='top')
    explan_u = ax.text(0,0,'')

    # update_rule = ax.text(3.2, 0.25, '$U_{i}^{k+1}=\mathscr{G}(U_{i-1}^{k+1})+\mathscr{F}(U_{i-1}^{k})-\mathscr{G}(U_{i-1}^{k})$', fontsize=12, color='black',verticalalignment='top')
    update_rule = ax.text(0,0,'')

    # Initialization: show G
    if k == 0:
        frame = []
        frame.append(truth_artist)
        frame.append(ax.scatter(t0, u0, c='black', zorder=1000))
        frame.append(get_comment_artist(f'True solution for 1D ODE system.\nThe initial condition is given (black dot).\n$u_0={u0[0]}$'))
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)

        # showing the intervals
        frame = []
        F_full = []
        temp = u0
        for i in range(p.N):
            temp = s.run_F_full(t[i], t[i+1],temp)
            F_full.append(temp)
            temp = temp[-1,:]

        interval_bands = []
        for i, segment in enumerate(F_full):
            if i % 2 == 0:
                c = 'black'
            else:
                c = 'gray'
            frame.append(ax.plot(np.linspace(t[i], t[i+1], num=segment.shape[0]), segment, c=c, lw=2, alpha=0.6)[0])
            frame.append(ax.scatter(t[i], segment[0,:], c='black', s=10, zorder=1000))
            interval_bands.append(ax.axvline(t[i+1], lw=1, alpha=0.4))
            add_artist(frame, plot_one_init_cond_label(t[i], segment[0,:], i, ax, k=None))
        interval_bands.append(ax.axvline(t[0], lw=1, alpha=0.4))
        
        frame.append(get_comment_artist(f'Divide the timespan into $N={N}$ intervals.\n The true intervals are shown.'))
        frame.append(ax.scatter(t0,u0, c='black', zorder=1000))
        frame.extend(interval_bands)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)
        all_frames.append(frame)


        container, *_ = plot_evolution(uG[...,0], s.run_G_full, t, ax, k, fine=False)
        iter_text = get_iteration_artist(0)
        add_artist(container, truth_artist)
        add_artist(container, iter_text)
        add_artist(container, interval_bands)
        add_artist(container, explan_u)
        add_artist(container, get_comment_artist('Initialization: building approximate \ninitial conditions for each\ninterval using $\mathscr{G}$.'))
        add_artist(container, get_parall_artist(parall=False))
        all_frames.extend(container)
    

    
    
    # Show Approximate initial conditions
    frame = []
    init_cond_artist = plot_init_cond(t, u[...,k], ax, k, fine=False)
    frame.extend(init_cond_artist)
    if k == 0:
        frame.append(get_comment_artist('Approximate initial conditions\n'+f'$U_{"i"}^{k+1}$'+' from $\mathscr{G}$'))
    else:
        frame.append(get_comment_artist('Updated initial conditions '+f'$U_{"i"}^{k+1}$'))
    iter_frames.append(frame)
    iter_frames.append(frame)
    iter_frames.append(frame)
    iter_frames.append(frame)


    base = []
    # Run F in parallel
    container, u_list, c = plot_evolution_oneshot(u[...,k], s.run_F_full, t, ax, k, fine=True)
    container[0].extend(init_cond_artist)
    base.extend(container[0])
    add_artist(container, get_parall_artist(parall=True))
    if k == 0:
        add_artist(container, get_comment_artist('Running $\mathscr{F}$ in parallel from the \napproximate initial conditions'+f' $U_{"i"}^{k+1}$'))
    else:
        add_artist(container, get_comment_artist('Running $\mathscr{F}$ in parallel from the \nupdated initial conditions'+f' $U_{"i"}^{k+1}$'))
    iter_frames.extend(container)
    iter_frames.extend(container)
    iter_frames.extend(container)
    iter_frames.extend(container)



    start = []
    # Add previously converged intervals
    # Add newly converged interval

    # base = []
    add_artist(base, update_rule)
    base.extend(init_cond_artist)
    for i in range(I):
        for j in range(0, I):
            base.append(ax.plot(*u_list[j], c='black')[0])
        base.append(ax.scatter(u_list[i][0][0], u_list[i][1][0], c='black', s=p_s, zorder=1000))
        base.append(ax.scatter(u_list[i][0][-1], u_list[i][1][-1], c='black', s=p_s, zorder=1000))

    if k != 7:
        frame = []
        frame.extend(base)
        add_artist(frame, get_comment_artist('Black line: converged intervals\nRed dots: current initial conditions'+f' $U_{"i"}^{k+1}$'))
        start.append(frame)  # this shows what has converged until now
        start.append(frame)
        start.append(frame)
        start.append(frame)
    
    # Catching early_stopping behavior, plot the converged ones together with the fine
    if k == 6:
        print(I, p.N)
    if I == p.N:
        print('here')
        iter_frames.extend(start)
        frame = []
        frame.extend(base)
        frame.append(ax.plot(*u_list[-1], c=c[-1], lw=1)[0])
        add_artist(frame, get_comment_artist('Algorithm has converged'))
        iter_frames.append(frame)
    else:
        # k=0 i=6 should be paused
        for i in range(I, p.N):
            curr_frame, base = show_one_update(base, t, uG, uF, u, i, k, ax)
            add_artist(curr_frame, get_interval_artist(i+1))
            add_artist(curr_frame, get_parall_artist(parall=False))
            if k != 0 or i != 5:
                add_artist(curr_frame, get_comment_artist('Updating initial conditions using $\mathscr{G}$'))
            else:
                add_artist(curr_frame, get_comment_artist('Explanation: run $\mathscr{G}$ (red line) from the updated \ninitial condition '+f' $U_{i}^{k+2}$'+' (green dot) and get an \napproximate initial condition for interval 6 \n  (gray dot,'+' $\mathscr{G}$' +f'$(U_{i}^{k+2})$'+'). Correct this using the difference\n    between '+'$\mathscr{F}$' +f'$(U_{i}^{k+1})$'+' (blue dot) and '+'$\mathscr{G}$' +f'$(U_{i}^{k+1})$'+'\n        (red dot) previously computed. This gives \n            the bottom green dot,'+f' $U_{i+1}^{k+2}$'+'. Now run $\mathscr{G}$ \n                again from this point. Repeat.',font_size=10))
                start.append(curr_frame)
                start.append(curr_frame)
                start.append(curr_frame)
            start.append(curr_frame)
        add_artist(base, get_comment_artist(f'End of iteration {k+1}. \nBlack line: converged intervals\nRed dots: previous initial conditions\n  Green dots: updated initial conditions', font_size=11))
        start.append(base)
        start.append(base)
        start.append(base)
        start.append(base)
        iter_frames.extend(start)


    iter_text = get_iteration_artist(k+1)


    for idx, frame in enumerate(iter_frames):
        # print(idx)
        frame.append(truth_artist)
        frame.append(iter_text)
        frame.append(explan_u)
    all_frames.extend(iter_frames)
    # all_frames.append([truth_artist])

    # # cols = matplotlib.cm.get_cmap('Blues')
    # # c = cols(np.linspace(0.3, 0.99, t.shape[0]-1))
    # # new_artist = ax.scatter(*uG[:-1,:,0].T, c=c)
    # # all_frames.append([truth_artist,new_artist])

    

    # plot_evolution(uG[...,0], s.run_G_full, t, ax, fine=False)
    if k == 1:
        break
last_frame = []
last_frame.append(truth_artist)
last_frame.append(iter_text)
last_frame.append(explan_u)
# F_full = []
# for i in range(p.N):
#     temp = s.run_F_full(t[i], t[i+1],u[i,:,-1])
#     F_full.append(temp)
#     temp = temp[-1,:]
for i, segment in enumerate(F_full):
    last_frame.append(ax.plot(np.linspace(t[i], t[i+1], num=segment.shape[0]), segment, c='black', lw=2, alpha=0.6)[0])
    last_frame.append(ax.scatter(t[i], segment[0,:], c='black', s=p_s))
    last_frame.append(plot_one_init_cond_label(t[i], segment[0,:], i, ax, k+1))
add_artist(last_frame, get_comment_artist('All initial conditions $U_i$ have stabilized.\nParareal has converged.'))
all_frames.append(last_frame)


# # all_frames = all_frames[:2]
# ani = animation.ArtistAnimation(fig=fig, artists=all_frames, interval=400)
# plt.show()
# ani.save('parareal_1D_nocomm.mp4')

# del ani

def save_single_frame(fig, arts, frame_number):
    # make sure everything is hidden
    for frame_arts in arts:
        for art in frame_arts:
            art.set_visible(False)
    # make the one artist we want visible
    for art in arts[frame_number]:
        art.set_visible(True)
    # arts[frame_number][artist_number].set_visible(True)
    fig.savefig(os.path.join('img', f'paper_frame_{frame_number}.pdf'))

from matplotlib.lines import Line2D
# lines = [Line2D([0], [0], color=c) for c in ['red','gray']]
# labels = [r'$\mathscr{G}(U_i^0)$','Truth']
# out = ax.legend(lines, labels, fontsize=18, loc='upper right')
# all_frames[17].append(out)
lines = [Line2D([0], [0], color=c) for c in ['gray','blue','red']]
lines.append(Line2D([], [], color='#1f77b4', marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1.5, label='Vertical line', c='black'))
labels = ['Truth', r'$\mathscr{F}(U_i^0)$', r'$\mathscr{G}(U_i^1)$', r'$(\mathscr{F}-\mathscr{G})(U_i^0)$']
out = ax.legend(lines, labels, fontsize=18, loc='upper right')
all_frames[36].append(out)
all_frames[36].append(ax.text(1-0.2, 0.6, f'$U_{1}^{1}$', fontsize=18))
# save_single_frame(fig, all_frames, 17)

# paper_frame_36.pdf
save_single_frame(fig, all_frames, 36)

# for i in range(len(all_frames)):
#     save_single_frame(fig, all_frames, i)

