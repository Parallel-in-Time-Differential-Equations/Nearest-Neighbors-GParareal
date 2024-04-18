from globals import *

############# All scalability results connected to Thomas Labirynth ###########

#%% Code to run the simulations 

# prevents accidental runs
if False:

    #%%%% Models
    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product, repeat
    from new_lib import *
    import sys

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)


    from itertools import product


    def ThomasLabyrinth(t, u):
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


    def ThomasLabyrinth_n(t, u):
        mn, mx = jnp.array([[-12, -12, -12], [12, 12, 12]])
        u = Systems._tr_inv(u, mn, mx)
        out = ThomasLabyrinth(t, u)
        out = out * Systems._scale(mn, mx)
        return out



    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 1
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        N, mdl = sys.argv[1:]
        N = int(N)
        dir_name = 'tomlab_scal_final'
        name = f'{dir_name}_{N}_{mdl}'
        
        assert workers >= N
        
        # generate folder
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        
        ########## CHANGE THIS #########
        if N == 32:
            tot_time = 10
        elif N == 64:
            tot_time = 10
        elif N == 128:
            tot_time = 40
        elif N == 256:
            tot_time = 100
        elif N == 512:
            tot_time = 100
        
        c_ng = 10
        fine_steps = 1e9
        
        tspan = [0, tot_time]
        u0 = np.array([4.6722764,5.2437205e-10,-6.4444208e-10])
        u0 = Systems._tr(u0, *jnp.array([[-12, -12, -12], [12, 12, 12]]))
        # N = 40
        Ng = N * c_ng
        Nf = Ng * int(np.ceil(fine_steps/Ng))
        epsilon = 5e-7
        F = 'RK4'
        G = 'RK1'
        ode_name='TomLab'
        
        scaling = 109
        
        s = Parareal(f=ThomasLabyrinth_n, tspan=tspan, u0=u0, N=N, Ng=Ng, Nf=Nf, epsilon=epsilon, 
                    F=F, G=G, ode_name=ode_name)
        s.RK_thresh = s.Nf/s.N/scaling
        
        # This is necessary as parall mpi requires the function to be pickable for Rk parallel eval
        s.f = ThomasLabyrinth_n
        
        #####################################
        
        # run the code, storing intermediates in custom folder
        if mdl == 'para':
            res = s.run(pool=p, parall='mpi', store_int=True, int_dir=dir_name)
        elif mdl == 'gp':
            res = s.run(model='gpjax', pool=p, parall='mpi', store_int=True, int_dir=dir_name,
                        fatol=10**-1, xatol=10**-1)
        elif mdl == 'nngp':
            res = s.run(model='nngp', pool=p, parall='mpi', store_int=True, int_dir=dir_name,
                        nn=18, n_restarts=1, fatol=10**-3, xatol=10**-3, seed=45)
        else:
            raise Exception('Unknown model type', mdl)
            
            
        # dump the final result
        s.store(name=name, path=dir_name)
        
    #%%%% Runner
    # #!/bin/bash
    # if [ $1 -eq 32 ]
    # then
    #     nodes=1
    # elif [ $1 -eq 64 ]
    # then
    #     nodes=2
    # elif [ $1 -eq 128 ]
    # then
    #     nodes=3
    # elif [ $1 -eq 256 ]
    # then
    #     nodes=6
    # elif [ $1 -eq 512 ]
    # then
    #     nodes=11
    # else
    #     echo "$1 Not valid"
    #     exit 0
    # fi
    # echo > scal_out_file_"$1".txt
    # file=$(date +%s)
    # cp "/home/maths/strkss/massi/python/run_scal.py" "/home/maths/strkss/massi/python/run_scal_"$file".py"
    # #for mdl in "nngp" "gp" "para"; do for dx in 10 12 14 16; do
    # #for mdl in "nngp" "gp" "para"; do
    #     sbatch <<EOT
    # #!/bin/bash
    # #SBATCH --open-mode=append
    # #SBATCH -o "scal_out_file_"$1".txt"
    # #SBATCH -e "scal_out_file_"$1".txt"
    # #SBATCH --nodes=$nodes
    # #SBATCH --ntasks-per-node=48
    # #SBATCH --cpus-per-task=1
    # #SBATCH --mem-per-cpu=3700
    # #SBATCH --time=48:00:00

    # module purge
    # module load GCC/11.3.0 OpenMPI/4.1.4 
    # module load GCCcore/11.3.0 Python/3.10.4
    # module load SciPy-bundle/2022.05
    # module load matplotlib/3.5.2
    # cd "/home/maths/strkss/massi/python"

    # # Multiprocess application
    # #srun python -u -m mpi4py.futures run_scal_"$file".py $1 $mdl $dx
    # srun python -u -m mpi4py.futures run_scal_"$file".py $1 $mdl

    # exit 0

    # EOT
    # done
    # #done

    # # bash run_scal.slurm 32

#%%% Analysis
from new_lib import *

#%%%% Gen Tables

exp_serial_c = []
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
for N in [32, 64, 128, 256,512]:
    for mdl in ['para','gp','nngp']:
        name = f'tomlab_scal_final_{N}_{mdl}'
        try:
            with open(os.path.join('tomlab_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            exp_serial_c.append(run['timings']['F_time_serial_avg']/run['k']*N)
        except:
            pass
exp_serial_c = np.array(exp_serial_c).mean()
        
for N in [32, 64, 128, 256,512]:
    runs = {}
    for mdl in ['para','gp', 'nngp']:
        name = f'tomlab_scal_final_{N}_{mdl}'
        try:
            with open(os.path.join('tomlab_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
                runs.update(solver.runs)
                # print(solver.runs.keys())
        except Exception as e:
            if mdl == 'gp':
                path = f'TomLab_{N}_GP_int'
                max_mdl = len(os.listdir(os.path.join('tomlab_scal_final',path)))-1
                name = f'TomLab_{N}_GP_int_{max_mdl}'
                with open(os.path.join('tomlab_scal_final', path, name), 'rb') as f:
                    s = pickle.load(f)
                
                G = s.objs['G_time']
                k = max_mdl+1
                F = s.objs['F_time']
                mdl_time = s.mdl.pred_times.sum()
                diff = 48*3600-(F+mdl_time)
                print(f'Tomlab, N={N}, k={k}, G={G:.2e}, F/k={F/k:.2e}, mdl={mdl_time:.2e}, tot={F+mdl_time:.2e}, discrepancy={diff:.2e}, actual mdl={48*3600-F:.2e}, F/k={F/k:.2e},G/k={G/k:.2e}')
    solver.runs = runs
    # print(f'\n$N={N}$\n')
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title='Thomas Labirinth')
    
#%%%% Gen Intermediate running time GP/NNGP plots

########## GP #################
out = []
for N in [32, 64, 128, 256, 512]:
    path = f'TomLab_{N}_GP_int'
    try:
        for k in range(len(os.listdir(os.path.join('tomlab_scal_final', path)))):
            name = f'TomLab_{N}_GP_int_{k}'
            with open(os.path.join('tomlab_scal_final', path, name), 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                s = pickle.load(f)
            mdl = s.mdl
            col = ['N','k','t','conv']
            if N in [32, 64]:
                out.append([N, k, mdl.pred_times[k], s.objs['I']])
            elif N in [128]:
                out.append([N, k, mdl.pred_times[k]/60, s.objs['I']])
            else:
                out.append([N, k, mdl.pred_times[k]/3600, s.objs['I']])
    except Exception as e:
        print('Missing', name, e)
    

# Those that didn't manage, plot the time evolution
df = pd.DataFrame.from_records(out)
df.columns = col

# fig, axs = plt.subplot_mosaic('001122;.3344.', figsize=(10,6))
# for idx, (lev, gr) in enumerate(df.groupby('N')):
#     ax = axs[str(idx)]
#     ax.plot(gr['k'], gr['t'])
#     N = gr["N"].iloc[0]
#     if N in [32, 64]:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}s')
#         ax.set_ylabel('Train time for iter $k$ (sec)', color='tab:blue')
#     elif N in [128]:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}m')
#         ax.set_ylabel('Train time for iter $k$ (min)', color='tab:blue')
#     else:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}h')
#         ax.set_ylabel('Train time for iter $k$ (h)', color='tab:blue')
#     ax.set_xlabel('k')
#     ax1 = ax.twinx()
#     ax1.set_ylabel('Converged intervals (%)', color='tab:red')
#     ax1.set_ylim((0,100))
#     ax1.plot(gr['k'], gr['conv']/gr["N"].iloc[0]*100, color='red')
# # fig.suptitle('GParareal')
# fig.tight_layout()

# store_fig(fig, 'tom_lab_scal_gp_int_runtime')

fig, axs = plt.subplot_mosaic('001122;.3344.', figsize=(10,6))
for idx, (lev, gr) in enumerate(df.groupby('N')):
    ax = axs[str(idx)]
    ax.plot(gr['k'], gr['t'], color='mediumblue')
    N = gr["N"].iloc[0]
    if N in [32, 64]:
        ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}s')
        ax.set_ylabel('Train time for iter $k$ (sec)', color='mediumblue', fontsize=11)
    elif N in [128]:
        ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}m')
        ax.set_ylabel('Train time for iter $k$ (min)', color='mediumblue', fontsize=11)
    else:
        ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}h')
        ax.set_ylabel('Train time for iter $k$ (h)', color='mediumblue', fontsize=11)
    ax.set_xlabel('k')
    ax1 = ax.twinx()
    
    ax1.set_ylim((0,100))
    ax1.plot(gr['k'], gr['conv']/gr["N"].iloc[0]*100, color='red')
    ax.tick_params(colors='mediumblue', axis='y')
    ax1.tick_params(colors='red', axis='y')
# fig.suptitle('GParareal')
ax1.set_ylabel('Converged intervals (%)', color='red', fontsize=11)
fig.tight_layout()

store_fig(fig, 'tom_lab_scal_gp_int_runtime_upd')


############### NNGP ####################

# Might be interesting to show the same for NNGP as well
out = []
for N in [32, 64, 128, 256, 512]:
    path = f'TomLab_{N}_NNGP_int'
    try:
        for k in range(len(os.listdir(os.path.join('tomlab_scal_final', path)))):
            name = f'TomLab_{N}_NNGP_int_{k}'
            with open(os.path.join('tomlab_scal_final', path, name), 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                s = pickle.load(f)
            mdl = s.mdl
            col = ['N','k','t','conv']
            if N in [32, 64, 128, 256, 512]:
                out.append([N, k, mdl.pred_times[k], s.objs['I']])
            elif N in None:
                out.append([N, k, mdl.pred_times[k]/60, s.objs['I']])
            else:
                out.append([N, k, mdl.pred_times[k]/3600, s.objs['I']])
    except Exception as e:
        print('Missing', name, e)
    

# Those that didn't manage, plot the time evolution
df = pd.DataFrame.from_records(out)
df.columns = col

# fig, axs = plt.subplot_mosaic('001122;.3344.', figsize=(10,6))
# for idx, (lev, gr) in enumerate(df.groupby('N')):
#     ax = axs[str(idx)]
#     ax.plot(gr['k'], gr['t'])
#     N = gr["N"].iloc[0]
#     if N in [32, 64, 128]:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}s')
#         ax.set_ylabel('Train time for iter $k$  (sec)', color='tab:blue')
#     elif N in [256, 512]:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum()/60:.0f}m')
#         ax.set_ylabel('Train time for iter $k$  (sec)', color='tab:blue')
#     else:
#         ax.set_title(f'N: {N}, Tot tain time: {gr["t"].sum():.0f}h')
#         ax.set_ylabel('Train time for iter $k$  (h)', color='tab:blue')
#     ax.set_xlabel('k')
#     ax1 = ax.twinx()
#     ax1.set_ylabel('Converged intervals (%)', color='tab:red')
#     ax1.set_ylim((0,100))
#     ax1.plot(gr['k'], gr['conv']/gr["N"].iloc[0]*100, color='red')
# # fig.suptitle('NN-GParareal')
# fig.tight_layout()

# store_fig(fig, 'tom_lab_scal_nngp_int_runtime')


fig, axs = plt.subplot_mosaic('001122;.3344.', figsize=(10,6))
for idx, (lev, gr) in enumerate(df.groupby('N')):
    ax = axs[str(idx)]
    ax.plot(gr['k'], gr['t'], color='mediumblue')
    N = gr["N"].iloc[0]
    if N in [32, 64, 128]:
        ax.set_title(f'N: {N}, Tot train time: {gr["t"].sum():.0f}s')
        ax.set_ylabel('Train time for iter $k$  (sec)', color='mediumblue', fontsize=11)
    elif N in [256, 512]:
        ax.set_title(f'N: {N}, Tot train time: {gr["t"].sum()/60:.0f}m')
        ax.set_ylabel('Train time for iter $k$  (sec)', color='mediumblue', fontsize=11)
    else:
        ax.set_title(f'N: {N}, Tot train time: {gr["t"].sum():.0f}h')
        ax.set_ylabel('Train time for iter $k$  (h)', color='mediumblue', fontsize=11)
    ax.set_xlabel('k')
    ax1 = ax.twinx()
    
    ax1.set_ylim((0,100))
    ax1.plot(gr['k'], gr['conv']/gr["N"].iloc[0]*100, color='red')
    ax.tick_params(colors='mediumblue', axis='y')
    ax1.tick_params(colors='red', axis='y')
# fig.suptitle('NN-GParareal')
ax1.set_ylabel('Converged intervals (%)', color='tab:red', fontsize=11)
fig.tight_layout()

store_fig(fig, 'tom_lab_scal_nngp_int_runtime_upd')


# For NNGP, they go down but that is because we have a N-k factor in there. 
# If we normalize by ther number of times we make a prediction, we expect 
# the cost to be constant

# Show normalized time for NNGP
fig, axs = plt.subplot_mosaic('001122;.3344.', figsize=(10,6))
for idx, (lev, gr) in enumerate(df.groupby('N')):
    ax = axs[str(idx)]
    N = gr["N"].iloc[0]
    ax.plot(gr['k'], gr['t']/(N-gr['conv']))
    if N in [32, 64, 128]:
        ax.set_title(f'N: {N}')
        ax.set_ylabel('Time (seconds)', color='tab:blue')
    elif N in [256, 512]:
        ax.set_title(f'N: {N}')
        ax.set_ylabel('Time (seconds)', color='tab:blue')
    else:
        ax.set_title(f'N: {N}, Tot time: {gr["t"].sum():.0f}h')
        ax.set_ylabel('Time (hours)', color='tab:blue')
    ax.set_xlabel('k')
    ax1 = ax.twinx()
    ax1.set_ylabel('Converged (%)', color='tab:red')
    ax1.set_ylim((0,100))
    ax1.plot(gr['k'], gr['conv']/gr["N"].iloc[0]*100, color='red')
# fig.suptitle('NN-GParareal - normalized time')
fig.tight_layout()
    
store_fig(fig, 'tom_lab_scal_nngp_int_runtime_norm')
   
    
#%%%% Speedup

def tool_append(d, k, val):
    l = d.get(k, list())
    l.append(val)
    d[k] = l
    
def tr_key(key):
    mdl, typ = key.split('_')
    mdl_d = {'gp':'GPara','para':'Para','nngp':'NN-GPara'}
    typ_d = {'exp':'Theoretical', 'act':'Actual','exprough':'Theoretical approx', 'ub': 'Upper bound'}
    appdx = {'gp':'$K_{GPara}/N$', 'para':'', 'nngp':'$K_{NN}/N$'}
    if typ == 'ub':
        return f'{mdl_d[mdl]} {typ_d[typ]} {appdx[mdl]}'
    return f'{mdl_d[mdl]} {typ_d[typ]}'
    
n_restarts = 1
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
Ns = [32, 64, 128, 256,512]
store = {}
for N in Ns:
    for mdl in ['para', 'gp', 'nngp']:
        try:
            name = f'tomlab_scal_final_{N}_{mdl}'
            with open(os.path.join('tomlab_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            run['timings']['F_time_serial_avg'] = run['timings']['F_time_serial_avg']/run['k']
            if mdl == 'gp':
                exp = calc_exp_speedup(run, calc_exp_gp_cost, n_cores=n_cores_d[N], N=N)
                upp_b = N/run['k']
                tool_append(store, mdl+'_ub', upp_b)
            elif mdl == 'nngp':
                exp = calc_exp_speedup(run, calc_exp_nngp_cost_precise_v1, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
                # exp_rough = calc_exp_speedup(run, calc_exp_nngp_cost_rough, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
                # tool_append(store, mdl+'_exprough', exp_rough)
                upp_b = N/run['k']
                tool_append(store, mdl+'_ub', upp_b)
            elif mdl =='para':
                exp = calc_exp_speedup(run, calc_exp_para_mdl_cost, n_cores=n_cores_d[N], N=N)
            else:
                pass
            act = calc_speedup(run, N=N)
            tool_append(store, mdl+'_exp', exp)
            tool_append(store, mdl+'_act', act)
            
            # upp_b = N/run['k']
            # tool_append(store, mdl+'_ub', upp_b)
        except Exception as e:
            print(e)
           

cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8}, loc='upper center')
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()

store_fig(fig, 'tom_lab_scal_speedup')

