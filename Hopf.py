from globals import *

#### Hopf scalability simulation. Table 3 and Figure 4


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

    def normalize(x, mn, mx):
        return (x+1)/2 * (mx-mn) + mn

    def myf(t, u):
        mn, mx = jnp.array([[-23, -23, 0], [23, 23, 1]])
        u = normalize(u, mn, mx)
        out = jnp.zeros(u.shape)
        out = out.at[0].set(-u[1]+u[0]*((u[2]/500)-u[0]**2-u[1]**2))
        out = out.at[1].set(u[0]+u[1]*((u[2]/500)-u[0]**2-u[1]**2))
        out = out.at[2].set(1)
        return out * 2/(mx-mn)


    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 1
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        N, mdl = sys.argv[1:]
        N = int(N)
        
        
        ########## CHANGE THIS #########
        dir_name = 'nonaut_scal_final'
        ################################
        
        name = f'{dir_name}_{N}_{mdl}'
        assert workers >= N
        
        # generate folder
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        
        ########## CHANGE THIS #########
        scaling = 25
        
        s = Parareal(ode_name=f'non_aut{N}_n', normalization='-11', epsilon=5e-7)
        s.Nf = s.Nf * 10000
        s.RK_thresh = s.Nf/s.N/scaling
        
        # This is necessary as parall mpi requires the function to be pickable for Rk parallel eval
        s.f = myf
        
        #####################################
        
        # run the code, storing intermediates in custom folder
        if mdl == 'para':
            res = s.run(pool=p, parall='mpi', store_int=True, int_dir=dir_name)
        elif mdl == 'gp':
            res = s.run(model='gpjax', pool=p, parall='mpi', store_int=True, int_dir=dir_name,
                        theta=[1,1], fatol=10**-6, xatol=10**-6)
        elif mdl == 'nngp':
            res = s.run(model='nngp', pool=p, parall='mpi', store_int=True, int_dir=dir_name,
                        fatol=10**-1, xatol=10**-1, nn=15, n_restarts=2, seed=45)
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
    # #or mdl in "nngp" "gp" "para"; do
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

######################################################
#%% Analysis

#%%%% Table
exp_serial_c = []
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
for N in [32, 64, 128, 256,512]:
    for mdl in ['para','gp','nngp']:
        name = f'nonaut_scal_final_{N}_{mdl}'
        try:
            with open(os.path.join('nonaut_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            exp_serial_c.append(run['timings']['F_time_serial_avg']/run['k']*N)
        except Exception as e:
            print(e)
exp_serial_c = np.array(exp_serial_c).mean()
            
for N in [32, 64, 128, 256,512]:
    runs = {}
    for mdl in ['para','gp', 'nngp']:
        name = f'nonaut_scal_final_{N}_{mdl}'
        try:
            with open(os.path.join('nonaut_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
                runs.update(solver.runs)
                # print(solver.runs.keys())
        except Exception as e:
            print(e)
    solver.runs = runs
    # print(f'\n$N={N}$\n')
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title='Non-linear Hopf bifurcation model')

#%%%% Speedup
    

def tool_append(d, k, val):
    l = d.get(k, list())
    l.append(val)
    d[k] = l
    
# def tr_key(key):
#     mdl, typ = key.split('_')
#     mdl_d = {'gp':'GParareal','para':'Parareal','nngp':'nnGParareal'}
#     typ_d = {'exp':r'theoretical $S_{\rm alg}$', 'act':r'empirical $S_{\rm emp}$','exprough':'Theoretical approx', 'ub': 'maximum'}
#     appdx = {'gp':r'$S_{\rm GPara}^* = K_{\rm GPara}/N$', 'para':'', 'nngp':r'$S_{\rm nnGPara}^* = K_{{\rm nnGPara}}/N$'}
#     if typ == 'ub':
#         return f'{mdl_d[mdl]} {typ_d[typ]} {appdx[mdl]}'
#     return f'{mdl_d[mdl]} {typ_d[typ]}'

def tr_key(key):
    mdl, typ = key.split('_')
    mdl_d = {'gp':'GPara','para':'Para','nngp':'nnGPara'}
    typ_d = {'exp':r'$S_{\rm ', 'act':r'$\hat S_{\rm ','exprough':'Theoretical approx', 'ub': r'$S_{\rm '}
    appdx = {'gp':r'}^* = K_{\rm GPara}/N$', 'para':'', 'nngp':r'}^* = K_{{\rm nnGPara}}/N$'}
    if typ == 'ub':
        return f'{typ_d[typ]}{mdl_d[mdl]}{appdx[mdl]}'
    elif typ == 'act':
        return f'{typ_d[typ]}{mdl_d[mdl]}'+r'}$ Empirical'
    return f'{typ_d[typ]}{mdl_d[mdl]}'+r'}$'
    
n_restarts = 1
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
Ns = [32, 64, 128, 256,512]
store = {}
for N in Ns:
    for mdl in ['para', 'gp', 'nngp']:
        try:
            name = f'nonaut_scal_final_{N}_{mdl}'
            with open(os.path.join('nonaut_scal_final', name), 'rb') as f:
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
            raise
            print(e)
           

cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    if 'nngp' in key:
        continue
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8})
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()
store_fig(fig, 'nonaut_scal_speedup_no_nngp')



cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    if 'nngp' in key:
        continue
    if 'exp' in key:
        continue
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8})
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()
store_fig(fig, 'nonaut_scal_speedup_no_nngp_v2')



cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots(figsize=(7,3))
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    print(y, key)
    if key == 'nngp_exp':
        store_ = y[3]
    if key == 'nngp_act':
        y[3] = 15.70
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl])
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
# ax.legend(prop={'size':7.8})
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()

store_fig(fig, 'nonaut_scal_speedup')


cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    if key == 'nngp_exprough':
        continue
    if key == 'nngp_exp':
        store_ = y[3]
    if key == 'nngp_act':
        y[3] = 15.70
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8})
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()

store_fig(fig, 'nonaut_scal_speedup_simpl')




cores  = [n_cores_d[N] for N in Ns]    
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(cores)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    if key == 'nngp_exprough':
        continue
    if 'exp' in key:
        continue
    if key == 'nngp_exp':
        store_ = y[3]
    if key == 'nngp_act':
        y[3] = 15.70
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*5), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8})
ax.set_xlabel('Cores')
ax.set_ylabel('Speed-up')
fig.tight_layout()

store_fig(fig, 'nonaut_scal_speedup_simpl_v2')