from globals import *

############# All scalability results connected to FHN PDE ###########

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

    print(sys.argv, sys.argv[-3:])
    N, mdl, dx = sys.argv[-3:] 
    N = int(N)
    d_x = int(dx)
    d_y = d_x
    # N = 512        

    if d_x == 10:
        mul = 3
        T = 150
        G = 'RK2'
    elif d_x == 12:
        mul = 12
        T = 550
        G = 'RK2'
    elif d_x == 14:
        mul = 25
        T = 950
        G = 'RK2'
    elif d_x == 16:
        mul = 25
        T = 1100
        G = 'RK4'
    else:
        raise Exception('Invalid d_x val')
                            
    Ng =  N*mul                             
    Nf = int(np.ceil(1e8/Ng)*Ng)        
    F = 'RK8'                               
    epsilon = 5e-7  
    tspan = [0,T]

    d = 2*(d_x*d_y)                                 
    xspan = [-1,1]                           
                                

    # params
    dt = (tspan[1]-tspan[0])/Nf
    t_fine = np.linspace(tspan[0], tspan[-1], num=Nf+1)
    dx = (xspan[1]-xspan[0])/(d_x-1)
    dy = (xspan[1]-xspan[0])/(d_y-1)

    z1 = np.ones(d_x)
    Txx = np.diag(-2*z1)
    idxs = np.arange(d_x-1)
    Txx[idxs, idxs+1] = z1[:d_x-1]
    Txx[idxs+1, idxs] = z1[:d_x-1]
    Dxx = (1/(dx**2))*Txx

    z1 = np.ones(d_y)
    Tyy = np.diag(-2*z1)
    idxs = np.arange(d_y-1)
    Tyy[idxs, idxs+1] = z1[:d_y-1]
    Tyy[idxs+1, idxs] = z1[:d_y-1]
    Dyy = (1/(dy**2))*Tyy


    # boundary conditions (periodic)
    Dxx[0,-1] = 1/(dx**2)
    Dxx[-1,0] = 1/(dx**2)
    Dyy[0, -1] = 1/(dy**2)
    Dyy[-1,0] = 1/(dy**2)

    # construct differentiation matrices (using kronecker products)
    DXX = np.kron(np.eye(d_y,d_y),Dxx)
    DYY = np.kron(Dyy,np.eye(d_x,d_x))

    def operator(t,u,DXX,DYY):

        d = int(u.shape[0]/2)
        u1 = u[:d]
        u2 = u[d:]
        
        a = 2.8E-4
        b = 5E-3
        k = -5E-3
        tau = 0.1
        U = a*(DXX + DYY)@u1 + u1 - (u1**3) - u2 + k*jnp.ones(d)
        V = (1/tau)*( b*(DXX + DYY)@u2 + u1 - u2 )

        return jnp.hstack([U, V])

    def f_fhn(t, u):
        return operator(t,u,DXX,DYY)


    def f_fhn_n(t, u):
        mn, mx = jnp.array([[-1]*d, [1]*d])
        u = Systems._tr_inv(u, mn, mx)
        out = f_fhn(t, u)
        out = out * Systems._scale(mn, mx)
        return out


    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 1
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        # N, mdl = sys.argv[1:]
        # N = int(N)
        
        
        ########## CHANGE THIS #########
        dir_name = 'FHN_scal_times'
        ################################
        
        name = f'{dir_name}_{d_x}_{N}_{mdl}'
        int_name = name + '_int'
        assert workers >= N
        
        # generate folder
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        
        ########## CHANGE THIS #########
        scaling = 25
        
        ode_name='fhn_pde'
        f = f_fhn_n
        np.random.seed(45)
        u0 = np.random.rand(d)
        u0 = Systems._tr(u0, *jnp.array([[-1]*d, [1]*d]))
        
        
        s = Parareal(f=f, tspan=tspan, u0=u0, N=N, Ng=Ng, Nf=Nf, epsilon=epsilon, 
                    F=F, G=G, ode_name=ode_name)
        
        
        # s = Parareal(ode_name=f'non_aut{N}_n', normalization='-11', epsilon=5e-7)
        # s.Nf = s.Nf * 10000
        s.RK_thresh = s.Nf/s.N/scaling
        
        # # This is necessary as parall mpi requires the function to be pickable for Rk parallel eval
        # s.f = myf
        
        #####################################
        
        # run the code, storing intermediates in custom folder
        if mdl == 'para':
            res = s.run(pool=p, parall='mpi', store_int=True, int_dir=dir_name, int_name=int_name)
        elif mdl == 'gp':
            res = s.run(model='gpjax', pool=p, parall='mpi', store_int=True, int_dir=dir_name, int_name=int_name)
        elif mdl == 'nngp':
            res = s.run(model='nngp', pool=p, parall='mpi', store_int=True, int_dir=dir_name, int_name=int_name, 
                        nn=20, calc_detail_avg=True, calc_parall_overhead=True)
        else:
            raise Exception('Unknown model type', mdl)
            
            
        # dump the final result
        s.store(name=name, path=dir_name)
        
    #%%%% Runner

    #!/bin/bash
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
    # for mdl in "nngp" "gp" "para"; do for dx in 10 12 14 16; do
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
    # srun python -u -m mpi4py.futures run_scal_"$file".py $1 $mdl $dx
    # #srun python -u -m mpi4py.futures run_scal_"$file".py $1 $mdl

    # exit 0

    # EOT
    # done
    # done

    # # bash run_scal.slurm 32

#%%% Analysis

#%%%% Tables

def get_exp_serial_c(T):
    exp_serial_c = []
    n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
    N = 512
    for mdl in ['para','gp','nngp']:
        name = f'FHN_scal_times_{T}_{N}_{mdl}'
        try:
            with open(os.path.join('FHN_scal_times', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            exp_serial_c.append(run['timings']['F_time_serial_avg']/run['k']*N)
        except Exception as e:
            print(e)
    exp_serial_c = np.array(exp_serial_c).mean()
    return exp_serial_c
        
N = 512
for T in [10,12,14,16]:
    runs = {}
    for mdl in ['para','gp','nngp']:
        name = f'FHN_scal_times_{T}_{N}_{mdl}'
        try:
            with open(os.path.join('FHN_scal_times', name), 'rb') as f:
                solver = pickle.load(f)
                runs.update(solver.runs)
                # print(solver.runs.keys())
        except Exception as e:
            # if mdl == 'gp':
            #     path = f'{name}_int'
            #     max_mdl = len(os.listdir(os.path.join('FHN_scal_times',path)))-1
            #     name = f'{path}_{max_mdl}'
            #     with open(os.path.join('FHN_scal_times', path, name), 'rb') as f:
            #         s = pickle.load(f)
                
            #     G = s.objs['G_time']
            #     k = max_mdl+1
            #     F = s.objs['F_time']
            #     mdl_time = s.mdl.pred_times.sum()
            #     diff = 48*3600-(F+mdl_time)
            #     print(f'Tomlab, N={N}, k={k}, G={G:.2e}, F/k={F/k:.2e}, mdl={mdl_time:.2e}, tot={F+mdl_time:.2e}, discrepancy={diff:.2e}, actual mdl={48*3600-F:.2e}, F/k={F/k:.2e},G/k={G/k:.2e}')
            print(e)
    solver.runs = runs
    Parareal.print_speedup(solver, md=False, fine_t = get_exp_serial_c(T), mdl_title='FitzHugh-Nagumo PDE')
        
        
#%%%% nSpeedup

def tool_append(d, k, val):
    l = d.get(k, list())
    l.append(val)
    d[k] = l
    
def tr_key(key):
    mdl, typ = key.split('_')
    mdl_d = {'gp':'GParareal','para':'Parareal','nngp':'nnGParareal'}
    typ_d = {'exp':r'theoretical $S_{\rm alg}$', 'act':r'empirical $S_{\rm emp}$','exprough':'Theoretical approx', 'ub': 'maximum'}
    appdx = {'gp':r'$S_{\rm GPara}^* = K_{\rm GPara}/N$', 'para':'', 'nngp':r'$S_{\rm nnGPara}^* = K_{{\rm nnGPara}}/N$'}
    if typ == 'ub':
        return f'{mdl_d[mdl]} {typ_d[typ]} {appdx[mdl]}'
    return f'{mdl_d[mdl]} {typ_d[typ]}'
    
n_restarts = 1
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
Ts = [10,12,14,16]
store = {}
N = 512
for T in Ts:
    for mdl in ['para', 'gp', 'nngp']:
        try:
            name = f'FHN_scal_times_{T}_{N}_{mdl}'
            with open(os.path.join('FHN_scal_times', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            run['timings']['F_time_serial_avg'] = run['timings']['F_time_serial_avg']/run['k']
            if mdl == 'gp':
                exp = calc_exp_speedup(run, calc_exp_gp_cost, n_cores=n_cores_d[N], N=N)
                upp_b = N/run['k']
                tool_append(store, mdl+'_ub', upp_b)
            elif mdl == 'nngp':
                exp = calc_exp_speedup(run, calc_exp_nngp_cost_precise_v1, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
                exp_rough = calc_exp_speedup(run, calc_exp_nngp_cost_rough, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
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
           
          

ds = [2*d**2 for d in Ts]
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(ds)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.plot(x[mask], y, label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1)
    ax.set_xticks(x)
ax.plot(x, np.array([1]*4), ls='dashed', c='black', lw=1, label='Fine solver')
ax.legend(prop={'size':7.8}, loc='upper left', frameon=False)
ax.set_xlabel('d')
ax.set_ylabel('Speed-up')
fig.tight_layout()

store_fig(fig, 'fhn_pde_speedup')


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
Ts = [10,12,14,16]
store = {}
N = 512
for T in Ts:
    for mdl in ['para', 'gp', 'nngp']:
        try:
            name = f'FHN_scal_times_{T}_{N}_{mdl}'
            with open(os.path.join('FHN_scal_times', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            run['timings']['F_time_serial_avg'] = run['timings']['F_time_serial_avg']/run['k']
            if mdl == 'gp':
                exp = calc_exp_speedup(run, calc_exp_gp_cost, n_cores=n_cores_d[N], N=N)
                upp_b = N/run['k']
                tool_append(store, mdl+'_ub', upp_b)
            elif mdl == 'nngp':
                exp = calc_exp_speedup(run, calc_exp_nngp_cost_precise_v1, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
                exp_rough = calc_exp_speedup(run, calc_exp_nngp_cost_rough, n_cores=n_cores_d[N], N=N, n_restarts=n_restarts)
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
    
def get_propr(mdl, typ):
    c = {'gp':'red','para':'gray','nngp':'blue'} 
    props = {'exp':{'marker':'x', 'c':c[mdl]},
         'act':{'marker':'o', 'facecolors':'none', 'edgecolor':c[mdl]},
            'ub':{'marker':'1', 'c':c[mdl]}}
    return props[typ]

ds = [2*d**2 for d in Ts]
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
mrkr = {'exp':'circle', 'act':'x','exprough':'dotted', 'ub':(0,(1,10))}    
props = {'exp':{'marker':'x', 'facecolor':None,'edgecolor':None},
         'act':{'marker':'o', 'facecolors':'none', 'edgecolor':'r'},
            'ub':{'marker':'1', 'facecolor':None,'edgecolor':None}}
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots(figsize=(7,3))
for key, val in store.items():
    mdl, typ = key.split('_')
    x = np.array(ds)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.scatter(x[mask], y,  label=tr_key(key),**get_propr(mdl, typ))
    # ax.scatter(x[mask], y,  label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1, **props[typ])
    ax.set_xticks(x)
ax.scatter(x, np.array([1]*4),marker=1, c='black', label='Fine solver')
# ax.legend(prop={'size':7.8}, loc='upper left', frameon=False)
ax.legend(bbox_to_anchor=(1.05, 1.07), loc='upper left')
ax.set_xlabel('System dimension $d$')
ax.set_ylabel('Speed-up $S$')
fig.tight_layout()

store_fig(fig, 'fhn_pde_speedup_upd')



ds = [2*d**2 for d in Ts]
ls = {'exp':'dashed', 'act':'solid','exprough':'dotted', 'ub':(0,(1,10))}    
mrkr = {'exp':'circle', 'act':'x','exprough':'dotted', 'ub':(0,(1,10))}    
props = {'exp':{'marker':'x', 'facecolor':None,'edgecolor':None},
         'act':{'marker':'o', 'facecolors':'none', 'edgecolor':'r'},
            'ub':{'marker':'1', 'facecolor':None,'edgecolor':None}}
c = {'gp':'red','para':'gray','nngp':'blue'}    
fig, ax = plt.subplots()
for key, val in store.items():
    mdl, typ = key.split('_')
    if 'exp' in key and 'nngp' not in key:
        continue
    x = np.array(ds)
    y = np.array(val)
    mask = np.arange(y.shape[0])
    ax.scatter(x[mask], y,  label=tr_key(key),**get_propr(mdl, typ))
    # ax.scatter(x[mask], y,  label=tr_key(key), linestyle=ls[typ], c=c[mdl], lw=1, **props[typ])
    ax.set_xticks(x)
ax.scatter(x, np.array([1]*4),marker=1, c='black', label='Fine solver')
ax.legend(prop={'size':7.8}, loc='upper left', frameon=False)
ax.set_xlabel('System dimension $d$')
ax.set_ylabel('Speed-up $S$')
fig.tight_layout()

store_fig(fig, 'fhn_pde_speedup_upd_simpl')



