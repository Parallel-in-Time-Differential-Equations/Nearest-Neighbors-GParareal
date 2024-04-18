from globals import *

############# All scalability results connected to Viscous Burgers' equation ###########

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

    print(sys.argv, sys.argv[-2:]     )
    N, mdl = sys.argv[-2:]      
    N = int(N)       
    T = 5.9                 
    Ng = N * 4                              
    Nf = Ng*10000                    
    G = 'RK1'                          
    F = 'RK8'  
    d = N                                  
    xspan = [-1,1]                           
    tspan = [0,T]                            
    nu = 1/100    
    dt = (tspan[1]-tspan[0])/Nf
    t_fine = np.linspace(tspan[0], tspan[-1], num=Nf+1)
    dx = (xspan[1]-xspan[0])/(d-1);
    x_fine = np.linspace(xspan[0], xspan[-1], num=(d-1)+1)
    z0 = np.zeros(d)
    z1 = np.ones(d)
    Txx = np.diag(-2*z1)
    idxs = np.arange(d-1)
    Txx[idxs, idxs+1] = z1[:d-1]
    Txx[idxs+1, idxs] = z1[:d-1]
    Dxx = (nu/(dx**2))*Txx
    Tx = np.diag(z0)
    Tx[idxs, idxs+1] = z1[:d-1]
    Tx[idxs+1, idxs] = -z1[:d-1]
    Dx = (1/(2*dx))*Tx
    Dxx[0,-1] = 1*(nu/(dx**2))
    Dxx[-1,0] = 1*(nu/(dx**2))
    Dx[0, -1] = -1*(1/(2*dx))
    Dx[-1,0] = 1*(1/(2*dx))
    epsilon = 5e-7

    def f_burg(t, u):
        return Dxx@u - u*(Dx@u)

    def f_burg_n(t, u):
        mn, mx = jnp.array([[0]*d, [1]*d])
        u = Systems._tr_inv(u, mn, mx)
        out = f_burg(t, u)
        out = out * Systems._scale(mn, mx)
        return out

    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 1
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        # N, mdl = sys.argv[1:]
        # N = int(N)
        print(sys.argv)
        
        
        ########## CHANGE THIS #########
        dir_name = 'Burges_scal_final'
        ################################
        
        name = f'{dir_name}_{T}_{N}_{mdl}'
        int_name = name + '_int'
        assert workers >= N
        
        # generate folder
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        
        ########## CHANGE THIS #########
        scaling = 200
        
        ode_name='Burg'
        f = f_burg_n
        u0 = 0.5*(np.cos(4.5*np.pi*x_fine) + 1)
        u0 = Systems._tr(u0, *jnp.array([[0]*d, [1]*d]))
        
        s = Parareal(f=f, tspan=tspan, u0=u0, N=N, Ng=Ng, Nf=Nf, epsilon=epsilon, 
                    F=F, G=G, ode_name=ode_name, verbose='v')
        
        
        # s = Parareal(ode_name=f'non_aut{N}_n', normalization='-11', epsilon=5e-7)
        # s.Nf = s.Nf * 10000
        s.RK_thresh = s.Nf/s.N/scaling
        
        # # This is necessary as parall mpi requires the function to be pickable for Rk parallel eval
        # s.f = myf
        
        #####################################
        int_dir = dir_name
        # run the code, storing intermediates in custom folder
        if mdl == 'para':
            res = s.run(pool=p, parall='mpi', store_int=True, int_dir=int_dir, int_name=int_name)
        elif mdl == 'gp':
            res = s.run(model='gpjax', pool=p, parall='mpi', store_int=True, int_dir=int_dir, int_name=int_name)
        elif mdl == 'nngp':
            res = s.run(model='nngp', pool=p, parall='mpi', store_int=True, int_dir=int_dir, int_name=int_name,
                        nn=18)
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
    # for mdl in "nngp" "gp" "para"; do
    #     sbatch <<EOT
    # #!/bin/bash
    # #SBATCH --open-mode=append
    # #SBATCH -o "scal_out_file_"$1"_burg.txt"
    # #SBATCH -e "scal_out_file_"$1"_burg.txt"
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
    # # run with both T=5 and T=5.9

#%%% Analysis
from new_lib import Parareal

def do(T):
    N = 128
    exp_serial_c = []
    for mdl in ['para','gp','nngp']:
        name = f'Burges_scal_final_{T}_{N}_{mdl}'
        try:
            with open(os.path.join('Burges_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
            run = solver.runs[list(solver.runs.keys())[0]]
            exp_serial_c.append(run['timings']['F_time_serial_avg']/run['k']*N)
        except Exception as e:
            print(e)
    exp_serial_c = np.array(exp_serial_c).mean()
            

    runs = {}
    for mdl in ['para','gp', 'nngp']:
        name = f'Burges_scal_final_{T}_{N}_{mdl}'
        try:
            with open(os.path.join('Burges_scal_final', name), 'rb') as f:
                solver = pickle.load(f)
                runs.update(solver.runs)
                # print(solver.runs.keys())
        except Exception as e:
            print(e) 
    solver.runs = runs
    # print(f'\n$N={N}$\n')
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title='Viscous Burgers\' equation')
        
    
do(5)

do(5.9)

    

