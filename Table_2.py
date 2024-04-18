from globals import *

##### Parareal system-wise performance (iterations to convergence) #####

#%% Code to run the simulations 

# prevents accidental runs
if False:

#%%% Server simulation, don't run

    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product, repeat
    from article_lib import *

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)


    def do(mdl, nn, epsilon):
        solver = Parareal(ode_name=mdl, epsilon=epsilon)
        res = solver.run(model='NNGP', nn=nn)
        res = solver.run(model='GPjax', fatol=10**-6, xatol=10**-6)
        res = solver.run()
        solver.plot()
        solver.data_tr = None
        solver.data_tr_inv = None
        return [solver, mdl, nn, epsilon]

    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 2
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        
        mdls = ['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n','dbl_pend_n'] + ['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n','dbl_pend_n']
        nns = [15, 15, 15, 14, 14, 15] + [13, 13, 12, 12, 13, 14]
        epsilons = [5e-7]*6 + [5e-9]*6
        
        res = list(p.map(do, mdls, nns, epsilons))
        
        with open(os.path.join('all_models'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
            
    # call srun python -m mpi4py.futures run.py

#%% Analysis


res = read_pickle('all_models')

# we can read data off the column K
# for solver, mdl, nn, epsilon in res:
#     print(f'mdl: {mdl}, epsilon: {epsilon}')
#     solver.print_times()
    
def build(data, eps=5e-7, do_plots=False):
    d = {'fhn_n':'FHN', 'lorenz_n':'Lorenz', 'rossler_long_n':'Rossler', 'non_aut32_n':'Hopf',
          'brus_2d_n':'Brusselator','dbl_pend_n':'Double Pendulum'}
    res = ['System&Parareal&GParareal&NN-GParareal\\\\']
    # res.append(['|---|---|---|---|'])
    for enu, (solver, system, nn, epsilon) in enumerate(data):
        if epsilon != eps:
            continue
        l = [d[system]]
        for mdl in ['Parareal', 'GP', 'NNGP']:
            l.append(str(solver.runs[mdl]['k']))
        solver.runs['GParareal'] = solver.runs['GP']
        solver.runs['NN-GParareal'] = solver.runs['NNGP']
        del solver.runs['NNGP']
        del solver.runs['GP']
        l_str = ' & '.join(l)
        res.append(l_str+'\\\\')
        if do_plots:
            solver.plot(cstm_title=d[system])
    print('\n'.join(res))
    
        
build(res, eps=5e-7, do_plots=False)
