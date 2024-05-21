from globals import *
#### Distribution of K across different neighbour sizes m###

#%% Code to run the simulations 

# prevents accidental runs
if False:
    #%%%% All systems except double pendulum

    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product
    from article_lib import *

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)



    def do_gp(ins):
        (mdl, e_stop), restarts, tol, nn, epsilon, seed = ins
        
        pool = MPIPoolExecutor(1)
        solver = Parareal(ode_name=mdl, global_debug=False, normalization='-11', epsilon=epsilon)
        try:
            _ = solver.run(model='NNGP', cstm_mdl_name='NNGP', fatol=10**tol, xatol=10**tol, nn=nn, n_restarts=restarts, seed=seed, early_stop=e_stop, pool=pool)
            if solver.runs['NNGP']['converged']:
                res = [solver.ode_name, solver.runs['NNGP']['k'], epsilon, nn, restarts, tol, seed, solver.runs['NNGP']['timings']['runtime']]
            else:
                res = [solver.ode_name, solver.N, epsilon, nn, restarts, tol, seed, solver.runs['NNGP']['timings']['runtime']]
        except Exception as e:
            res = [solver.ode_name, solver.N, epsilon, nn, restarts, tol, str(e)]
            
        try:
            res_str = ','.join([str(i) for i in res])
            with open(os.path.join(f'sim_{mdl}_nngp.txt'), 'a') as w:
                w.write('\n')
                w.write(res_str)
        except Exception as e:
            pass
        return res

    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 2
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        
        epsilons = [5e-7, 5e-9]
        nns = ['adaptive', 11, 12, 13,14, 15, 16]
        tol = [-1]
        restarts = [1]
        mdls = ['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n']
        e_stops = [10, 18, 16, 24, 17]
        seeds = [45, 46, 47, 48, 49]
        ins = list(product(zip(mdls, e_stops), restarts, tol, nns, epsilons, seeds))
        
        
        res = list(p.map(do_gp, ins))
        
        with open(os.path.join('NNGP_all_but_pend'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
            
    #%%%% Simulation for double pendulum


    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product
    from article_lib import *

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)



    def do_gp(ins):
        epsilon, restarts, tol, coord1, coord2, nn = ins
        
        mdl = 'dbl_pend_n'
        solver = Parareal(ode_name=mdl, global_debug=False, normalization='-11', epsilon=epsilon, u0=[coord1, coord2,0,0])
        try:
            _ = solver.run(model='NNGP', cstm_mdl_name='NNGP', fatol=10**tol, xatol=10**tol, nn=nn, n_restarts=restarts, seed=45)
            res = [solver.ode_name, solver.runs['NNGP']['k'], epsilon, nn, restarts, tol, coord1, coord2, solver.runs['NNGP']['timings']['runtime']]
        except Exception as e:
            res = [solver.ode_name, solver.N, epsilon, nn, restarts, tol, coord1, coord2, str(e)]
            
        try:
            res_str = ','.join([str(i) for i in res])
            with open(os.path.join(f'sim_{mdl}_nngp.txt'), 'a') as w:
                w.write('\n')
                w.write(res_str)
        except Exception as e:
            pass
        return res

    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 2
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        
        epsilons = [5e-7, 5e-9]
        nns = ['adaptive', 11, 12, 13,14, 15]
        tol = [-1,-2,-3, -4]
        restarts = [1,2,3,4]
        coord1 = [-0.5]
        coord2 = [0]
        ins_fast = list(product(epsilons, restarts, tol, coord1, coord2, nns))
        
        epsilons = [5e-7, 5e-9]
        nns = ['adaptive', 11, 12, 13,14, 15]
        tol = [-5,-6]
        restarts = [1,2,3,4]
        coord1 = [-0.5]
        coord2 = [0]
        ins_slow = list(product(epsilons, restarts, tol, coord1, coord2, nns))
        
        ins = ins_fast + ins_slow
        
        res = list(p.map(do_gp, ins))
        
        with open(os.path.join('NNGP_pend'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
            
#%% Code to run the analysis

res = read_pickle('NNGP_pend')
df_nngp_pend = pd.DataFrame.from_records(res)
cols = ['mdl','k','eps','nn','rest','tol','seed','t']
df_nngp_pend.columns = cols 

res = read_pickle('NNGP_all_but_pend')
df_nngp = pd.DataFrame.from_records(res)
cols = ['mdl','k','eps','nn','rest','tol','seed','t']
df_nngp.columns = cols 

df = pd.concat([df_nngp, df_nngp_pend])

# titles = ['FHN', 'Rossler', 'Hopf', 'Brussellator', 'Lorenz', 'Double Pendulum']
# fig, axs = plt.subplots(2, 3, figsize=(10,6))
# for i,mdl in enumerate(['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n','dbl_pend_n']):
#     ax = axs[int(i/3), i%3]
#     a = df.loc[(df.mdl==mdl)&(df.eps == 5e-7)&(np.logical_not(np.isnan(df.t))),'k']
#     if len(a.unique() ) == 1:
#         ax.hist(a, density=True)
#     else:
#         # a.plot(kind='density', ax=ax)
#         # bins = np.arange(a.min(), a.max() + 1.5) - 0.5
#         # ax.hist(a, density=True, bins=bins)
#         # ax.set_xticks(bins + 0.5)
#         res = np.unique(a, return_counts=True)
#         # print(type(res[0]))
#         ax.bar(res[0], res[1]/res[1].sum())
    
#     ax.set_title(titles[i])
#     ax.set_ylabel('')
#     ax.set_xlabel('K')
# axs[0, 0].set_ylabel('Density')
# axs[1, 0].set_ylabel('Density')

titles = ['FHN', 'Rossler', 'Hopf', 'Brussellator', 'Lorenz', 'Dbl Pendulum']
# fig, axs = plt.subplots(2, 3, figsize=(10,6))
fig, axs = plt.subplots(1, 6, figsize=(12,3))
for i,mdl in enumerate(['fhn_n', 'rossler_long_n', 'non_aut32_n', 'brus_2d_n', 'lorenz_n','dbl_pend_n']):
    # ax = axs[int(i/3), i%3]
    ax = axs[i]
    a = df.loc[(df.mdl==mdl)&(df.eps == 5e-7)&(np.logical_not(np.isnan(df.t))),'k']
    if len(a.unique() ) == 1:
        if i == 0:
            ax.hist(a, density=True,bins=[3.5,4.5,5.5, 6.5])
            ax.set_xticks([4,5,6])
        else:
            ax.hist(a, density=True)
    else:
        # a.plot(kind='density', ax=ax)
        # bins = np.arange(a.min(), a.max() + 1.5) - 0.5
        # ax.hist(a, density=True, bins=bins)
        # ax.set_xticks(bins + 0.5)
        res = np.unique(a, return_counts=True)
        # print(type(res[0]))
        ax.bar(res[0], res[1]/res[1].sum())
        if i == 3:
            ax.set_xticks([16,17,18,19])
    
    ax.set_title(titles[i])
    ax.set_ylabel('')
    
fig.supxlabel(r'$K_{\rm nnGPara}$', fontsize=15)
fig.supylabel('Density', fontsize=15)
# axs[0].set_ylabel('Density')
fig.tight_layout()
for ax in axs:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] ):
        item.set_fontsize(16)
for ax in axs:
    for item in ([ ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(11)


# fig.savefig(os.path.join('img', 'nngp_m_distr'))
fig.savefig(os.path.join('img', 'nngp_m_distr_oneline.pdf'))