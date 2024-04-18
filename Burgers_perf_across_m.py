from globals import *

############ Look at how Burgers performs in terms of K as m changes ############

#%% Code to run the simulations 

# prevents accidental runs
if False:

    #%%% Code
    from mpi4py.futures import MPIPoolExecutor
    import pickle
    import os
    from itertools import product, repeat
    import numpy as np
    from new_lib import *

    import jax 
    import jax.numpy as jnp
    from jax.config import config
    config.update("jax_enable_x64", True)
    import sys


    print(sys.argv, sys.argv[-2:]     )
    N, idx = sys.argv[-2:]      
    N = int(N)  
    idx = int(idx)

    T = 5                 
    Ng = N * 4                              
    Nf = Ng*500                    
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

    class ParaMod(Parareal):
        def run(self, *args, **kwargs):
            pool = kwargs.get('pool', None)
            if isinstance(pool, int):
                pool = concurrent.futures.ProcessPoolExecutor(max_workers=pool)
            elif isinstance(pool, type(None)):
                pool = MyPool()
            kwargs['pool'] = pool
            
            try:
                out = self._run(*args, **kwargs)
            except Exception as e:
                # pool.shutdown()
                raise
                
            # pool.shutdown()
            return out




    if __name__ == '__main__':
        
        avail_work = int(os.getenv('SLURM_NTASKS'))
        workers = avail_work - 1
        print('Total workes', workers)
        p = MPIPoolExecutor(workers)
        
        seeds = 100
        
        ins = list(product(np.random.choice(int(1e5), size=seeds, replace=False), np.arange(11, 31)))
        tot = len(ins)
        idxs = np.array(np.ceil(np.linspace(0, tot, 17)),dtype=int)
        ins = ins[idxs[idx]:idxs[idx+1]]
        
        out = []
        for seed, nn in ins:
            
            ode_name='Burg'
            f = f_burg_n
            u0 = 0.5*(np.cos(4.5*np.pi*x_fine) + 1)
            u0 = Systems._tr(u0, *jnp.array([[0]*d, [1]*d]))
            
            s = ParaMod(f=f, tspan=tspan, u0=u0, N=N, Ng=Ng, Nf=Nf, epsilon=epsilon, 
                        F=F, G=G, ode_name=ode_name, verbose=None)
            res = [s.N, nn, seed]
            try:
                res_nn = s.run(model='nngp', pool=p, parall='mpi',nn=nn, seed=seed)
                res.extend([res_nn['k'], res_nn['timings']['runtime'], res_nn['timings']['F_time'], res_nn['timings']['mdl_tot_t']]) 
            except Exception as e:
                res.extend([0, 0,0,0])
                res.append(str(e))
            print(res)
            out.append(res)
        
        try:
            dirname = 'Burges_nngp_exp_val_speed'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            with open(os.path.join(dirname, f'{dirname}_{T}_{N}_{idx}'), 'wb') as f:
                pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            with open(os.path.join(f'{dirname}_{T}_{N}_{idx}'), 'wb') as f:
                pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
                
#%%% Analysis - Updated 

def get_bin_lab(bins, gp):
    out = [r'$K_{GPara} < K_{NN\text{--}GPara}'+fr' < {bins[0]}$']
    for i in range(1, len(bins)):
        # out.append(rf'${bins[i-1]} \leq' + ' K_{NN} ' + rf'< {bins[i]}$')
        out.append(r'$K_{NN\text{--}GPara} '+rf'={bins[i-1]}$' )
    out.append(r'$K_{NN\text{--}GPara}' + rf' \geq {bins[-1]}$')
    return out
        
def build_bins(df, bins, bins_lab):
    bins.insert(0, df.knn.min())
    bins.append(df.knn.max()+1)
    
    out_df = []
    for idx, gr in df.groupby('nn'):
        for i in range(len(bins)-1):
            mask = (gr.knn>=bins[i]) & (gr.knn < bins[i+1])
            bin_val = gr.loc[mask, 'count'].sum()/gr['count'].sum()
            out_df.append([idx, bins_lab[i], bin_val])
    out_df = pd.DataFrame.from_records(out_df)
    out_df.columns = ['nn','bin','count']
    return out_df.set_index('nn')

def get_bin_lab_speed(bins, gp):
    out = [r'$S_{GPara} < S_{NN\text{--}GPara}'+rf' < {bins[0]}$']
    for i in range(1, len(bins)):
        out.append(rf'${bins[i-1]} \leq '+r'S_{NN\text{--}GPara}'+rf' < {bins[i]}$')
    out.append(r'$S_{NN\text{--}GPara}' +rf' \geq {bins[-1]} $' )
    return out
        
def build_bins_speed(df, bins, bins_lab):
    bins.insert(0, df.speedup.min())
    bins.append(df.speedup.max()+1)
    
    out_df = []
    for idx, gr in df.groupby('nn'):
        for i in range(len(bins)-1):
            mask = (gr.speedup>=bins[i]) & (gr.speedup < bins[i+1])
            bin_val = gr.loc[mask, 'speedup'].count()/gr['speedup'].count()
            out_df.append([idx, bins_lab[i], bin_val])
    out_df = pd.DataFrame.from_records(out_df)
    out_df.columns = ['nn','bin','count']
    return out_df.set_index('nn')



def do(T, bins_k, bins_speed, kgp, speed_gp):
    bdir = 'Burges_nngp_exp_val_speed'
    N_orig = 128
    N = N_orig
    res = []
    for idx in range(16):
        try:
            out = read_pickle(os.path.join(bdir, f'{bdir}_{T}_{N}_{idx}'))
            res.extend(out)
        except Exception as e:
            print(e)
        
    df = pd.DataFrame.from_records(res)
    df.columns = ['N','nn','seed','knn','t','Ft','mdl_t']
    
    # Iter to conv
    for idx, gr in df.groupby(['N']):
        N = idx[0]
        fig, ax = plt.subplots(figsize=(6,6))
        out = gr.groupby(['nn', 'knn']).agg(count=('N','count')).reset_index(level=1)
        out.pivot_table(values='count', index=out.index, columns='knn', fill_value=0).plot.bar(stacked=True)
        bins = bins_k
        bins_lab = get_bin_lab(bins, kgp)
        out = build_bins(out.reset_index(), bins, bins_lab)
        out_pv = out.pivot_table(values='count', index=out.index, columns='bin', fill_value=0)
        out_pv.reindex(sorted(out_pv.columns, key=lambda x: bins_lab.index(x)), axis=1).plot.bar(stacked=True, ax=ax, label=2)
        sum_filt = lambda knn: knn[knn<= kgp].count()/knn.count()
        temp = gr.groupby(['nn']).agg(count_filt=('knn',sum_filt)).reset_index()
        ax.plot(temp.count_filt, label='$K_{GPara}='+f'{kgp}$', c='white')
        ax.fill_between(np.arange(len(temp.count_filt)), [0]*len(temp.count_filt), temp.count_filt, hatch='/', alpha=0.0)
        ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
        ax.set_xlabel('m')
        ax.set_ylabel('Proportion')
        ax.set_title(f'$t_N={T}$')
        store_fig(fig, f'Burges_perf_across_m_{T}_K_updated')
        
        
        # Expected Speedup
        # Since it was run in parallel, we already have the paralle cost of mdl correctly estimated
        # We just need to add what we expect F to cost. Take this from available data
        N = N_orig
        name = f'Burges_scal_final_{T}_128_{"nngp"}'
        with open(os.path.join('Burges_scal_final', name), 'rb') as f:
            solver = pickle.load(f)
        run = solver.runs[list(solver.runs.keys())[0]]
        F_per_k = run['timings']['F_time']/run['k']

        exp_serial_c = []
        for N in [128]:
            for mdl in ['para','gp','nngp']:
                name = f'Burges_scal_final_{T}_128_{mdl}'
                try:
                    with open(os.path.join('Burges_scal_final', name), 'rb') as f:
                        solver = pickle.load(f)
                    run = solver.runs[list(solver.runs.keys())[0]]
                    exp_serial_c.append(run['timings']['F_time_serial_avg']/run['k']*N)
                except:
                    pass
        exp_serial_c = np.array(exp_serial_c).mean()
        
        # Iter to conv
        for idx, gr in df.groupby(['N']):
            N = idx[0]
            fig, ax = plt.subplots(figsize=(6,6))
            gr['speedup'] = exp_serial_c/(F_per_k*gr['knn'] + gr['mdl_t'])
            out = gr.groupby(['nn', 'knn']).agg(count=('N','count')).reset_index(level=1)
            print(gr.speedup.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
            # plt.figure()
            # plt.hist(gr.speedup, bins=40)
            
            # print(np.histogram(gr.speedup, bins=np.sort(gr.speedup.unique())))
            bins = bins_speed
            bins_lab = get_bin_lab_speed(bins, speed_gp)
            out = build_bins_speed(gr, bins, bins_lab)
            out_pv = out.pivot_table(values='count', index=out.index, columns='bin', fill_value=0)
            out_pv.reindex(sorted(out_pv.columns, key=lambda x: bins_lab.index(x)), axis=1).plot.bar(stacked=True, ax=ax, label=2)
            sum_filt = lambda knn: knn[knn<= speed_gp].count()/knn.count()
            temp = gr.groupby(['nn']).agg(count_filt=('speedup',sum_filt)).reset_index()
            ax.plot(temp.count_filt, label='$S_{GPara}='+f'{speed_gp}$', c='red')
            ax.fill_between(np.arange(len(temp.count_filt)), temp.count_filt, [1]*len(temp.count_filt), hatch='/', alpha=0.0)
            ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
            ax.set_xlabel('m')
            ax.set_ylabel('Proportion')
            ax.set_title(f'$t_N={T}$')
            store_fig(fig, f'Burges_perf_across_m_{T}_speedup_updated')


do(5, [8,9,10, 11], [11,12,13,14], 6, 7.94)
do(5.9, [13,14, 15, 16, 17], [7,7.5, 8,8.5], 8, 3.84)


