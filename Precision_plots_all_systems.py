from globals import *

############# All precision and convergence plots for FHN ODE, Lorenz, Double Pendulum, Brussellator, Hopf Bifurcation, and Rossler ###########

#%% Code to run the simulations 

# prevents accidental runs
if False:
    pass
    # Code is the same as Table 2

class ParaMod(Parareal):
    pass


all_mdl_but_time = read_pickle('all_models')
time_mdls = read_pickle('nngptime_diff_subsets2')



# solver.plot()


for (solver, system, _, _), (solver_time, system_t, _, _) in zip(all_mdl_but_time, time_mdls):
    assert system == system_t
    print(system)
    if system == 'fhn_n':
        cstm_title = 'FitzHugh-Nagumo ODE'
    elif system == 'rossler_long_n':
        cstm_title = 'Rossler'
    elif system == 'non_aut32_n':
        cstm_title = 'Hopf Bifurcation'
    elif system == 'brus_2d_n':
        cstm_title = 'Brussellator'
    elif system == 'lorenz_n':
        cstm_title = 'Lorenz'
    elif system == 'dbl_pend_n':
        cstm_title = 'Double Pendulum'
    else:
        cstm_title = system
    solver.runs['GParareal'] = solver.runs['GP']
    solver.runs['nnGParareal'] = solver.runs['NNGP']
    solver.runs['nnGPara with Time'] = solver_time.runs['NNGPnn']    
    del solver.runs['NNGP']
    del solver.runs['GP']

    solver.plot(cstm_title=cstm_title)

    # break
    