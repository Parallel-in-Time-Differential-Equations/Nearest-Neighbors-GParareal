#%%

def mak_light_run(run, skipu=False):
    keys = ['u', 'x', 'D', 'data_x', 'data_D', 'uG', 'uF']
    if 'u' in run.keys():
        if not skipu:
            run['d'] = run['u'].shape[1]
    for k in keys:
        if k == 'u' and skipu:
            continue
        if k in run.keys():
            run.pop(k)
    return run

res = read_pickle('all_models')
i=0
for i in range(len(res)):
    solver = res[i][0]
    k = list(solver.runs.keys())[0]
    for k in solver.runs.keys():
        solver.fine = 0
        solver.runs[k].keys()
        skipu=True
        run = solver.runs[k]
        solver.runs[k] = mak_light_run(solver.runs[k], skipu=True)

store_pickle(res, 'all_models')