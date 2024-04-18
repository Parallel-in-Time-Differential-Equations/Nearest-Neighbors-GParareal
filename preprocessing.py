### Make some data_stores lighter
import os
from globals import *

def mak_light_run(run):
    keys = ['u', 'x', 'D', 'data_x', 'data_D', 'uG', 'uF']
    if 'u' in run.keys():
        run['d'] = run['u'].shape[1]
    for k in keys:
        if k in run.keys():
            run.pop(k)
    return run

def make_lightweight(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            make_lightweight(os.path.join(path, file))
        return
    try:
        s = read_pickle(path)
        if len(s.runs) > 0:
            run =  s.runs[list(s.runs.keys())[0]]
        else:
            run = s.objs
            s.u0 = 'emptied'
            s.objs['conv_int'] = 'emptied'
            s.objs['kwargs'] = 'emptied'
            s.objs['err'] = 'emptied'
            s.objs['t_shift'] = 'emptied'
            s.objs['t'] = 'emptied'
        run = mak_light_run(run)
        if hasattr(s, 'mdl'):
            for k in ['x','y']:
                if hasattr(s.mdl, k):
                    setattr(s.mdl, k, None)
        store_pickle(s, path)
    except Exception  as e:
        raise
        print(e)

def make_light_batch(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        make_lightweight(file_path)



make_light_batch('nonaut_scal_final')
make_light_batch('Burges_scal_final')
make_light_batch('tomlab_scal_final')
make_light_batch('FHN_scal_times')



res = read_pickle('all_models')
i=0
for i in range(len(res)):
    solver = res[i][0]
    for k in solver.runs.keys():
        solver.fine = 0
        solver.runs[k] = mak_light_run(solver.runs[k])

store_pickle(res, 'all_models')




res = read_pickle('lorenz_nngptime_sim_w_errors')

for k in res['full_data']:
    res['full_data'][k] = ('', '', res['full_data'][k][2],'')

for k in res['nn']:
    _,_,j = k
    if j != 0:
        res['nn'][k] = ''
    else:
        for counter in range(200):
            if counter % 19 != 0:
                res['nn'][k][counter] = ''
            else:
                res['nn'][k][counter] = (res['nn'][k][counter][0], '', res['nn'][k][counter][2], '','','','','','')
store_pickle(res, 'lorenz_nngptime_sim_w_errors')


# #%%
# def list_obj_sizes(obs):
#     from pympler import asizeof
#     for i in obs.__dir__():
#         if not i.startswith('__'):
#             print(i, asizeof.asizeof(getattr(obs, i)))

# def list_obj_sizes_dict(obs):
#     from pympler import asizeof
#     for i in obs.keys():
#         if not i.startswith('__'):
#             print(i, asizeof.asizeof(obs[i]))

# def list_obj_sizes_list(obs):
#     from pympler import asizeof
#     for i in range(len(obs)):
#         print(i, asizeof.asizeof(obs[i]))



# list_obj_sizes(res[1][0])
# list_obj_sizes_dict(res)
# list_obj_sizes_list(res[1][0])


