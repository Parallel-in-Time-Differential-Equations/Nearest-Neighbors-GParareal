
import os
import numpy as np

import jax 
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True) # essential
import matplotlib.pyplot as plt
import time
import scipy
import pickle
import pandas as pd

from article_lib import *

def store_pickle(obj, name):
    with open(os.path.join(name), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def read_pickle(name):
    with open(os.path.join(name), 'rb') as f:
        data = pickle.load(f)
    return data

def store_fig(fig, name):
    fig.savefig(os.path.join('img', name), bbox_inches='tight')
    fig.savefig(os.path.join('img', name+'.pdf'), bbox_inches='tight')




