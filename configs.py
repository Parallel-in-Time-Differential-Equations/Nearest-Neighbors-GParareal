from math import e
import numpy as np

from systems import FHN_ODE, FHN_PDE, Rossler, Hopf, DblPend, Brusselator, Lorenz, ThomasLabyrinth, ODE

class Config:
    def _fhn_ode(self, *args, **kwargs):
        tspan = [0,40]                    
        u0 = np.array([-1,1])               
        N = 40
        Ng = N*4
        Nf = int(160000/160*Ng)
        G = 'RK2'
        F = 'RK4'

        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def _rossler(self, *args, **kwargs):
        tspan = [0,170]                   
        u0 = np.array([0,-6.78,0.02])               
        N = 20                            
        Ng = 45000                        
        Nf = 2250000                    
        G = 'RK1'                         
        F = 'RK4'                         
        tspan2 = [0, tspan[-1]*2]
        N2 = N*2
        Ng2 = Ng*2
        Nf2 = Nf*2

        return {'tspan': tspan2, 'u0': u0, 'N': N2, 'Ng': Ng2/N2, 'Nf': Nf2/N2, 'G': G, 'F': F}
    

    def _hopf(self, N, *args, **kwargs):
        tspan = [-20, 500]                   
        u0 = np.array([0.1,0.1,tspan[0]])        
        Ng = 2*1024                        
        Nf = Ng*85                    
        G = 'RK1'                         
        F = 'RK8' 

        if N is None:
            raise Exception('N must be provided')

        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def _pend(self, *args, **kwargs):
        tspan = [0,80]                    
        u0 = np.array([-0.5,0,0,0])                    
        N = 32                       
        Ng = 3072+N                    
        Nf = Ng*70              
        G = 'RK1'                           
        F = 'RK8'   

        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def _brus(self, *args, **kwargs):
        tspan = [0,100]                      
        u0 = np.array([1,3.07])                         
        N = 25                                
        Ng = N*10                               
        Nf = Ng*100                          
        G = 'RK4'                           
        F = 'RK4' 

        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def _lorenz(self, *args, **kwargs):
        tspan = [0,18]
        u0 = np.array([-15,-15,20])
        N = 50
        Ng = N*6                     
        Nf = Ng*75                     
        G = 'RK4'                            
        F = 'RK4' 

        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def _tomlab(self, N, *args, **kwargs):
        if N == 32:
            tot_time = 10
        elif N == 64:
            tot_time = 10
        elif N == 128:
            tot_time = 40
        elif N == 256:
            tot_time = 100
        elif N == 512:
            tot_time = 100
        else:
            raise Exception('Invalid N value')


        tspan = [0, tot_time]                  
        u0 = np.array([4.6722764,5.2437205e-10,-6.4444208e-10])
        Ng = N*10                      
        Nf = Ng * int(np.ceil(1e6/Ng))      
        G = 'RK1'                         
        F = 'RK4' 
        return {'tspan': tspan, 'u0': u0, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def fhn_pde(self, dx, *args, **kwargs):
        N = 512        
        d_y = d_x = dx
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
            # raise Exception('Invalid d_x val')
            mul = 25
            T = 1100
            G = 'RK4'
                                
        Ng =  N*mul                             
        Nf = int(np.ceil(1e4/Ng)*Ng)        
        F = 'RK8'                               
        epsilon = 5e-7  
        tspan = [0,T]
        return {'tspan': tspan, 'N': N, 'Ng': Ng/N, 'Nf': Nf/N, 'G': G, 'F': F}
    

    def __init__(self, ode:ODE, N=None, d_x=None):
        if isinstance(ode, FHN_ODE):
            config = self._fhn_ode()
        elif isinstance(ode, Rossler):
            config = self._rossler()
        elif isinstance(ode, Hopf):
            config = self._hopf(N)
            ode.name += f'_{N}'
        elif isinstance(ode, DblPend):
            config = self._pend()
        elif isinstance(ode, Brusselator):
            config = self._brus()
        elif isinstance(ode, Lorenz):
            config = self._lorenz()
        elif isinstance(ode, ThomasLabyrinth):
            config = self._tomlab(N)
            ode.name += f'_{N}'
        elif isinstance(ode, FHN_PDE):
            config = self.fhn_pde(d_x)
            # ode.name += f'_{d_x}'
        else:
            raise Exception('No config for input ODE')
        
        if 'u0' in config:
            ode.set_default_init_cond(config['u0'])

        self.config = config


    def _enforce_types(self, config):
        for key, val in config.items():
            if key in ['N', 'Ng', 'Nf']:
                config[key] = int(val)
            elif key in ['u0']:
                config[key] = np.array(val)
        return config


    def get(self):
        return self._enforce_types(self.config)
    



        