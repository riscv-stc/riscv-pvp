from ...isa.inst import *
import numpy as np

class Vmvnr_v(Inst):
    name = 'vmvnr.v'
    '''
    vmv<nr>r.v vd, vs2 # General form  
       vmv1r.v v1, v2  # Copy v1=v2  
       vmv2r.v v10,v12 # Copy v10=v12; v11=v13  
       vmv4r.v v4, v8  # Copy v4=v8; v5=v9; ...; v7=v11  
       vmv8r.v v0, v8  # Copy v0=v8; v1=v9; ...; v7=v15
    '''
    def golden(self):
        if 'vs2' not in self:
            return 0

        if self['vl']==0:
            return self['vs2'] 
               
        result = self['ori'].copy() 
        # vstart = self['vstart'] if 'vstart' in self else 0 
        for ii in range( self['leng']):
            result[ii] = self['vs2'][ii]
        return result

class Vmv1r_v(Vmvnr_v):
    name = 'vmv1r.v'

class Vmv2r_v(Vmvnr_v):
    name = 'vmv2r.v'

class Vmv4r_v(Vmvnr_v):
    name = 'vmv4r.v'

class Vmv8r_v(Vmvnr_v):
    name = 'vmv8r.v'