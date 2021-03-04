from ...isa.inst import *
import numpy as np

class Vmv_s_x(Inst):
    name = 'vmv.s.x'
    # vmv.s.x vd, rs1 # vd[0] = x[rs1] (vs2=0)
    def golden(self):
        if 'rs1' not in self:
            return 0

        vstart = self['vstart'] if 'vstart' in self else 0
        if self['vl'] > 0 and vstart < self['vl']:    
            self['vd'][0] = self['rs1']
            
        return self['vd']
