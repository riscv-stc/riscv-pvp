from ...isa.inst import *
import numpy as np
import math

class Vid_v(Inst):
    name = 'vid.v'

    def golden(self):
        if self['sew'] <= 64:
            if 'origin' in self:
                res = self['origin'].copy()
            else:
                res = np.zeros(self['vlen']*8//self['sew']).astype('uint'+str(self['sew']))
            
            if 'start' in self:
                start = self['start']
            else:
                start = 0
            
            order_num = np.arange(self['vl']).astype('uint'+str(self['sew']))
            if 'mask' in self:
                mask = np.unpackbits(self['mask'], bitorder='little')[0:self['vl']]
                if 'origin' in self:
                    order_num = np.where(mask==1, order_num, res[0: self['vl']])
                else:
                    order_num = np.where(mask==1, order_num, 0)
                
            res[start: self['vl']] = order_num[start: self['vl']]
            return res
