from ...isa.inst import *
import math

class Vlex_v(Inst):
    def golden(self): 
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vl'], dtype=self['rs1'].dtype)
        if 'start' in self:
            start = self['start']
        else:
            start = 0
        
        if 'origin' in self:
            origin = self['origin']
        else:
            origin = np.zeros(self['vlen']//self['rs1'].itemsize, dtype=self['rs1'].dtype)

        if self['vl'] == 0:
            return origin
        
        if 'offset' in self:
            rs1 = self['rs1'].copy()
            rs1.dtype = np.uint8
            mul = self['rs1'].itemsize // rs1.itemsize
            rs1[0: (self['vl']*mul-self['offset'])] = rs1[self['offset'] : self['vl']*mul]
            rs1[(self['vl']*mul-self['offset']): self['vl']*mul] = 0
            rs1.dtype = self['rs1'].dtype
        else:
            rs1 = self['rs1'].copy()

        res = self.masked(rs1, origin[0: self['vl']])
        origin[start: self['vl']] = res[start: self['vl']]
         
        return origin


class Vle1_v(Inst):
    name = 'vle1.v'
    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vl'], dtype=self['rs1'].dtype)
        newLen = math.ceil(self['vl']/8)
        
        if 'start' in self:
            start = self['start']
        else:
            start = 0
        res = np.zeros(self['vlen']//self['rs1'].itemsize, dtype=self['rs1'].dtype)
        res[start: newLen] = self['rs1'][start: newLen]
        return res

