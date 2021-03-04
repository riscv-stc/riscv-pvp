from ...isa.inst import *
import numpy as np
import math

class Vsex_v(Inst):
    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vl'], dtype=self['vs3'].dtype)
        if 'start' in self:
            start = self['start']
        else:
            start = 0
        
        if 'origin' in self:
            origin = self['origin']
        else:
            origin = np.zeros(self['vlen']//self['vs3'].itemsize, dtype=self['vs3'].dtype)

        if 'offset' in self:
            vs3 = self['vs3'][0: self['vl']].copy()
            vs3.dtype = np.uint8
            mul = self['vs3'].itemsize // vs3.itemsize
            vs3[0: (self['vl']*mul-self['offset'])] = vs3[self['offset'] : self['vl']*mul]
            vs3[(self['vl']*mul-self['offset']): self['vl']*mul] = 0
            vs3.dtype = self['vs3'].dtype
        else:
            vs3 = self['vs3'][0: self['vl']].copy()

        res = self.masked(vs3, origin[0: self['vl']])
        origin[start: self['vl']] = res[start: self['vl']]
         
        return origin

class Vse1_v(Inst):
    name = 'vse1.v'
    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vl'], dtype=self['vs3'].dtype)
        newLen = math.ceil(self['vl']/8)
        
        if 'start' in self:
            start = self['start']
        else:
            start = 0
        res = np.zeros(self['vlen'], dtype=self['vs3'].dtype)
        res[start: newLen] = self['vs3'][start: newLen]
        return res
