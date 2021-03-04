from ...isa.inst import *
import numpy as np
import math

class Vlsegxexff_v(Inst):
    name = 'vlsegxexff.v'

    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vlen']*8//(self['rs1'].itemsize*8), dtype=self['rs1'].dtype)
        nf = self['nf']
        vl = self['vl']
        emul = self['eew'] /self['sew']*factor_lmul[self['lmul']]
        emul = 1 if emul < 1 else int(emul)
        vlmax = int(emul * self['vlen'] // (self['rs1'].itemsize*8))

        rs1 = self['rs1'].reshape(vl, nf).T

        if 'start' in self:
            start = self['start']
        else:
            start = 0

        if 'nvl' in self:
            end = self['nvl']
        else :
            end = vl
        
        if 'origin' in self:
            origin = self['origin']
        else:
            origin = np.zeros(self['vlen']*8//(self['rs1'].itemsize*8), dtype=self['rs1'].dtype)
        

        for i in range(nf):
            origin[vlmax*i+start: vlmax*i+end] = self.masked(rs1[i], origin[vlmax*i: vlmax*i+end])[start: end]

        return origin
