from ...isa.inst import *
import numpy as np
import math

class Vlssegxex_v(Inst):
    name = 'vlssegxex.v'

    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vlen']*8//(self['rs1'].itemsize*8), dtype=self['rs1'].dtype)
        nf = self['nf']
        vl = self['vl']
        emul = self['eew'] /self['sew']*factor_lmul[self['lmul']]
        emul = 1 if emul < 1 else int(emul)
        vlmax = int(emul * self['vlen'] // (self['rs1'].itemsize*8))

        rs1 = self['rs1']
        eew = self['eew']//8
        stride = self['rs2']//eew

        if 'start' in self:
            start = self['start']
        else:
            start = 0

        if 'mask' in self:
            mask = np.unpackbits(self['mask'], bitorder='little')[0: self['vl']]
        else :
            mask = np.ones(self['vl'], dtype=np.uint8)
        
        if 'origin' in self:
            origin = self['origin']
        else:
            origin = np.zeros(self['vlen']*8//(self['rs1'].itemsize*8), dtype=self['rs1'].dtype)
        
        tmp = np.zeros(vl*stride+nf, dtype=self['rs1'].dtype)

        for i in range(vl):
            tmp[i*stride: i*stride+nf] = rs1[i*nf: i*nf+nf]

        for i in range(start, vl):
            if mask[i] != 0:
                origin[i: (nf-1)*vlmax+i+1: vlmax] = tmp[i*stride: i*stride+nf]

        return origin
