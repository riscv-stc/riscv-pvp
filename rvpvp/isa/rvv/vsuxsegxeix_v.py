from ...isa.inst import *
import numpy as np
import math

class Vsuxsegxeix_v(Inst):
    name = 'vsuxsegxeix.v'

    def golden(self):
        if 'isExcept' in self:
            if self['isExcept'] > 0:
                return np.zeros(self['vlen']*8//(self['vs3'].itemsize*8), dtype=self['vs3'].dtype)
        nf = self['nf']
        vl = self['vl']
        lmul = 1 if factor_lmul[self['lmul']] < 1 else factor_lmul[self['lmul']]
        vlmax = int(self['vlen']*lmul//self['sew'])

        vs3 = self['vs3'][0:vlmax*nf].reshape(nf, vlmax).T
        index = self['vs2']

        if 'start' in self:
            start = self['start']
        else:
            start = 0

        if 'mask' in self:
            mask_src = self['mask']
            mask_src.dtype = np.uint8
            mask = np.unpackbits(mask_src, bitorder='little')[0: self['vl']]
        else :
            mask = np.ones(self['vl'], dtype=np.uint8)
        
        if 'origin' in self:
            origin = self['origin']
        else:
            origin = np.zeros(self['vlen']*8//(self['vs3'].itemsize*8), dtype=self['vs3'].dtype)
        
        tmp = np.zeros(int(np.max(self['vs2'])//self['vs3'].itemsize+nf), dtype=self['vs3'].dtype)

        for i in range(start, vl):
            curIdx = int(index[i] // self['vs3'].itemsize)
            tmp[curIdx: curIdx+nf] = origin[i*nf: i*nf+nf]

        for i in range(start, vl):
            if mask[i] != 0:
                curIdx = int(index[i] // self['vs3'].itemsize)
                tmp[curIdx: curIdx+nf] = vs3[i]
                
        for i in range(start, vl):
            curIdx = int(index[i] // self['vs3'].itemsize)
            origin[i*nf: i*nf+nf] = tmp[curIdx: curIdx+nf]


        return origin