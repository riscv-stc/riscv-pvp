from ...isa.inst import *
import numpy as np
import math

class Vmsof_m(Inst):
    name = 'vmsof.m'

    def golden(self):
        if 'mask' in self:
            if 'vs2' in self:
              tmp = np.unpackbits(self['vs2'] & self['mask'], bitorder='little')[0: self['vl']]
            else:
              tmp = np.unpackbits(self['mask'] & self['mask'], bitorder='little')[0: self['vl']]
        else:
            tmp = np.unpackbits(self['vs2'], bitorder='little')[0: self['vl']]
    
        res = np.zeros(self['vlen'], dtype=np.uint8)
        if np.size(np.where(tmp == 1)) > 0:
          firstOne = np.min(np.where(tmp == 1))
          res[firstOne] = 1

        if 'origin' in self:
          origin_bits = np.unpackbits(self['origin'], bitorder='little')
          if 'mask' in self:
            mask = np.unpackbits(self['mask'], bitorder='little')[0: self['vl']]
            res[0:self['vl']] = np.where( mask == 1, res[0:self['vl']], origin_bits[0:self['vl']])
          origin_bits[0:self['vl']] = res[0:self['vl']]
          return np.packbits(origin_bits, bitorder='little')

        return np.packbits(res, bitorder='little')


class Vmsbf_m(Inst):
    name = 'vmsbf.m'

    def golden(self):
        if 'mask' in self:
            if 'vs2' in self:
              tmp = np.unpackbits(self['vs2'] & self['mask'], bitorder='little')[0: self['vl']]
            else:
              tmp = np.unpackbits(self['mask'] & self['mask'], bitorder='little')[0: self['vl']]
        else:
            tmp = np.unpackbits(self['vs2'], bitorder='little')[0: self['vl']]
        res = np.zeros(self['vlen'], dtype=np.uint8)
        res[0: self['vl']] = np.ones(self['vl'], dtype=np.uint8)
        if np.size(np.where(tmp == 1)) > 0:
          firstOne = np.min(np.where(tmp == 1))
          for i in range(firstOne, self['vl']):
            res[i] = 0
        if 'mask' in self:
          mask = np.unpackbits(self['mask'], bitorder='little')[0:self['vl']]
          res[0:self['vl']] = np.where( mask == 1, res[0:self['vl']], 0)
        if 'origin' in self:
          origin_bits = np.unpackbits(self['origin'], bitorder='little')
          if 'mask' in self:
            mask = np.unpackbits(self['mask'], bitorder='little')[0: self['vl']]
            res[0:self['vl']] = np.where( mask == 1, res[0:self['vl']], origin_bits[0:self['vl']])
          origin_bits[0:self['vl']] = res[0:self['vl']]
          return np.packbits(origin_bits, bitorder='little')
        return np.packbits(res, bitorder='little')

class Vmsif_m(Inst):
    name = 'vmsif.m'

    def golden(self):
        if 'mask' in self:
            if 'vs2' in self:
              tmp = np.unpackbits(self['vs2'] & self['mask'], bitorder='little')[0: self['vl']]
            else:
              tmp = np.unpackbits(self['mask'] & self['mask'], bitorder='little')[0: self['vl']]
        else:
            tmp = np.unpackbits(self['vs2'], bitorder='little')[0: self['vl']]
        res = np.zeros(self['vlen'], dtype=np.uint8)
        res[0: self['vl']] = np.ones(self['vl'], dtype=np.uint8)
        if np.size(np.where(tmp == 1)) > 0:
          firstOne = np.min(np.where(tmp == 1))
          for i in range(firstOne+1, self['vl']):
            res[i] = 0
        if 'mask' in self:
          mask = np.unpackbits(self['mask'], bitorder='little')[0:self['vl']]
          res[0:self['vl']] = np.where( mask == 1, res[0:self['vl']], 0)
        if 'origin' in self:
          origin_bits = np.unpackbits(self['origin'], bitorder='little')
          if 'mask' in self:
            mask = np.unpackbits(self['mask'], bitorder='little')[0: self['vl']]
            res[0:self['vl']] = np.where( mask == 1, res[0:self['vl']], origin_bits[0:self['vl']])
          origin_bits[0:self['vl']] = res[0:self['vl']]
          return np.packbits(origin_bits, bitorder='little')
        return np.packbits(res, bitorder='little')