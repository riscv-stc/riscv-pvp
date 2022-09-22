from ...isa.inst import *
import numpy as np

class Vwredsum_vs(Inst):
    name = 'vwredsum.vs'
    def golden(self):        
        if 'vs2' not in self:
            return 0

        if self['vl']==0:
            return self['vs1'][0]    

        if self['vs2'].dtype == self['vs1'].dtype: 
            self['vs2'].dtype = self.intdtype() 
        
        result   = self['vs1'][0]
        maskflag = 1 if 'mask' in self else 0 

        for ii in range(self['vl']): 
            if (maskflag == 0) or (maskflag == 1 and np.unpackbits(self['mask'], bitorder='little')[ii] ):
                result += self['vs2'][ii]
        return result  
        
class Vwredsumu_vs(Inst):
    name = 'vwredsumu.vs'
    def golden(self):    
        if 'vs2' not in self:
            return 0
                        
        if self['vl'] == 0: 
            return self['ori'][0]

        if self['vs2'].dtype == self['vs1'].dtype: 
            self['vs2'].dtype = self.uintdtype() 
        
        result   = self['vs1'][0] # extension
        maskflag = 1 if 'mask' in self else 0 

        for ii in range(self['vl']): 
            if (maskflag == 0) or (maskflag == 1 and np.unpackbits(self['mask'], bitorder='little')[ii] ):
                result += self['vs2'][ii]
        return result
  