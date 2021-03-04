from ...isa.inst import *

class Vnull_vs(Inst):
    name = 'vnull_vs'
    # test exception case, no need golden
    def golden(self):
        return 0

class Vnull_fs(Inst):
    name = 'vnull_fs'
    # test exception case, no need golden
    def golden(self):
        return 0

class Vnull_fs2(Vnull_fs):
    name = 'vnull_fs2'
