/*
 * Copyright (c) 2020 Stream Computing Corp.
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Tests for vfmv.s.f instructions
 * 
 *    vfmv.s.f vd, rs1
 * 
 * Authors: Li Zhiyong
 */
#ifndef __TEST_MACROS_VFMV_S_F_H
#define __TEST_MACROS_VFMV_S_F_H

#include "test_macros_v.h"
#include "test_macros.h"


/**
 * General functional tests without mask
 *
 * @testnum   test case number
 * @result    start addr for test result
 * @vd        dest vector addr
 * @val1      start addr for source vector 1
 * @vlen      vector length
 * @ebits     element bits, 8 for byte, 16 for half, 32 for word
 * @ldins     inst to load source vector
 * @stins     inst to store vector
 * fldins     inst to load a float number
 * @eqm       macro for compare two vectors
 */
#define TEST_V_FMV_S_F_INTERNAL(testnum, result, vd, val1, vlen, ebits, ldins, stins, fldins, eqm) \
test_ ## testnum: \
    COPY(RD_ADDR, vd, vlen, lh, sh, 2); \
    li TESTNUM, testnum; \
    li a0, vlen ; \
    la a1, RD_ADDR ; \
    la a2, val1 ; \
    fldins fa1, (a2) ; \
  vsetvli t0, a0, e ## ebits; \
  ldins v0, (a1) ; \
    PERF_BEGIN() \
  vfmv.s.f v0, fa1; \
    PERF_END(testnum ## _ ## vlen) \
  stins v0, (a1); \
    COPY(vd, RD_ADDR, vlen, lh, sh, 2); \
    eqm(result, vd, vlen);

/**
 * Functional tests without mask
 *
 * @testnum   test case number
 * @result    start addr for test result
 * @vd        addr for dest vector
 * @val1      start addr for source vector1
 * @vlen      vector length
 */
#define TEST_V_FMV_S_F_HF(testnum, result, vd, rs1, vlen) \
  TEST_V_FMV_S_F_INTERNAL(testnum, result, vd, rs1, vlen, 16, vlh.v, vsh.v, flw, VV_SH_CHECK_EQ)



#endif // __TEST_MACROS_VFMV_S_F_H
