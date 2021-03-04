# Verification Case Specification

This is a detailed reference guide to RISC-V PVP Verification Case Specification.

The best way to get started with YAML Verification Case Specification is to read
the customization guide. After that, to learn how to write your YAML cases
for your needs, see below topics.

### YAML basics

Verification Case Specification is written in YAML file format.
To learn the basics of YAML, please see [Learn YAML in Y Minutes].

[Learn YAML in Y Minutes]: https://learnxinyminutes.com/docs/yaml/

## Structure

A verification case **Group** is a group of one or more verification **Cases**. 
These **Cases** share same **Environment**, **Headers**, **Templates**,
case **Params** and **Check** rules.

This hierarchy is reflected in the structure of a YAML file like:

* Group 1
    * Environment
    * Headers
    * Templates
    * Params
    * Check
* Group 2
    * Environment
    * Headers
    * Templates
    * Params
    * Check

For example, We could define two simple case groups for rvv instructions, 
vadd.vi and vsub.vi.

```yaml
vadd_vi:
  env: ...
  head: ...
  templates: ...
  cases: ...
  check: ...

vsub_vi:
  env: ...
  head: ...
  templates: ...
  cases: ...
  check: ...
```

Since vadd.vi and vsub.vi are both Vector Integer Arithmetic Instructions, and
could use same **Environment**, **Headers**, **Templates**, case **Params** and
**Check** rules. So we could use YAML's "anchors" feature to duplicate/inherit
properties from a same base group.

```yaml
_: &default
  env: ...
  head: ...
  templates: ...
  cases: ...
  check: ...

vadd_vi:
  <<: *default

vsub_vi:
  <<: *default
```

The base group is named `_`, the `vadd_vi` and `vsub_vi` groups will inherit all
its properties, such as env, head, templates, cases, check. Group names start
with `_` will be ignored by parser, and not consider as a valid group.

So, the two syntax above are equivalent.

## Environment

We reuse riscv-test-env to define target environments, test virtual machine
([TVM]) is used to define only used features for tests.

[TVM]: https://github.com/riscv/riscv-tests/#test-virtual-machines

The following table shows the TVMs currently defined.

TVM Macro       | TVM Name | Description
---             | ---      | ---
`RVTEST_RV32U`  | `rv32ui` | RV32 user-level, integer only
`RVTEST_RV32UF` | `rv32uf` | RV32 user-level, integer and floating-point
`RVTEST_RV32UV` | `rv32uv` | RV32 user-level, integer, floating-point, and vector
`RVTEST_RV32S`  | `rv32si` | RV32 supervisor-level, integer only
`RVTEST_RV32M`  | `rv32mi` | RV32 machine-level, integer only
`RVTEST_RV64U`  | `rv64ui` | RV64 user-level, integer only
`RVTEST_RV64UF` | `rv64uf` | RV64 user-level, integer and floating-point
`RVTEST_RV64UV` | `rv64uv` | RV64 user-level, integer, floating-point, and vector
`RVTEST_RV64S`  | `rv64si` | RV64 supervisor-level, integer only
`RVTEST_RV64M`  | `rv64mi` | RV64 machine-level, integer only

For example, we use `RVTEST_RV64UV` for all rvv cases.

```yaml
_: &default
  env: RVTEST_RV64UV
  ...

vadd_vi:
  <<: *default

vsub_vi:
  <<: *default
```

## Headers

**Headers** section is used for test case templates to include header files.
The header files are placed in `macros` directory.

For example, all rvv cases will include "test_macros_v.h".

```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  ...

vadd_vi:
  <<: *default

vsub_vi:
  <<: *default
```

## Templates

**Templates** section is used to place assembly code template for test cases.
The key is the test case name, value is the code template string for this test
case.

Variables in templates are delimited by braces `{}`. Except for internal
variables, other variables will be instantiated with the parameters defined in
the **Cases** section.

The following table shows the internal variables currently defined.

Variable  | Description
---       | ---
`num`     | test case index number
`name`    | instruction name converted from group name, `vadd_vi` will convert into `vadd.vi`
`*_shape` | data will be generated by numpy's ndarray, this variable is the data shape.
`*_data`  | data will be generated by numpy's ndarray, this variable is the data address.


For example, the below template is a basic test case `test_basic_without_mask`
for `v*.vi`.
Except for internal variables, other variables such as `vs2_data`, `imm`, `vl`,
`sew`, `lmul` should be defined in **Cases** section.

```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  templates:
    test_basic_without_mask: |
      test_{num}:
        li TESTNUM, {num};

        // set vtype and vl
        li a0, {vl}; 
        vsetvli t0, a0, e{sew},m{lmul},ta,ma;

        // load input data into source vregs
        la a2, {vs2_data}; 
        vle{sew}.v v16, (a2);

        // run tested instruction
        {name} v24, v16, {imm};

        // store output data into result sections
        la a4, test_{num}_data;
        vse{sew}.v v24, (a4);

        // push a result sub-section into signature in data section
        .pushsection .data, 1;
        .balign ({sew}/8)
      test_{num}_data:
        .fill ({vl}), ({sew}/8), 0;
        .popsection

    test_basic_with_mask: |
      ...
  ...

vadd_vi:
  <<: *default

vsub_vi:
  <<: *default
```


## Cases

**Cases** section defines test case parameters to instantiate the teamplate
variables.

There are three method to define parameters.

### Parameter Array

The simplest method is define a parameter array. Continue with the above example,
we will define `vs2_data`, `imm`, `vl`, `sew`, `lmul` parameters for
`test_basic_without_mask` test case.

Each array item is a parameter array written in python.

```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  templates:
    test_basic_without_mask: |
      ...
  cases:
    test_basic_without_mask @ vs2, imm, vl, sew @ lmul = 1:
      - '[np.array([4], dtype=np.int8),             4,  1,  8]'
      - '[np.array([-2], dtype=np.int16),           3,  1, 16]'
      - '[np.linspace(0, 1, 16, dtype=np.int32),    9, 16, 32]'
      - '[np.linspace(0, -1, -31, dtype=np.int32), 13, 31,  8]'
      - '[np.linspace(0, -15, 16, dtype=np.int32), 15, 32, 16]'
```

### Parameter Matrix

We could define a parameter matrix to combine parameters. The parameter matrix
is a YAML object with a single `matrix` key in it. The keys of the matrix object
are variable names, and the values are also written in python.

```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  templates:
    test_basic_without_mask: |
      ...
  cases:
    test_basic_without_mask:
      matrix:
        sew : [8, 16, 32, 64]
        lmul: get_lmul_w(sew)
        vl  : get_vl(lmul, sew, vlen)
        vs1 : np.linspace(-100, 100).astype(get_intdtype(sew))
        imm : [1, -1, 100]
```

### Setup Script

Pure python script to define parameters is also supported. A YAML object with a
single `setup` key in it is used to define setup script. Int the python script,
we should define and push all parameters into a `params_yml` array.


```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  templates:
    test_basic_without_mask: |
      ...
  cases:
    test_basic_without_mask @ vs2, imm, vl, sew, lmul:
      setup: |
        params_yml = []
        for sew in [8, 16, 32, 64]:
          for lmul in get_lmul_w(sew):
            for vl in get_vl(lmul, sew, vlen):
              vs2 = np.linspace(-100, 100).astype(get_intdtype(sew))
              imm = [1, -1, 100]
              params_yml.append([vs2, imm, vl, sew, lmul])
```


## Check

Finally, we should define the check result rules in `check` section.
RISC-V PVP will compile the parameters injected templates into target elf file,
and run simulators to get result. Then, the result and golden will convert into
numpy ndarray's.

The check section is python code how to check `result` and `golden`, it will
return true if the check pass, otherwise false.

```yaml
_: &default
  env: RVTEST_RV64UV
  head: |
    #include "test_macros_v.h"
  templates:
    test_basic_without_mask: |
      ...
  cases:
    test_basic_without_mask: ...
     
  check:
    test_basic_without_mask: np.array_equal(result, golden)
```