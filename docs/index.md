# Overview

RISC-V PVP is a modular and parameterized RISC-V Processor Verification Platform.

It currently supports the following features:

- Supported RISC-V ISA:
    - V Extension, 1.0
- YAML based hand-written sequence templates
- numpy based golden data for vector ISA
- Easy to support new targets with new or custom ISAs
- End-to-end RTL&ISS co-simulation flow

## Motivation

Open source RISC-V processor verification solutions such as riscv-tests,
riscv-arch-test have provided good sanity verifications for RISCV basic ISAs,
but new extensions(for example, Vector, Bitmanip) support is missing.

Our motivation is build a high quality open source verification platform to
support more extensions easily to improve the verification quality of RISC-V
processors.

## Issues

Please file an issue under this repository for any bug report / integration
issue / feature request. We are looking forward to knowing your experience of
using this flow and how we can make it better together.

