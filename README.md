# Public GPUVerify Benchmarks

This repository contains updated versions of the public OpenCL and CUDA kernels
used for evaluating [GPUVerify](http://multicore.doc.ic.ac.uk/tools/GPUVerify).

Each kernel includes a header for use with GPUVerify's testing tool
`gvtester.py`.

## Sources

The kernels originate from the following sources:

* The AMD Accelerated Parallel Processing SDK v2.6 (79 OpenCL kernels).
* The NVIDIA GPU Computing SDK 5.0 (188 CUDA kernels). Also included are
  8 CUDA kernels from version 2.0 of the SDK, which are included in version
  5.0 of the SDK.
* The C++ AMP Sample Projects hand-translated to CUDA (20 kernels).
* The gpgpu-sim benchmarks published at ISPASS 2009 (33 CUDA kernels).
* The Parboil benchmarks v2.5 (25 OpenCL kernels).
* The Rodinia benchmark suite v2.4 (40 OpenCL kernels).
* The SHOC benchmark suite (87 OpenCL kernels).
* A set of kernels generated from the PolyBench/C benchmarks v4.0a by the
  PPCG parallel code generator (64 OpenCL kernels).

The kernels are copyright their respective owners.

## Updates

Updates to the kernels are as follows:

* The PolyBench/C kernels have been updated to v4.0a, while most of GPUVerify's
  evaluation used v3.2.
* If data-races were found in kernels and fixes were provided by the authors
  of the kernels, then these fixes have been applied.
* If kernels were no longer accepted by recent versions of the Clang/LLVM
  compiler, which GPUVerify uses as its font-end, then the issues causing
  the non-acceptance have been fixed.

## GPUVerify's Evaluation

GPUVerify's evaluation can be found in the following papers (in chronological
order):

* Adam Betts, Nathan Chong, Alastair F. Donaldson, Shaz Qadeer, Paul Thomson:
  GPUVerify: A verifier for GPU kernels. OOPSLA 2012: 113-132
* Peter Collingbourne, Alastair F. Donaldson, Jeroen Ketema, Shaz Qadeer:
  Interleaving and Lock-Step Semantics for Analysis and Verification of GPU
  Kernels. ESOP 2013: 270-289
* Ethel Bardsley, Adam Betts, Nathan Chong, Peter Collingbourne, Pantazis
  Deligiannis, Alastair F. Donaldson, Jeroen Ketema, Daniel Liew, Shaz Qadeer:
  Engineering a Static Verification Tool for GPU Kernels. CAV 2014: 226-242
* Adam Betts, Nathan Chong, Alastair F. Donaldson, Jeroen Ketema, Shaz Qadeer,
  Paul Thomson, John Wickerson: The Design and Implementation of a Verification
  Technique for GPU Kernels. ACM Trans. Program. Lang. Syst. 37(3): 10:1-10:49
  (2015)
* Adam Betts, Nathan Chong, Pantazis Deligiannis, Alastair F. Donaldson, Jeroen
  Ketema: Implementing and Evaluating Candidate-Based Invariant Generation. To
  appear

The benchmark set has also been used to evaluate a termination checker for
OpenCL and CUDA kernels:

* Jeroen Ketema, Alastair F. Donaldson: Termination analysis for GPU kernels.
  Sci. Comput. Program. 148: 107-122 (2017)

