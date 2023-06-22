# xsimdVSIntelntrinsics


This simple application demonstrates performance parity of [XSIMD library](https://github.com/xtensor-stack/xsimd) and native Intel Instrincs operations on FMA operations. 


### Computation Description ##

Perform FMA operations on 10000 floating point data points and measure the time taken by  native AVX2 Intel intrinsics, AVX2 xsimd library and standard C++ libs

### System used ##
- 12th Gen Intel(R) Core(TM) i9-12900   2.40 GHz
- 32.0 GB (31.6 GB usable)
- 64-bit operating system, x64-based processor


### Results ##
<br> Elapsed time for FMA operation using native Intel AVX2 Instrinsic is :: 6 microseconds<br> 

<br> Elapsed time for FMA operation using XSIMD wrapper for AVX2 is :: 23 microseconds<br> 

<br> Elapsed time for Scalar FMA operation using standard libs is :: 18 microseconds<br> 

