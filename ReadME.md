# Matrix Addition in CUDA C

The ```main.cu``` program consists of three cuda kernels for adding two square matrices of dimension N, namely:

1. ```kernel_1t1e``` for element wise addition with N^2 threads, 
2. ```kernel_1t1r``` for row-wise addition with N threads, and
3. ```kernel_1t1c``` for column-wise addition with N threads. 

A technical report is included in the repository for the detailed performance analysis of each kernel. This is a partial requirement for CS 239 (Parallel Programming I).

Should you have any comments on the report or code, please do message me on LinkedIn or email me at ```bacong.junelle@gmail.com```.