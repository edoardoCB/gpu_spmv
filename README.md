# gpu_spmv

This project contains the code to perform the SpMV product on NVIDIA's GPUs.

CSR format.
	- Naive version
	- CUSPARSE function.

AXC format.
	- New version.

K1 format.
	- Original version from paper.
	- Improved version (reordering integrated)

AXT format.
	- Uncompacted with tileHeight = 1.
	- Uncompacted with tileHeight > 1. 
	- Compacted with tileHeight = 1 (not completed)


The AXT format has been published in the paper "A new AXT format for an efficient SpMV product using AVX-512 instructions and CUDA" in the journal 
Advances in Engineering Software. The paper can be found on the following link: https://www.sciencedirect.com/science/article/pii/S0965997821000260 .

The AXT format's software is part of the AXT_SPL library registered under a GPL license. The Universidad de Santiago de Compostela partially owns the 
intellectual rights.
