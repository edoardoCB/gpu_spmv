// ┌────────────────────────────────┐
// │program: scan.cu                │
// │author: Edoardo Coronado        │
// │date: 30-09-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#ifndef __SCAN_HEADER__
#define __SCAN_HEADER__



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#ifndef OMP_SCH
	#define OMP_SCH static
	char omp_schedule[7] = "static";
#endif



#ifndef FP_FLOAT
	#define FP_FLOAT  1
#endif



#ifndef FP_DOUBLE
	#define FP_DOUBLE 2
#endif



#if FP_TYPE == FP_FLOAT
	typedef float  FPT;
	char fptMsg[6] = "float";
#endif



#if FP_TYPE == FP_DOUBLE
	typedef double FPT;
	char fptMsg[7] = "double";
#endif



#ifndef UIN
	typedef unsigned int UIN;
#endif



#ifndef HDL
	#define HDL { fflush(stdout); printf( "---------------------------------------------------------------------------------------------------------\n" ); fflush(stdout); }
#endif



#ifndef BM
	#define BM { fflush(stdout); printf( "\nFile: %s    Line: %d.\n", __FILE__, __LINE__ ); fflush(stdout); }
#endif



#ifndef NUM_ITE
	#define NUM_ITE 250
#endif



typedef struct { UIN al; UIN cbs; UIN ompMT; } str_inputArgs;



static str_inputArgs checkArgs( const UIN argc, char ** argv )
{
	if ( argc < 3 )
	{
		fflush(stdout);
		printf( "\n\tMissing input arguments.\n" );
		printf( "\n\tUsage:\n\n\t\t%s <arrayLength> <cudaBlockSize>\n\n", argv[0] );
		printf( "\t\t\t<arrayLength>:   number of elements of the array.\n" );
		printf( "\t\t\t<cudaBlockSize>: number of threads of a cuda block.\n" );
		fflush(stdout);
		exit( EXIT_FAILURE );
	}
	str_inputArgs sia;
	sia.al    = atoi( argv[1] );
	sia.cbs   = atoi( argv[2] );
	sia.ompMT = 1;
	#pragma omp parallel if(_OPENMP)
	{
		#pragma omp master
		{
			sia.ompMT = omp_get_max_threads();
		}
	}
	return( sia );
}



#ifndef ABORT
	#define ABORT { fflush(stdout); printf( "\nFile: %s Line: %d execution is aborted.\n", __FILE__, __LINE__ ); fflush(stdout); exit( EXIT_FAILURE ); }
#endif



static void printRunSettings( const str_inputArgs sia )
{
	HDL; printf( "input settings\n" ); HDL;
	printf( "FPT:           %s\n", fptMsg    );
	printf( "sizeof(FPT):   %zu bytes\n", sizeof(FPT) );
	printf( "arrayLength:   %d\n", sia.al    );
	printf( "cudaBlockSize: %d\n", sia.cbs   );
	printf( "ompMaxThreads: %d\n", sia.ompMT );
	printf( "NUM_ITE:       %d\n", (UIN) NUM_ITE ); fflush(stdout);
	return;
}



#ifndef HANDLE_CUDA_ERROR
	#define HANDLE_CUDA_ERROR( ceID ) { if ( ceID != cudaSuccess ) { printf( "FILE: %s LINE: %d CUDA_ERROR: %s\n", __FILE__, __LINE__, cudaGetErrorString( ceID ) ); fflush(stdout); printf( "\nvim %s +%d\n", __FILE__, __LINE__); exit( EXIT_FAILURE ); } }
#endif



static __host__ void getCudaDeviceCounter( int * counter )
{
	HANDLE_CUDA_ERROR( cudaGetDeviceCount(counter) );
	HDL; printf( "cuda device properties\n" ); HDL;
	printf( "cudaDeviceCounter = %d\n\n", *counter );
	return;
}



static __host__ const char * getCudaComputeModeString( const int computeModeCode )
{
	switch( computeModeCode )
	{
		case 0: return "cudaComputeModeDefault";
		case 1: return "cudaComputeModeExclusive";
		case 2: return "cudaComputeModeProhibited";
		case 3: return "cudaComputeModeExclusiveProcess";
	}
	return "Unknown cudaComputeModeCode";
}



static __host__ void printCudaDeviceProperties( int cudaDeviceID )
{
	cudaDeviceProp cudaDeviceProperties;
	HANDLE_CUDA_ERROR( cudaGetDeviceProperties( &cudaDeviceProperties, cudaDeviceID ) );
	printf( "cudaDeviceID:                                     %d <-------------------\n", cudaDeviceID );
	printf( "cudaDeviceProperties.name:                        %s\n", cudaDeviceProperties.name );
	printf( "cudaDeviceProperties.totalGlobalMem:              %.1f %s\n", ( (float) cudaDeviceProperties.totalGlobalMem / (float) ( 1024 * 1024 * 1024 ) ), "GBytes" );
	printf( "cudaDeviceProperties.sharedMemPerBlock:           %.1f %s\n", ( (float) cudaDeviceProperties.sharedMemPerBlock / (float) 1024 ), "KBytes" );
	printf( "cudaDeviceProperties.textureAlignment:            %.1f %s\n", ( (float) cudaDeviceProperties.textureAlignment ), "Bytes" );
	printf( "cudaDeviceProperties.maxThreadsDim[0]:            %d\n", cudaDeviceProperties.maxThreadsDim[0] );
	printf( "cudaDeviceProperties.maxGridSize[0]:              %d\n", cudaDeviceProperties.maxGridSize[0] );
	printf( "cudaDeviceProperties.maxThreadsPerBlock:          %d\n", cudaDeviceProperties.maxThreadsPerBlock );
	printf( "cudaDeviceProperties.maxThreadsPerMultiProcessor: %d\n", cudaDeviceProperties.maxThreadsPerMultiProcessor );
	printf( "cudaDeviceProperties.multiProcessorCount:         %d\n", cudaDeviceProperties.multiProcessorCount  );
	printf( "cudaDeviceProperties.warpSize:                    %d\n", cudaDeviceProperties.warpSize );
	printf( "cudaDeviceProperties.canMapHostMemory:            %d\n", cudaDeviceProperties.canMapHostMemory );
	printf( "cudaDeviceProperties.major:                       %d\n", cudaDeviceProperties.major );
	printf( "cudaDeviceProperties.minor:                       %d\n", cudaDeviceProperties.minor );
	printf( "cudaDeviceProperties.regsPerBlock:                %d\n", cudaDeviceProperties.regsPerBlock );
	printf( "cudaDeviceProperties.multiProcessorCount:         %d\n", cudaDeviceProperties.multiProcessorCount );
	printf( "cudaDeviceProperties.computeMode:                 %s\n", getCudaComputeModeString( cudaDeviceProperties.computeMode ) );
	// set the bandwidth of the shared memory banks
	if ( sizeof(FPT) == 4 ) HANDLE_CUDA_ERROR( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeFourByte  ) );
	if ( sizeof(FPT) == 8 ) HANDLE_CUDA_ERROR( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );
	// verify bandwidth of the shared memory banks
	cudaSharedMemConfig csmc;
	HANDLE_CUDA_ERROR( cudaDeviceGetSharedMemConfig( &csmc ) );
	unsigned short int bpb;
	if ( csmc == cudaSharedMemBankSizeFourByte  ) bpb = 4;
	if ( csmc == cudaSharedMemBankSizeEightByte ) bpb = 8;
	printf( "cudaDeviceSharedMemConfig:                        %1hu bytes\n", bpb );
	//HDL;
	return;
}


#ifndef TEST_POINTER
	#define TEST_POINTER( p ) { if ( p == NULL ) { fflush(stdout); printf( "\nFile: %s Line: %d Pointer: %s is null\n", __FILE__, __LINE__, #p ); fflush(stdout); exit( EXIT_FAILURE ); } }
#endif



static void init_vec( const UIN ompNT, const UIN len, FPT * vec )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( i = 0 ; i < len; i++ )
		vec[i] = 1.0;
	return;
}



#ifndef FULL_MASK
	#define FULL_MASK 0xffffffff
#endif



static __device__ FPT warp_red1( const UIN tidWARP, const FPT val )
{
	UIN id = tidWARP + 1;
	FPT v1 = val;
	FPT v2 = __shfl_up_sync( FULL_MASK, v1,  1); if ((id% 2) == 0) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  2); if ((id% 4) == 0) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  4); if ((id% 8) == 0) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  8); if ((id%16) == 0) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1, 16); if ((id%32) == 0) v1 = v1 + v2;
	return( v1 );
}



static __device__ FPT warp_red2( const UIN tidWARP, const FPT val )
{
	FPT v1 = val;
	FPT v2 = __shfl_up_sync( FULL_MASK, v1,  1); if (tidWARP >=  1) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  2); if (tidWARP >=  2) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  4); if (tidWARP >=  4) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  8); if (tidWARP >=  8) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1, 16); if (tidWARP >= 16) v1 = v1 + v2;
	return( v1 );
}


static __device__ FPT warp_scan1( const UIN tidWARP, const FPT val )
{
	FPT v1 = val;
	FPT v2 = __shfl_up_sync( FULL_MASK, v1,  1); if (tidWARP >=  1) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  2); if (tidWARP >=  2) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  4); if (tidWARP >=  4) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1,  8); if (tidWARP >=  8) v1 = v1 + v2;
	    v2 = __shfl_up_sync( FULL_MASK, v1, 16); if (tidWARP >= 16) v1 = v1 + v2;
	return( v1 );
}



static __device__ FPT warp_scan2( const UIN tidWARP, const FPT val )
{
	FPT v1 = val;
	FPT v2;
	UIN i;
	for ( i = 1; i <=16; i = i * 2 )
	{
		v2 = __shfl_up_sync( FULL_MASK, v1, i );
		if (tidWARP >= i) v1 = v1 + v2;
	}
	return( v1 );
}



static __device__ FPT warp_scan3( const UIN tidWARP, const FPT val )
{
	UIN id = tidWARP + 1;
	FPT v1 = val;
	FPT v2 = __shfl_up_sync( FULL_MASK, v1, 1 );
	if (id ==  2) v1 = v1 + v2;
	if (id ==  4) v1 = v1 + v2;
	if (id ==  6) v1 = v1 + v2;
	if (id ==  8) v1 = v1 + v2;
	if (id == 10) v1 = v1 + v2;
	if (id == 12) v1 = v1 + v2;
	if (id == 14) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 18) v1 = v1 + v2;
	if (id == 20) v1 = v1 + v2;
	if (id == 22) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 26) v1 = v1 + v2;
	if (id == 28) v1 = v1 + v2;
	if (id == 30) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v2 = __shfl_up_sync( FULL_MASK, v1,  2 );
	if (id ==  4) v1 = v1 + v2;
	if (id ==  8) v1 = v1 + v2;
	if (id == 12) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 20) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 28) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v2 = __shfl_up_sync( FULL_MASK, v1,  4 );
	if (id ==  8) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v2 = __shfl_up_sync( FULL_MASK, v1,  8 );
	if (id == 16) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v2 = __shfl_up_sync( FULL_MASK, v1, 16 );
	if (id == 32) v1 = v1 + v2;
	v2 = __shfl_up_sync( FULL_MASK, v1, 31 );
	if (tidWARP==31) v1 = v2;
	FPT v3  = v1;
	v2 = __shfl_down_sync( FULL_MASK, v1, 16 );
	if (id == 16) v1 = v2;
	v2 = __shfl_up_sync  ( FULL_MASK,  v3, 16 );
	if (id == 32) v1 = v1 + v2;
	v3  = v1;
	v2 = __shfl_down_sync( FULL_MASK, v1,  8 );
	if (id ==  8) v1 = v2;
	if (id == 24) v1 = v2;
	v2 = __shfl_up_sync  ( FULL_MASK,  v3,  8 );
	if (id == 16) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v3  = v1;
	v2 = __shfl_down_sync( FULL_MASK, v1,  4 );
	if (id ==  4) v1 = v2;
	if (id == 12) v1 = v2;
	if (id == 20) v1 = v2;
	if (id == 28) v1 = v2;
	v2 = __shfl_up_sync  ( FULL_MASK,  v3,  4 );
	if (id ==  8) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v3  = v1;
	v2 = __shfl_down_sync( FULL_MASK, v1,  2 );
	if (id ==  2) v1 = v2;
	if (id ==  6) v1 = v2;
	if (id == 10) v1 = v2;
	if (id == 14) v1 = v2;
	if (id == 18) v1 = v2;
	if (id == 22) v1 = v2;
	if (id == 26) v1 = v2;
	if (id == 30) v1 = v2;
	v2 = __shfl_up_sync  ( FULL_MASK,  v3,  2 );
	if (id ==  4) v1 = v1 + v2;
	if (id ==  8) v1 = v1 + v2;
	if (id == 12) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 20) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 28) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	v3  = v1;
	v2 = __shfl_down_sync( FULL_MASK, v1,  1 );
	if (id ==  1) v1 = v2;
	if (id ==  3) v1 = v2;
	if (id ==  5) v1 = v2;
	if (id ==  7) v1 = v2;
	if (id ==  9) v1 = v2;
	if (id == 11) v1 = v2;
	if (id == 13) v1 = v2;
	if (id == 15) v1 = v2;
	if (id == 17) v1 = v2;
	if (id == 19) v1 = v2;
	if (id == 21) v1 = v2;
	if (id == 23) v1 = v2;
	if (id == 25) v1 = v2;
	if (id == 27) v1 = v2;
	if (id == 29) v1 = v2;
	if (id == 31) v1 = v2;
	v2 = __shfl_up_sync  ( FULL_MASK,  v3,  1 );
	if (id ==  2) v1 = v1 + v2;
	if (id ==  4) v1 = v1 + v2;
	if (id ==  6) v1 = v1 + v2;
	if (id ==  8) v1 = v1 + v2;
	if (id == 10) v1 = v1 + v2;
	if (id == 12) v1 = v1 + v2;
	if (id == 14) v1 = v1 + v2;
	if (id == 16) v1 = v1 + v2;
	if (id == 18) v1 = v1 + v2;
	if (id == 20) v1 = v1 + v2;
	if (id == 22) v1 = v1 + v2;
	if (id == 24) v1 = v1 + v2;
	if (id == 26) v1 = v1 + v2;
	if (id == 28) v1 = v1 + v2;
	if (id == 30) v1 = v1 + v2;
	if (id == 32) v1 = v1 + v2;
	__syncthreads();
	return( v1 );
}



static __global__ void scan1( FPT * a  )
{
	const UIN tidBLCK = threadIdx.x;
	const UIN tidWARP = tidBLCK & 31;
	const UIN widGRID = tidBLCK >> 5;
	      FPT v1      = a[tidBLCK];
	      FPT v2;
	__shared__ FPT blk1[32];

	if (widGRID == 0) blk1[tidWARP] = 0.0;
	__syncthreads();

	v1 = warp_scan1( tidWARP, v1 );
	__syncthreads();

	if (tidWARP == 31) blk1[widGRID] = v1;
	__syncthreads();

	if (widGRID == 0)
	{
		v2 = blk1[tidWARP];
		v2 = warp_scan1( tidWARP, v2 );
		blk1[tidWARP] = v2;
	}
	__syncthreads();

	if (widGRID >= 1) v1 = v1 + blk1[widGRID-1];
	__syncthreads();

	a[tidBLCK] = v1;
	return;
}



static __global__ void scan2( FPT * a )
{
	const UIN tidBLCK = threadIdx.x;
	const UIN tidWARP = (tidBLCK & 31);
	const UIN widGRID = tidBLCK >> 5;
	      FPT v1      =  a[tidBLCK];
	      FPT v2;
	__shared__ FPT blk1[32];

	if (widGRID == 0) blk1[tidWARP] = 0.0;
	__syncthreads();

	v1 = warp_scan2( tidWARP, v1 );
	__syncthreads();

	if (tidWARP == 31) blk1[widGRID] = v1;
	__syncthreads();

	if (widGRID == 0)
	{
		v2 = blk1[tidWARP];
		v2 = warp_scan2( tidWARP, v2 );
		blk1[tidWARP] = v2;
	}
	__syncthreads();

	if (widGRID >= 1) v1 = v1 + blk1[widGRID-1];
	__syncthreads();

	a[tidBLCK] = v1;
	return;
}



static __global__ void scan3( FPT * a )
{
	const UIN tidBLCK = threadIdx.x;
	const UIN tidWARP = (tidBLCK & 31);
	const UIN widGRID = tidBLCK >> 5;
	      FPT v1      =  a[tidBLCK];
	      FPT v2, v3 = 0.0;
	__shared__ FPT blk1[32];

	if (widGRID == 0) blk1[tidWARP] = 0.0;
	__syncthreads();

	v1 = warp_scan3( tidWARP, v1 );
	__syncthreads();

	if (tidWARP == 31) blk1[widGRID] = v1;
	__syncthreads();

	if (widGRID == 0)
	{
		v2 = blk1[tidWARP];
		v3 = warp_scan3( tidWARP, v2 );
		blk1[tidWARP] = v3;
	}
	__syncthreads();

	if (widGRID >= 1) v1 = v1 + blk1[widGRID-1];
	__syncthreads();

	a[tidBLCK] = v1;
	return;
}



typedef struct { char n[48]; double t; FPT r; } str_res;



static __host__ str_res test_scan1( const UIN cbs, const UIN al, FPT * array )
{
	const UIN cbn = (al + cbs - 1) / cbs;
	FPT * d_a; HANDLE_CUDA_ERROR( cudaMalloc( &d_a, al * sizeof(FPT) ) ); TEST_POINTER( d_a );
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemcpy( d_a, array, al * sizeof(FPT), cudaMemcpyHostToDevice ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		scan1 <<<cbn, cbs>>> ( d_a );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( array, d_a, al * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	HANDLE_CUDA_ERROR( cudaFree( d_a ) );
	str_res sr;
	strcpy( sr.n, "scan1" );
	sr.t = (double) tt / (double) NUM_ITE;
	sr.r = array[al-1];
	return( sr );
}



static __host__ str_res test_scan2( const UIN cbs, const UIN al, FPT * array )
{
	const UIN cbn = (al + cbs - 1) / cbs;
	FPT * d_a; HANDLE_CUDA_ERROR( cudaMalloc( &d_a, al * sizeof(FPT) ) ); TEST_POINTER( d_a );
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemcpy( d_a, array, al * sizeof(FPT), cudaMemcpyHostToDevice ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		scan2 <<<cbn, cbs>>> ( d_a );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( array, d_a, al * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	HANDLE_CUDA_ERROR( cudaFree( d_a ) );
	str_res sr;
	strcpy( sr.n, "scan2" );
	sr.t = (double) tt / (double) NUM_ITE;
	sr.r = array[al-1];
	return( sr );
}



static __host__ str_res test_scan3( const UIN cbs, const UIN al, FPT * array )
{
	const UIN cbn = (al + cbs - 1) / cbs;
	FPT * d_a; HANDLE_CUDA_ERROR( cudaMalloc( &d_a, al * sizeof(FPT) ) ); TEST_POINTER( d_a );
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemcpy( d_a, array, al * sizeof(FPT), cudaMemcpyHostToDevice ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		scan3 <<<cbn, cbs>>> ( d_a );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( array, d_a, al * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	HANDLE_CUDA_ERROR( cudaFree( d_a ) );
	str_res sr;
	strcpy( sr.n, "scan3" );
	sr.t = (double) tt / (double) NUM_ITE;
	sr.r = array[al-1];
	return( sr );
}



#endif



int main( int argc, char ** argv )
{
	// check input arguments
	str_inputArgs sia = checkArgs( argc, argv );

	// print run settings
	printRunSettings( sia );

	// count available GPU devices
	int cudaDeviceCounter;
	getCudaDeviceCounter( &cudaDeviceCounter );

	// print GPUs' info
	if ( cudaDeviceCounter > 0 )
	{
		int cudaDeviceID;
		for ( cudaDeviceID = 0; cudaDeviceID < cudaDeviceCounter; cudaDeviceID++ )
			printCudaDeviceProperties( cudaDeviceID );
		cudaDeviceID = DEVICE;
		HANDLE_CUDA_ERROR( cudaSetDevice( cudaDeviceID) );
		HANDLE_CUDA_ERROR( cudaGetDevice(&cudaDeviceID) );
		printf( "cudaDeviceSelected:                               %d <-------------------\n", cudaDeviceID ); fflush(stdout);
	}

	FPT * array1 = (FPT *) calloc( sia.al, sizeof(FPT) ); TEST_POINTER( array1 );
	FPT * array2 = (FPT *) calloc( sia.al, sizeof(FPT) ); TEST_POINTER( array2 );
	FPT * array3 = (FPT *) calloc( sia.al, sizeof(FPT) ); TEST_POINTER( array3 );
	init_vec( sia.ompMT, sia.al, array1 );
	init_vec( sia.ompMT, sia.al, array2 );
	init_vec( sia.ompMT, sia.al, array3 );
	str_res sr1 = test_scan1( sia.cbs, sia.al, array1 );
	str_res sr2 = test_scan2( sia.cbs, sia.al, array2 );
	str_res sr3 = test_scan3( sia.cbs, sia.al, array3 );

	HDL; printf( "results\n" ); HDL;
	printf( "%10s %8s %13s\n", "function", "reduction", "time" );
	printf( "%10s %8.0lf %13.8lf\n", sr1.n, sr1.r, sr1.t );
	printf( "%10s %8.0lf %13.8lf\n", sr2.n, sr2.r, sr2.t );
	printf( "%10s %8.0lf %13.8lf\n", sr3.n, sr3.r, sr3.t );

//	UIN i = 0;
//	for ( ; i < sia.al; i++ )
//		printf( "%4d %4.0lf %4.0lf %4.0lf\n", i, array1[i], array2[i], array3[i] );


	return( EXIT_SUCCESS );
}



