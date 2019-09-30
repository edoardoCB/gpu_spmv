// ┌────────────────────────────────┐
// │program: reduce.cu              │
// │author: Edoardo Coronado        │
// │date: 21-08-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#ifndef __REDUCE_HEADER__
#define __REDUCE_HEADER__



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cusparse.h>
#ifdef _OMP_
	#include <omp.h>
	#ifndef OMP_SCH
		#define OMP_SCH static
		char omp_schedule[7] = "static";
	#endif
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
	printf( "FPT:           %s\n", fptMsg    );
	printf( "sizeof(FPT):   %zu bytes\n", sizeof(FPT) );
	printf( "arrayLength:   %d\n", sia.al    );
	printf( "cudaBlockSize: %d\n", sia.cbs   );
	printf( "ompMaxThreads: %d\n", sia.ompMT ); fflush(stdout);
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



static __global__ void red1( a  )
{
	const UIN tidGRID = blockDim.x * blockIdx.x + threadIdx.x;i
	      FPT val     = a[tidGRID];
	val = val + __shfl_down_sync( FULL_MASK, val, 16 );
	val = val + __shfl_down_sync( FULL_MASK, val,  8 );
	val = val + __shfl_down_sync( FULL_MASK, val,  4 );
	val = val + __shfl_down_sync( FULL_MASK, val,  2 );
	val = val + __shfl_down_sync( FULL_MASK, val,  1 );
	   red = red + val;
	return;


typedef struct { double t; FPT r; } str_res;



static __host__ str_res test_red1( const cbs, const UIN al, FPT * array )
{
	const UIN cbn = (al + cbs - 1) / cbs;
	FPT * d_a; HANDLE_CUDA_ERROR( cudaMalloc( &d_a, al * sizeof(FPT) ) ); TEST_POINTER( d_a );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_a, array, al * sizeof(FPT), cudaMemcpyHostToDevice ) );
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		red1 <<<cbn, cbs>>> ( d_a );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	ti = tt / NUM_ITE;
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( array, d_a, al * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	HANDLE_CUDA_ERROR( cudaFree( d_a ) );
	str_res sr;
	sr.t = ti;
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

	FPT * array = (FPT *) calloc( sia.al, sizeof(FPT) ); TEST_POINTER( array );
	init_vec( sia.ompMT, sia.al, array );

	str_res sr = test_red1( sia.cbs, sia.al, array );


	return( EXIT_SUCCESS );
}



