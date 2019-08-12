// ┌────────────────────────────────┐
// │program: gpuSpmv_header.h       │
// │author: Edoardo Coronado        │
// │date: 05-06-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#ifndef __GPU_SPMV_HEADER__
#define __GPU_SPMV_HEADER__



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include <cusparse.h>

#ifdef _OMP_
	#include <omp.h>
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
	#define HDL { printf( "-------------------------------------------------------------------------------------\n" ); }
#endif



#ifndef BM
	#define BM { fflush( stdout ); printf( "\nFile: %s    Line: %d.\n", __FILE__, __LINE__ ); fflush( stdout ); }
#endif



#ifndef NUM_ITE
	#define NUM_ITE 250
#endif



#ifndef HBRICK_SIZE
	#define HBRICK_SIZE 32
#endif



#ifndef CHUNK_SIZE
	#define CHUNK_SIZE 32
#endif



typedef struct { char matFileName[48]; UIN cudaBlockSize; UIN ompMaxThreads; } str_inputArgs;



static __host__ str_inputArgs checkArgs( const UIN argc, char ** argv )
{
	if ( argc < 3 )
	{
		printf( "\n\tMissing input arguments.\n" );
		printf( "\n\tUsage:\n\n\t\t%s <matFileName> <cudaBlockSize>\n\n", argv[0] );
		printf( "\t\t\t<matFileName>:   file's name that contains the matrix in CSR format [string].\n" );
		printf( "\t\t\t<cudaBlockSize>: number of threads per block [integer].\n\n" );
		exit( EXIT_FAILURE );
	}
	str_inputArgs sia;
	strcpy( sia.matFileName, argv[1] );
	sia.cudaBlockSize = atoi( argv[2] );
	#ifdef _OMP_
		#pragma omp parallel
		{
			#pragma omp master
				sia.ompMaxThreads = omp_get_max_threads();
		}
	#else
		sia.ompMaxThreads = 1;
	#endif
	return( sia );
}



static __host__ void printRunSettings( const str_inputArgs sia )
{
	HDL; printf( "run settings\n" ); HDL;
	#ifdef _CIRRUS_
	printf( "hostname:           %s\n", "cirrus.EPCC" );
	#endif
	#ifdef _KAY_
	printf( "hostname:           %s\n", "kay.ICHEC" );
	#endif
	#ifdef _CTGPGPU2_
	printf( "hostname:           %s\n", "ctgpgpu2.CITIUS" );
	#endif
	printf( "srcFileName:        %s\n", __FILE__ );
	printf( "matFileName:        %s\n", sia.matFileName );
	printf( "cudaBlockSize:      %d\n", sia.cudaBlockSize );
	#ifdef __CUDA_ARCH__
	printf( "__CUDA_ARCH__:      %d\n", (UIN) __CUDA_ARCH__ );
	#endif
	#ifdef _OMP_
	printf( "ompMaxThreads:      %d\n", sia.ompMaxThreads );
	#endif
	printf( "FPT:                %s\n", fptMsg );
	printf( "sizeof(FPT):        %zu bytes\n", sizeof(FPT) );
	printf( "NUM_ITE:            %d\n", (UIN) NUM_ITE );
	printf( "HBRICK_SIZE:        %d\n", (UIN) HBRICK_SIZE );
	printf( "CHUNK_SIZE:         %d\n", (UIN) CHUNK_SIZE );
	return;
}



#ifndef HANDLE_CUDA_ERROR
	#define HANDLE_CUDA_ERROR( ceID ) { if ( ceID != cudaSuccess ) { printf( "FILE: %s LINE: %d CUDA_ERROR: %s\n", __FILE__, __LINE__, cudaGetErrorString( ceID ) ); printf( "\nvim %s +%d\n", __FILE__, __LINE__); exit( EXIT_FAILURE ); } }
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
	#define TEST_POINTER( p ) { if ( p == NULL ) { printf( "\nFile: %s Line: %d Pointer: %s is null\n", __FILE__, __LINE__, #p ); exit( EXIT_FAILURE ); } }
#endif



#ifndef ABORT
	#define ABORT { printf( "\nFile: %s Line: %d execution is aborted.\n", __FILE__, __LINE__ ); exit( EXIT_FAILURE ); }
#endif



typedef struct { UIN nrows; UIN nnz; UIN rmin; FPT rave; UIN rmax; FPT rsd; UIN bw; FPT * val; UIN * row; UIN * col; UIN * rl; } str_matCSR;



static __host__ str_matCSR matrixReading( const char * matFileName )
{
	str_matCSR matCSR;
	if ( strstr( matFileName, ".csr" ) != NULL )
	{
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .csr file\n\n" );
			exit( EXIT_FAILURE );
		}
		if ( fscanf( fh, "%d %d", &(matCSR.nrows), &(matCSR.nnz) ) != 2 ) ABORT;
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		int i;
		for ( i = 0; i < ( matCSR.nnz ); i++ )
		{
			#if FP_TYPE == FPT_FLOAT
				if ( fscanf( fh, "%f %d\n",  &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#else
				if ( fscanf( fh, "%lf %d\n", &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#endif
		}
		for ( i = 0; i < ( matCSR.nrows + 1 ); i++ )
			if ( fscanf( fh, "%d", &(matCSR.row[i]) ) != 1 ) ABORT;
		fclose( fh );
	}
	else if ( strstr( matFileName, ".bin" ) != NULL )
	{
		size_t aux = 0;
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .bin file\n\n" );
			exit( EXIT_FAILURE );
		}
		aux = fread( &(matCSR.nrows), sizeof(UIN), 1, fh );
		aux = fread( &(matCSR.nnz),   sizeof(UIN), 1, fh );
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		aux = fread( matCSR.val, sizeof(FPT),   matCSR.nnz,         fh );
		aux = fread( matCSR.col, sizeof(UIN),   matCSR.nnz,         fh );
		aux = fread( matCSR.row, sizeof(UIN), ( matCSR.nrows + 1 ), fh );
		aux++;
		fclose(fh);
	}
	else
	{
		char buffer[128];
		strcpy( buffer, "matrixReading detected that " );
		strcat( buffer, matFileName );
		strcat( buffer, " has NOT .csr or .bin extension" );
		printf( "\n%s\n\n", buffer );
		exit( EXIT_FAILURE );
	}
	return( matCSR );
}



static __host__ void printMatrixStats( const char * matFileName, str_matCSR * matCSR )
{
	UIN    i, rl, rmin = 1e9, rmax = 0, j, bw = 0;
	int    dif;
	double rave1 = 0.0, rave2 = 0.0, rsd = 0.0;
	for ( i = 0; i < matCSR->nrows; i++ )
	{
		rl            = matCSR->row[i + 1] - matCSR->row[i];
		matCSR->rl[i] = rl;
		rave1         = rave1 +   rl;
		rave2         = rave2 + ( rl * rl );
		rmin          = min( rmin, rl );
		rmax          = max( rmax, rl );
		for ( j = matCSR->row[i]; j < matCSR->row[i+1]; j++ )
		{
			dif = abs( ((int) i) - ((int) matCSR->col[j]) );
			bw  = ( dif > bw ) ? dif : bw ;
		}
	}
	rave1 = rave1 / (double) (matCSR->nrows);
	rave2 = rave2 / (double) (matCSR->nrows);
	rsd   = sqrt( rave2 - ( rave1 * rave1 ) );
	matCSR->rmin = rmin;
	matCSR->rave = rave1;
	matCSR->rmax = rmax;
	matCSR->rsd  = rsd;
	matCSR->bw   = bw;
	char name[64];
	strcpy( name, matFileName );
	char * token1;
	const char deli[2] = ".";
	token1 = strtok( name, deli );
	strcat( token1, ".sta" );
	FILE * fh;
	fh = fopen( name, "w+" );
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "matrix's statistics\n");
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "name:  %28s\n",    matFileName );
	fprintf( fh, "nrows: %28d\n",    matCSR->nrows );
	fprintf( fh, "nnz:   %28d\n",    matCSR->nnz );
	fprintf( fh, "rmin:  %28d\n",    matCSR->rmin );
	fprintf( fh, "rave:  %28.2lf\n", matCSR->rave );
	fprintf( fh, "rmax:  %28d\n",    matCSR->rmax );
	fprintf( fh, "rsd:   %28.2lf\n", matCSR->rsd );
	fprintf( fh, "rsdp:  %28.2lf\n", ( ( rsd / rave1 ) * 100 ) );
	fprintf( fh, "bw:    %28d\n",    matCSR->bw );
	fclose( fh );
	return;
}



typedef struct { char name[18]; double mfp; double beta; double ct; } str_formatData;



static __host__ str_formatData getFormatDataCSR( const str_matCSR matCSR )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "CSR" );
	// CSR memory footprint
	fd.mfp =          (double) (   matCSR.nnz         * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) (   matCSR.nnz         * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( ( matCSR.nrows + 1 ) * sizeof(UIN) ); // row
	fd.mfp = fd.mfp + (double) (   matCSR.nrows       * sizeof(FPT) ); // vec
	// CSR occupancy ( beta )
	fd.beta = ( (double) matCSR.nnz / (double) matCSR.nnz );
	// CSR conversion time
	fd.ct = 0.0;
	return( fd );
}



static __host__ void initVec( const UIN len, FPT * vec )
{
	UIN i;
	for ( i = 0 ; i < len; i++ )
		vec[i] = (FPT) i;
	return;
}



#ifndef GT
	#define GT( t ) { gettimeofday( &t, NULL ); }
#endif



static __host__ void cf_CSR( const str_matCSR matCSR, const FPT * vec, FPT * res )
{
	UIN i, j;
	FPT aux;
	for ( i = 0; i < matCSR.nrows; i++ )
	{
		aux = (FPT) 0;
		for ( j = matCSR.row[i]; j < matCSR.row[i+1]; j++ )
		{
			aux = aux + matCSR.val[j] * vec[matCSR.col[j]];
		}
		res[i] = aux;
	}
	return;
}



static __host__ double measureTime( const struct timeval t2, const struct timeval t1 )
{
	double t = (double) ( t2.tv_sec - t1.tv_sec ) + ( (double) ( t2.tv_usec - t1.tv_usec ) ) * 1e-6;
	return( t );
}



typedef struct { double aErr; double rErr; UIN pos; } str_err;



static __host__ void getErrors( const UIN len, const FPT * ar, const FPT * ac, str_err * sErr )
{
	double dif, maxDif = 0.0;
	double val, maxVal = 0.0;
	UIN pos = 0;
	UIN i;
	for ( i = 0; i < len; i++ )
	{
		val = fabs(ar[i]);
		if ( val > maxVal ) maxVal = val;
		dif = fabs( fabs(ac[i]) - val );
		if ( dif > maxDif )
		{
			maxDif = dif;
			pos    = i;
		}
	}
	sErr->aErr = maxDif;
	sErr->rErr = maxDif/maxVal;
	sErr->pos  = pos;
	return;
}



typedef struct { char name[24]; double et; double flops; str_err sErr; } str_res;



static __host__ str_res test_cf_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		cf_CSR( matCSR, vec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "cf_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
	getErrors( matCSR.nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



static __global__ void gk_CSR( const UIN nrows, const FPT * val, const UIN * col, const UIN * row, const FPT * x, FPT * y )
{
	const UIN rowID = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rowID < nrows )
	{
		UIN i;
		FPT aux = 0.0;
		for ( i = row[rowID]; i < row[rowID + 1]; i++ )
			aux = aux + val[i] * x[col[i]];
		y[rowID] = aux;
	}
	return;
}



static __host__ str_res test_gk_CSR( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// get parameters
	const UIN        nrows = matCSR.nrows;
	const UIN          nnz = matCSR.nnz;
	const UIN cudaBlockNum = ( nrows + cudaBlockSize - 1 ) / cudaBlockSize;
	// allocate memory on GPU
	FPT * d_val; HANDLE_CUDA_ERROR( cudaMalloc( &d_val,          nnz * sizeof(FPT) ) ); TEST_POINTER( d_val );
	UIN * d_col; HANDLE_CUDA_ERROR( cudaMalloc( &d_col,          nnz * sizeof(UIN) ) ); TEST_POINTER( d_col );
	UIN * d_row; HANDLE_CUDA_ERROR( cudaMalloc( &d_row, (nrows + 1 ) * sizeof(UIN) ) ); TEST_POINTER( d_row );
	FPT * d_vec; HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,        nrows * sizeof(FPT) ) ); TEST_POINTER( d_vec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,        nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val, matCSR.val,          nnz * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col, matCSR.col,          nnz * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_row, matCSR.row, ( nrows + 1 )* sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec, vec,               nrows * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_CSR <<<cudaBlockNum, cudaBlockSize>>> ( nrows, d_val, d_col, d_row, d_vec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col ) );
	HANDLE_CUDA_ERROR( cudaFree( d_row ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_CSR" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
	getErrors( matCSR.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __host__ const char * cusparseGetErrorMessage( cusparseStatus_t statusID )
{
	switch(statusID)
	{
		case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
		case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
		case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
		case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
		case CUSPARSE_STATUS_MAPPING_ERROR:             return "CUSPARSE_STATUS_MAPPING_ERROR";
		case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
		case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	}
	return "<cusparse unknown>";
}



#ifndef HANDLE_CUSPARSE_ERROR
	#define HANDLE_CUSPARSE_ERROR( cseID ) { if ( cseID != CUSPARSE_STATUS_SUCCESS ) { printf( "FILE: %s LINE: %d CUBLAS_ERROR: %s\n", __FILE__, __LINE__, cusparseGetErrorMessage( cseID ) ); printf( "\nvim %s +%d\n", __FILE__, __LINE__); exit( EXIT_FAILURE ); } }
#endif



static __host__ str_res test_gcu_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// get parameteres for cuSPARSE
	const UIN                     nrows = matCSR.nrows;
	const UIN                       nnz = matCSR.nnz;
	      cusparseHandle_t    cusparseH = NULL;
	const cusparseAlgMode_t  cusparseAM = CUSPARSE_ALG1;
	const cusparseOperation_t cusparseO = CUSPARSE_OPERATION_NON_TRANSPOSE;
	      cusparseMatDescr_t cusparseMD = NULL;
	      size_t    cudaSpaceBufferSize;
	const FPT                      zero = (FPT)  0;
	const FPT                       one = (FPT)  1;
	#if FP_TYPE == FP_FLOAT
		cudaDataType cudaDT = CUDA_R_32F;
	#else
		cudaDataType cudaDT = CUDA_R_64F;
	#endif
	// create handlers for cuSPARSE
	HANDLE_CUSPARSE_ERROR( cusparseCreate(&cusparseH) );
	HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr( &cusparseMD ) );
	HANDLE_CUSPARSE_ERROR( cusparseSetMatIndexBase( cusparseMD, CUSPARSE_INDEX_BASE_ZERO ) );
	HANDLE_CUSPARSE_ERROR( cusparseSetMatType( cusparseMD, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	// allocate memory on GPU
	FPT * d_val; HANDLE_CUDA_ERROR( cudaMalloc( &d_val,           nnz * sizeof(FPT) ) ); TEST_POINTER( d_val );
	int * d_col; HANDLE_CUDA_ERROR( cudaMalloc( &d_col,           nnz * sizeof(int) ) ); TEST_POINTER( d_col );
	int * d_row; HANDLE_CUDA_ERROR( cudaMalloc( &d_row, ( nrows + 1 ) * sizeof(int) ) ); TEST_POINTER( d_row );
	FPT * d_vec; HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,         nrows * sizeof(FPT) ) ); TEST_POINTER( d_vec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,         nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val, matCSR.val,           nnz * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col, matCSR.col,           nnz * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_row, matCSR.row, ( nrows + 1 ) * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec, vec,                nrows * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// get space buffer for cusparseCsrmvEx
	HANDLE_CUSPARSE_ERROR( cusparseCsrmvEx_bufferSize( cusparseH, cusparseAM, cusparseO, matCSR.nrows, matCSR.nrows, matCSR.nnz, &one, cudaDT, cusparseMD, \
                                                        d_val, cudaDT, d_row, d_col, d_vec, cudaDT, &zero, cudaDT, d_res, cudaDT, cudaDT, &cudaSpaceBufferSize ) );
	void * cudaSpaceBuffer; HANDLE_CUDA_ERROR( cudaMalloc( &cudaSpaceBuffer, cudaSpaceBufferSize ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0, tt = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		HANDLE_CUSPARSE_ERROR( cusparseCsrmvEx( cusparseH, cusparseAM, cusparseO, matCSR.nrows, matCSR.nrows, matCSR.nnz, &one, cudaDT, cusparseMD, \
                                                  d_val, cudaDT, d_row, d_col, d_vec, cudaDT, &zero, cudaDT, d_res, cudaDT, cudaDT, cudaSpaceBuffer ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col ) );
	HANDLE_CUDA_ERROR( cudaFree( d_row ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gcu_CSR" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( (double) matCSR.nnz * 2.0 ) / sr.et;
	getErrors( matCSR.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN lenAX; UIN lenBRP; FPT * ax; UIN * brp; } str_matAXC;



static __host__ UIN getArrayBrpAXC( const str_matCSR matCSR, str_matAXC * matAXC )
{
	UIN rowID = 0, brickNum = 0;
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		brickNum                 = ( matCSR.rl[rowID] + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
		matAXC->brp[rowID + 1]   = matAXC->brp[rowID]  + ( 2 * brickNum * HBRICK_SIZE );
	}
	return( matAXC->brp[matAXC->nrows] );
}



static __host__ void getArrayAxAXC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	const UIN nrows = matAXC->nrows;
	UIN rowID, posAX, counter, posCSR;
	#pragma omp parallel for default(shared) private(rowID,posAX,counter,posCSR) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		  posAX = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			              matAXC->ax[posAX] = matCSR.val[posCSR];
			matAXC->ax[posAX + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			if ( counter == (HBRICK_SIZE - 1) )
			{
				posAX  = posAX + 1 + HBRICK_SIZE;
				counter = 0;
			}
			else
			{
				posAX++;
				counter++;
			}
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXC( const UIN ompNumThreads, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	// get AXC parameters
	 matAXC->nrows = matCSR.nrows;
	   matAXC->nnz = matCSR.nnz;
	matAXC->lenBRP = matCSR.nrows + 1;
	   matAXC->brp = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXC
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = getArrayBrpAXC( matCSR, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXC->ax = (FPT *) calloc( matAXC->lenAX, sizeof(FPT) ); TEST_POINTER( matAXC->ax );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayAxAXC( ompNumThreads, matCSR, vec, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXC" );
	// AXC memory footprint
	fd.mfp =          (double) ( matAXC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXC->lenBRP * sizeof(UIN) ); // brp ( stores the starting address of a row )
	// AXC occupancy ( beta )
	fd.beta = ( (double) matAXC->nnz / (double) (matAXC->lenAX >> 1) );
	// AXC conversion time
	fd.ct = tc;
	return( fd );
}



#ifndef FULL_MASK
	#define FULL_MASK 0xffffffff
#endif



static __global__ void gk_AXC( const FPT * ax, const UIN * brp, FPT * y )
{
	const UIN  tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN  widGRID = tidGRID >> 5;
	const UIN  tidWARP = tidGRID & 31;
	const UIN   offset = brp[widGRID];
	const UIN   stride = 2 * HBRICK_SIZE;
	const UIN upperLim = brp[widGRID+1];
	      UIN pos;
	      FPT val;
	      FPT aux = 0.0;
	for ( pos = offset + tidWARP; pos < upperLim; pos = pos + stride )
	{
		val = ax[pos] * ax[pos+HBRICK_SIZE];
		val = val + __shfl_down_sync( FULL_MASK, val, 16 );
		val = val + __shfl_down_sync( FULL_MASK, val,  8 );
		val = val + __shfl_down_sync( FULL_MASK, val,  4 );
		val = val + __shfl_down_sync( FULL_MASK, val,  2 );
		val = val + __shfl_down_sync( FULL_MASK, val,  1 );
		if (tidWARP == 0) aux = aux + val;
	}
	if (tidWARP == 0) y[widGRID] = aux;
	return;
}



static __host__ str_res test_gk_AXC( const UIN cudaBlockSize, const str_matAXC matAXC, const FPT * ref )
{
	// get parameters
	const UIN        nrows = matAXC.nrows;
	const UIN cudaBlockNum = ( (nrows * 32) + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN      wrpsGen = cudaBlockNum * wrpsPerBlock;
	const UIN          dif = wrpsGen - nrows;
	const UIN     devLenAX = matAXC.lenAX  + dif * 32;
	const UIN    devLenBRP = matAXC.lenBRP + dif;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_brp; HANDLE_CUDA_ERROR( cudaMalloc( &d_brp, devLenBRP * sizeof(UIN) ) ); TEST_POINTER( d_brp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,     nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_brp, 0, devLenBRP * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0,     nrows * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_brp, matAXC.brp, matAXC.lenBRP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXC <<<cudaBlockNum, cudaBlockSize>>> (  d_ax, d_brp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_brp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXC" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN ind; UIN val; } str_pair;



typedef struct { UIN nrows; UIN nnz; UIN chunkNum; UIN lenVC; UIN * permi; UIN * nmc; UIN * chp; FPT * val; UIN * col; } str_matK1;



static __host__ int orderFunction( const void * ele1, const void * ele2 )
{
	return (  ( (str_pair *) ele2 )->val - ( (str_pair *) ele1 )->val  );
}



static __host__ void getArrayPermiK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	str_pair * list = (str_pair *) malloc( matCSR.nrows * sizeof(str_pair) ); TEST_POINTER( list );
	UIN i;
	for ( i = 0; i < matK1->nrows; i++ )
	{
		list[i].ind = i;
		list[i].val = matCSR.rl[i];
	}
	qsort( list, matK1->nrows, sizeof(str_pair), orderFunction );
	for ( i = 0; i < matK1->nrows; i++ )
		matK1->permi[i] = list[i].ind;
	free( list );
	return;
}



static __host__ UIN getArraysNmcChpK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	UIN i, p, n, l = 0;
	for ( i = 0 ; i < matK1->chunkNum; i++ )
	{
		p             = matK1->permi[i * CHUNK_SIZE];
		n             = matCSR.rl[p];
		matK1->nmc[i] = n;
		l             = l + CHUNK_SIZE * n;
	}
	for ( i = 1; i < matK1->chunkNum; i++ )
		matK1->chp[i] = matK1->chp[i-1] + ( matK1->nmc[i-1] * CHUNK_SIZE );
	return l;
}



static __host__ void getArraysValColK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	UIN i1 = 0, i2, i3, i4, ri, rp, ep;
	for ( ; i1 < matK1->chunkNum; i1++ )
	{
		for ( i2 = 0; i2 < CHUNK_SIZE; i2++ )
		{
			ri = i1 * CHUNK_SIZE + i2;
			if ( ri == matCSR.nrows ) return;
			rp = matK1->permi[ri];
			for ( i3 = matCSR.row[rp], i4 = 0; i3 < matCSR.row[rp + 1]; i3++, i4++ )
			{
				ep             = matK1->chp[i1] + i4 * CHUNK_SIZE + i2;
				matK1->val[ep] = matCSR.val[i3];
				matK1->col[ep] = matCSR.col[i3];
			}
		}
	}
	return;
}



static __host__ str_formatData getFormatDataK1( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, str_matK1 * matK1 )
{
	// get K1 parameters
	UIN cudaBlockNum = ( matCSR.nrows + cudaBlockSize - 1 ) / cudaBlockSize;
	matK1->nrows     = matCSR.nrows;
	matK1->nnz       = matCSR.nnz;
	matK1->chunkNum  = ( cudaBlockNum * cudaBlockSize ) / CHUNK_SIZE;
	matK1->permi     = (UIN *) calloc( matK1->chunkNum * CHUNK_SIZE, sizeof(UIN) ); TEST_POINTER( matK1->permi );
	matK1->nmc       = (UIN *) calloc( matK1->chunkNum,              sizeof(UIN) ); TEST_POINTER( matK1->nmc   );
	matK1->chp       = (UIN *) calloc( matK1->chunkNum,              sizeof(UIN) ); TEST_POINTER( matK1->chp   );
	// get matK1
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayPermiK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matK1->lenVC = getArraysNmcChpK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matK1->val = (FPT *) calloc( matK1->lenVC, sizeof(FPT) ); TEST_POINTER( matK1->val );
	matK1->col = (UIN *) calloc( matK1->lenVC, sizeof(UIN) ); TEST_POINTER( matK1->col );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysValColK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "K1" );
	// K1 memory footprint
	fd.mfp =          (double) ( matK1->chunkNum              * sizeof(UIN) ); // nmc
	fd.mfp = fd.mfp + (double) ( matK1->chunkNum              * sizeof(UIN) ); // chp
	fd.mfp = fd.mfp + (double) ( matK1->lenVC                 * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) ( matK1->lenVC                 * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( matK1->chunkNum * CHUNK_SIZE * sizeof(UIN) ); // permi
	fd.mfp = fd.mfp + (double) ( matK1->nrows                 * sizeof(FPT) ); // vec
	// K1 occupancy ( beta )
	fd.beta = ( (double) matK1->nnz / (double) (matK1->lenVC) );
	// K1 conversion time
	fd.ct = tc;
	return( fd );
}



static __global__ void gk_K1( const int NROWS, const FPT * val, const UIN * col, const UIN * nmc, const UIN * chp, const UIN * permi, const FPT * x, FPT * y )
{
	const UIN gid = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN lid = threadIdx.x;
	const UIN cid = gid / CHUNK_SIZE;
	const UIN wid = lid & ( CHUNK_SIZE - 1 );
	if ( gid < NROWS )
	{
		UIN to = chp[cid] + wid;
		UIN ul = nmc[cid] * CHUNK_SIZE + to;
		FPT sum = val[to] * x[col[to]];
		for ( to = ( to + CHUNK_SIZE ); to < ul; to = ( to + CHUNK_SIZE ) )
			sum = sum + val[to] * x[col[to]];
		y[permi[gid]] = sum;
	}
	return;
}



static __host__ str_res test_gk_K1( const UIN cudaBlockSize, const str_matK1 matK1, const FPT * vec, const FPT * ref )
{
	// 
	UIN cudaBlockNum = ( matK1.nrows + cudaBlockSize - 1 ) / cudaBlockSize;
	// allocate memory on GPU
	FPT * d_val;   HANDLE_CUDA_ERROR( cudaMalloc( &d_val,   matK1.lenVC                 * sizeof(FPT) ) ); TEST_POINTER( d_val   );
	UIN * d_col;   HANDLE_CUDA_ERROR( cudaMalloc( &d_col,   matK1.lenVC                 * sizeof(UIN) ) ); TEST_POINTER( d_col   );
	UIN * d_nmc;   HANDLE_CUDA_ERROR( cudaMalloc( &d_nmc,   matK1.chunkNum              * sizeof(UIN) ) ); TEST_POINTER( d_nmc   );
	UIN * d_chp;   HANDLE_CUDA_ERROR( cudaMalloc( &d_chp,   matK1.chunkNum              * sizeof(UIN) ) ); TEST_POINTER( d_chp   );
	UIN * d_permi; HANDLE_CUDA_ERROR( cudaMalloc( &d_permi, matK1.chunkNum * CHUNK_SIZE * sizeof(UIN) ) ); TEST_POINTER( d_permi );
	FPT * d_vec;   HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,   matK1.nrows                 * sizeof(FPT) ) ); TEST_POINTER( d_vec   );
	FPT * d_res;   HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matK1.nrows                 * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val,   matK1.val,   matK1.lenVC                 * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col,   matK1.col,   matK1.lenVC                 * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_nmc,   matK1.nmc,   matK1.chunkNum              * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_chp,   matK1.chp,   matK1.chunkNum              * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_permi, matK1.permi, matK1.chunkNum * CHUNK_SIZE * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec,   vec,         matK1.nrows                 * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_K1 <<<cudaBlockNum, cudaBlockSize>>> (  matK1.nrows, d_val, d_col, d_nmc, d_chp, d_permi, d_vec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matK1.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matK1.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_nmc   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_chp   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_permi ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res   ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_K1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matK1.nnz ) ) / sr.et;
	getErrors( matK1.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct{ UIN nrows; UIN nnz; char mode[8]; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenSEC; UIN lenCON; UIN log; FPT * ax; UIN * sec; UIN * con; } str_matAXT;



static __host__ void getArraysLenAXT( const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN  nrows = matAXT->nrows;
	const UIN    thw = matAXT->tileHW;
	const UIN     th = matAXT->tileH;
	const UIN    ths = thw * th;
	const UIN grpLen = (th == 1) ? (thw) : (th) ;
	char mode[8];
	strcpy( mode, matAXT->mode );
	      UIN rowID = 0, rowStartPos = 0, rowOffT, rowOffR, rowOffC, pos, rowLen, positions, columns, totalColumns = 0, totalTiles;
	for ( ; rowID < nrows; rowID++ )
	{
		           rowOffT = ( (rowStartPos + ths)/ths ) - 1;
		           rowOffR =    rowStartPos % th;
		           rowOffC = ( (rowStartPos + th)/th ) - 1 - (rowOffT * thw);
		               pos = rowOffT * (2 * ths) + rowOffR * (2 * thw) + rowOffC;
		matAXT->con[rowID] = pos;
		            rowLen = matCSR.rl[rowID];
		         positions = ( strcmp( mode, "NOC" ) == 0 ) ? ( ( ( rowLen + grpLen - 1 ) / grpLen ) * grpLen ) : ( rowLen ) ;
		           columns = ( positions + th - 1 ) / th;
		      totalColumns = totalColumns + columns;
		       rowStartPos = rowStartPos + positions;
	}
	     totalTiles = ( totalColumns + thw - 1 ) / thw;
	 matAXT->tileN = totalTiles;
	 matAXT->lenAX = totalTiles * 2 * ths;
	if      ( (strcmp(mode, "NOC")==0) && (th==1) ) matAXT->lenSEC = totalTiles;
	else if ( (strcmp(mode, "NOC")==0) && (th!=1) ) matAXT->lenSEC = totalTiles * thw;
	else                                            matAXT->lenSEC = totalTiles * ths;
	return;
}



static __host__ void getArraysAxSecAXT_NOC_H1( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths));
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX++;
				ctrEle++;
				if ((ctrEle%thw)==0)
				{
					posAX = posAX + thw;
					posSEC++;
				}
			}
		}
	}
	return;
}



static __host__ void getArraysAxSecAXT_NOC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) if(_OMP_)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths))*thw + posAX%thw;
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX                 = posAX  + 2 * thw;
				ctrEle++;
				if ((ctrEle%th) == 0)
				{
					posAX = posAX + 1 - (2 * th * thw);
					posSEC++;
					if (posAX%thw==0) posAX = posAX + ((2*th)-1) * thw;
				}
			}
		}
	}
	return;
}



static __host__ void getArraysAxSecAXT_COM_H1( const UIN log, const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, eleCtr, posCSR, til, col, posAX, posSEC, q1, q2, offset;
	//#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			eleCtr = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				til                   = ((posCSR+ths)/ths)-1;
				col                   = posCSR%thw;
				posAX                 = til * 2 * ths + col;
				posSEC                = til * ths + col;
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( ((posCSR%thw) == 0) || (eleCtr==0) )
				{
					q1     = rowLen - eleCtr - 1;
					q2     = til * 2 * ths + 31 - posAX;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[posSEC] = rowID<<log | offset;
				}
				eleCtr++;
			}
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXT( const UIN ompNT, const UIN bs, const UIN thw, const UIN th, const char * mode, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	// set AXT parameters
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->tileHW = thw;
	matAXT->tileH  = th;
	strcpy( matAXT->mode, mode );
	matAXT->lenCON = matCSR.nrows;
	   matAXT->con = (UIN *) calloc( matAXT->lenCON, sizeof(UIN) ); TEST_POINTER( matAXT->con );
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileH) >> i) == 1 ) matAXT->logTH = i;
	// get AXT arrays' length
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysLenAXT( matCSR, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// get arrays ax[] and sec[]
	matAXT->ax  = (FPT *) calloc( matAXT->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXT->ax  );
	matAXT->sec = (UIN *) calloc( matAXT->lenSEC, sizeof(UIN) ); TEST_POINTER( matAXT->sec );
	tt = 0.0;
	if (strcmp(mode,"NOC")==0)
	{
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_NOC_H1( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_NOC( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	else
	{
		for ( i = 1; i < 10; i++ )
		{
			if ((bs>>i) == 1)
			{
				matAXT->log = i;
				break;
			}
		}
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM_H1( matAXT->log, ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXTC specific name
	str_formatData fd;
	char buffer[128];
	strcpy( buffer, "AXT_" );
	strcat( buffer, mode );
	strcat( buffer, "_H" );
	char TH[3]; sprintf( TH,  "%d", th );
	strcat( buffer, TH );
	strcat( buffer, "_HW" );
	char THW[3]; sprintf( THW, "%d", thw );
	strcat( buffer, THW );
	strcpy( fd.name, buffer );
	// AXTC memory footprint
	fd.mfp =          (double) ( matAXT->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXT->lenSEC * sizeof(UIN) ); // sec
	// AXTC occupancy ( beta )
	fd.beta = ( (double) matAXT->nnz / (double) (matAXT->lenAX >> 1) );
	// AXTC conversion time
	fd.ct = tc;
	return( fd );
}



#if __CUDA_ARCH__ < 600
static __device__ double customAtomicAdd( double * address, double val )
{
	unsigned long long int * address_as_ull = (unsigned long long int *) address;
	unsigned long long int              old = * address_as_ull;
	unsigned long long int          assumed;
	do {
		assumed = old;
		    old = atomicCAS( address_as_ull, assumed, __double_as_longlong( val + __longlong_as_double( assumed ) ) );
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while ( assumed != old );
	return __longlong_as_double( old );
}
#endif



static __global__ void gk_AXT_NOC_H1( const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN  tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN  widGRID = tidGRID >> 5;
	const UIN  tidWARP = tidGRID & 31;
	const UIN   offset = widGRID * 2 * thw;
	const UIN    rowID = rwp[widGRID];
	      UIN    posAX = offset + tidWARP;
	      FPT val = 0.0;
	val = ax[posAX] * ax[posAX+thw];
	val = val + __shfl_down_sync( FULL_MASK, val, 16 );
	val = val + __shfl_down_sync( FULL_MASK, val,  8 );
	val = val + __shfl_down_sync( FULL_MASK, val,  4 );
	val = val + __shfl_down_sync( FULL_MASK, val,  2 );
	val = val + __shfl_down_sync( FULL_MASK, val,  1 );
	#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
		if (tidWARP == 0) customAtomicAdd( &y[rowID], val );
	#else
		if (tidWARP == 0)       atomicAdd( &y[rowID], val );
	#endif
	return;
}



static __host__ str_res test_gk_AXT_NOC_H1( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref )
{
	// get parameters
	const UIN          thw = matAXT.tileHW;
	const UIN           tn = matAXT.tileN;
	const UIN        nrows = matAXT.nrows;
	const UIN          nnz = matAXT.nnz;
	const UIN cudaBlockNum = ( (tn * 32) + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN      wrpsGen = cudaBlockNum * wrpsPerBlock;
	const UIN          dif = wrpsGen - tn;
	const UIN     devLenAX = matAXT.lenAX  + dif * 32;
	const UIN    devLenSEC = matAXT.lenSEC + dif;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_sec; HANDLE_CUDA_ERROR( cudaMalloc( &d_sec, devLenSEC * sizeof(UIN) ) ); TEST_POINTER( d_sec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,     nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_sec, 0, devLenSEC * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0,     nrows * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_sec, matAXT.sec, matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXT_NOC_H1 <<<cudaBlockNum, cudaBlockSize>>> (  thw, d_ax, d_sec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_sec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	char buffer[128];
	strcpy( buffer, "gk_AXT_NOC_H1_HW" );
	char THW[3]; sprintf( THW, "%d", thw );
	strcat( buffer, THW );
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXT_NOC( const UIN th, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN   tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN   widGRID = tidGRID >> 5;
	const UIN   tidWARP = tidGRID & 31;
	const UIN    stride = 2 * thw;
	const UIN  offsetAX = widGRID * stride * th;
	const UIN  upperLim = (widGRID + 1) * stride * th;
	const UIN     rowID = rwp[(widGRID*thw) + tidWARP];
	      UIN     posAX = offsetAX + tidWARP;
	      FPT       val = ax[posAX] * ax[posAX+thw];
	for ( posAX = posAX + stride; posAX < upperLim; posAX = posAX + stride )
		val = val + ax[posAX] * ax[posAX+thw];
	#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
		customAtomicAdd( &y[rowID], val );
	#else
		atomicAdd( &y[rowID], val );
	#endif
	return;
}



static __host__ str_res test_gk_AXT_NOC( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref )
{
	// get parameters
	const UIN           th = matAXT.tileH;
	const UIN          thw = matAXT.tileHW;
	const UIN           tn = matAXT.tileN;
	const UIN        nrows = matAXT.nrows;
	const UIN          nnz = matAXT.nnz;
	const UIN cudaBlockNum = ( (tn * 32) + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN      wrpsGen = cudaBlockNum * wrpsPerBlock;
	const UIN          dif = wrpsGen - tn;
	const UIN     devLenAX = matAXT.lenAX  + dif * 32;
	const UIN    devLenSEC = matAXT.lenSEC + dif;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_sec; HANDLE_CUDA_ERROR( cudaMalloc( &d_sec, devLenSEC * sizeof(UIN) ) ); TEST_POINTER( d_sec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,     nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_sec, 0, devLenSEC * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0,     nrows * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_sec, matAXT.sec, matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXT_NOC <<<cudaBlockNum, cudaBlockSize>>> (  th, thw, d_ax, d_sec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_sec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	char buffer[128];
	strcpy( buffer, "gk_AXT_NOC_H" );
	char TH[3]; sprintf( TH, "%d", th );
	strcat( buffer, TH );
	strcat( buffer, "_HW" );
	char THW[3]; sprintf( THW, "%d", thw );
	strcat( buffer, THW );
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXT_COM_H1( const UIN log, const FPT * ax, const UIN * hdr, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID = tidGRID >> 5;
	const UIN tidWARP = tidGRID & 31;

	printf( "tidGRID:%d, widGRID:%d, tidWARP:%d\n", tidGRID, widGRID, tidWARP );


/*

	const UIN   bidGRID = blockIdx.x;
	const UIN        ts = blockDim.x;
	const UIN  tidBLOCK = threadIdx.x;
	const UIN   tidWARP = tidBLOCK & 31;
	const UIN  widBLOCK = tidBLOCK >> 5;
	const UIN   offsetB = bidGRID * 2 * ts;
	const UIN   offsetW = widBLOCK * 64;
	      UIN      lane = offsetB + offsetW + tidWARP;
	      UIN        ro = hdr[bidGRID * blockSize + tidBLOCK], row, off;
	      FPT val1, val2, val, vor;
	       __shared__ FPT sb1[32];
	extern __shared__ FPT sb2[];

	// use warp 0 to set sb1[] array's elements to zero
	if (widBLOCK == 0) sb1[tidWARP] = 0;
	__syncthreads();

	// use block's threads to set sb2[] array's elements to zero
	sb2[tidBLOCK] = 0;
	__syncthreads();

	// read values from global memory array bax[] and perform multiplication on registers
	val1 = ax[lane] * ax[lane+32];
	vor = val;
	__syncthreads();

	// perform warp-level reduction
	val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
	val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
	val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
	val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
	val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
	__syncthreads();

	// store warp-level results on shared memory block sb[]
	if (tidWARP == 31) sb1[widBLOCK] = val;
	__syncthreads();

	// use warp 0 to perform the reduction of the partial results stored on sb[]
	if (widBLOCK == 0)
	{
		val1 = sb1[tidWARP];
		val2 = __shfl_up_sync( FULL_MASK, val1,  1 ); if (tidWARP >=  1) val1 = val1 + val2;
		val2 = __shfl_up_sync( FULL_MASK, val1,  2 ); if (tidWARP >=  2) val1 = val1 + val2;
		val2 = __shfl_up_sync( FULL_MASK, val1,  4 ); if (tidWARP >=  4) val1 = val1 + val2;
		val2 = __shfl_up_sync( FULL_MASK, val1,  8 ); if (tidWARP >=  8) val1 = val1 + val2;
		val2 = __shfl_up_sync( FULL_MASK, val1, 16 ); if (tidWARP >= 16) val1 = val1 + val2;
		sb1[tidWARP] = val1;
	}
	__syncthreads();

	// update val with partial reductions
	if (widBLOCK > 0) val = val + sb1[widBLOCK-1];
	__syncthreads();

	// write in sb2[] complete reduction values
	sb2[tidBLOCK] = val;
	__syncthreads();

	// perform atomic addition to acumulate value on res[]
	if ( ro )
	{
		off = ro & (blockSize-1);
		val = sb2[tidBLOCK + off] - val + vor;
		row = ro>>log;
		#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
			customAtomicAdd( &y[row], val );
		#else
			      atomicAdd( &y[row], val );
		#endif
	}
*/
	return;
}



static __host__ str_res test_gk_AXT_COM_H1( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref )
{
	// get parameters
	const UIN          log = matAXT.log;
	const UIN          thw = matAXT.tileHW;
	const UIN           tn = matAXT.tileN;
	const UIN        nrows = matAXT.nrows;
	const UIN          nnz = matAXT.nnz;
	const UIN cudaBlockNum = ( (tn * 32) + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN      wrpsGen = cudaBlockNum * wrpsPerBlock;
	const UIN          dif = wrpsGen - tn;
	const UIN     devLenAX = matAXT.lenAX  + dif * 32;
	const UIN    devLenSEC = matAXT.lenSEC + dif;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_sec; HANDLE_CUDA_ERROR( cudaMalloc( &d_sec, devLenSEC * sizeof(UIN) ) ); TEST_POINTER( d_sec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,     nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_sec, 0, devLenSEC * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0,     nrows * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_sec, matAXT.sec, matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXT_COM_H1 <<<cudaBlockNum, cudaBlockSize, (cudaBlockSize*sizeof(FPT))>>> ( log, d_ax, d_sec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_sec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	char buffer[128];
	strcpy( buffer, "gk_AXT_COM_H1_HW" );
	char THW[3]; sprintf( THW, "%d", thw );
	strcat( buffer, THW );
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}






/*
static __host__ void if_AXT_NOC_TH1( const UIN ompThrds, const UIN thw, const UIN tn, const FPT * ax, const UIN * brp, FPT * res )
{
	const UIN ts = 2 * thw;
	      UIN tileID, pos;
	      FPT red;
	__m512d val, vec, pro;
	#pragma omp parallel for default(shared) private(tileID,pos,val,vec,pro,red) num_threads(ompThrds) schedule(static) if(_OMP_)
	for ( tileID = 0; tileID < tn; tileID++ )
	{
		pos = tileID * ts;
		val = _mm512_load_pd( &ax[pos]       );
		vec = _mm512_load_pd( &ax[pos + thw] );
		pro = _mm512_mul_pd( val, vec );
		red = _mm512_reduce_add_pd( pro );
		#pragma omp atomic
		res[tileID] = red;
	}
	return;
}



static __host__ str_res test_if_AXT_NOC_TH1( const UIN ompThrds, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN   thw = matAXT.tileHW;
	const UIN    tn = matAXT.tileN;
	const UIN nrows = matAXT.nrows;
	const UIN   nnz = matAXT.nnz;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	// timed iterations
	struct timeval t1, t2;
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		if_AXT_NOC_TH1( ompThrds, thw, tn, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "if_AXT_NOC_TH1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}

*/





/*
static void x_ax( const unsigned int wi, const st_matrixCSR mat, const unsigned int * brp, const unsigned int * brpm, const unsigned int * mapx, const TYPE * vec, TYPE * ax )
{

	const unsigned int hbrs = 8;
	unsigned int ind1 = 0, ind2 = 0, ind3 = 0;
	TYPE red = (TYPE) 0, sum = (TYPE) 0;
	__m512d vvec;
	__m512i vcol;

	#pragma omp parallel for default( shared ) private( ind1, ind2, ind3, vcol, vvec ) num_threads( wi ) schedule( runtime ) if( _OPENMP )
	for ( ind1 = 0; ind1 < mat.nrows; ind1++ )
	{
		for ( ind2 = brpm[ind1], ind3 = brp[ind1] + 8; ind2 < brpm[ind1 + 1]; ind2 = ind2 + hbrs, ind3 = ind3 + 16 )
		{
			if ( ind2 % 16 == 0 )
				vcol = _mm512_load_epi32( &mapx[ind2] );
			else
			{
				vcol = _mm512_load_epi32( &mapx[ind2 & 0xFFFFFFF0] );
				vcol = _mm512_permute4f128_epi32( vcol, _MM_PERM_BADC );
			}
			vvec = _mm512_i32logather_pd( vcol, vec, 8 );
			_mm512_store_pd( &ax[ind3], vvec );
		}
	}

	return;
}
*/


/*
static void int_axc( const unsigned int wi, const unsigned int nrows, const TYPE * ax, const unsigned int * brp, TYPE * res )
{
	#ifdef __MIC__

		const unsigned int hbrs = 8;
		unsigned int ind1 = 0, ind2 = 0, brpb = 0, brpe = 0;
		TYPE red = (TYPE) 0, sum = (TYPE) 0;
		__m512d val, rhs, pro;

		#pragma omp parallel for default( shared ) private( ind1, ind2, brpb, brpe, val, rhs, pro, red, sum ) num_threads( wi ) schedule( runtime ) if( _OPENMP )
		for ( ind1 = 0; ind1 < nrows; ind1++ )
		{
			brpb = brp[ind1];
			brpe = brp[ind1 + 1];
			sum = (TYPE) 0;
			for ( ind2 = brpb; ind2 < brpe; ind2 = ind2 + ( 2 * hbrs ) )
			{
				val = _mm512_load_pd( &ax[ind2]      );
				rhs = _mm512_load_pd( &ax[ind2 + hbrs] );
				pro = _mm512_mul_pd( val, rhs );
				red = _mm512_reduce_add_pd( pro );
				sum = sum + red;
			}
			res[ind1] = sum;
		}

	#endif

	return;
}
*/


































#endif



