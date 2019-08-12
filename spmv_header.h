// ┌────────────────────────────────┐
// │program: spmv_header.h          │
// │author: Edoardo Coronado        │
// │date: 05-06-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#ifndef __SPMV_HEADER__
#define __SPMV_HEADER__



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
	#define HBRICK_SIZE 16
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
			#pragma omp master
				sia.ompMaxThreads = omp_get_max_threads();
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



typedef struct { char name[12]; double mfp; double beta; double ct; } str_formatData;



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



static __host__ void cpuSpmvCSR( const str_matCSR matCSR, const FPT * vec, FPT * res )
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



static __host__ str_res testCpuSpmvCSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) malloc( matCSR.nrows * sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		cpuSpmvCSR( matCSR, vec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "cpuSpmvCSR" );
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
		FPT aux = (FPT) 0;
		for ( i = row[rowID]; i < row[rowID + 1]; i++ )
			aux = aux + val[i] * x[col[i]];
		y[rowID] = aux;
	}
	return;
}



static __host__ str_res testGpuSpmvCSR( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// allocate memory on GPU
	FPT * d_val; HANDLE_CUDA_ERROR( cudaMalloc( &d_val,   matCSR.nnz         * sizeof(FPT) ) ); TEST_POINTER( d_val );
	UIN * d_col; HANDLE_CUDA_ERROR( cudaMalloc( &d_col,   matCSR.nnz         * sizeof(UIN) ) ); TEST_POINTER( d_col );
	UIN * d_row; HANDLE_CUDA_ERROR( cudaMalloc( &d_row, ( matCSR.nrows + 1 ) * sizeof(UIN) ) ); TEST_POINTER( d_row );
	FPT * d_vec; HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,   matCSR.nrows       * sizeof(FPT) ) ); TEST_POINTER( d_vec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matCSR.nrows       * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val, matCSR.val,   matCSR.nnz        * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col, matCSR.col,   matCSR.nnz        * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_row, matCSR.row, ( matCSR.nrows + 1 )* sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec, vec,          matCSR.nrows      * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	const UIN cudaBlockNum = ( matCSR.nrows + cudaBlockSize - 1 ) / cudaBlockSize;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_CSR <<<cudaBlockNum, cudaBlockSize>>> ( matCSR.nrows, d_val, d_col, d_row, d_vec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matCSR.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matCSR.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
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



static __host__ str_res testGpuCusparseSpmvCSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// variables for cuSPARSE
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
	FPT * d_val; HANDLE_CUDA_ERROR( cudaMalloc( &d_val,   matCSR.nnz         * sizeof(FPT) ) ); TEST_POINTER( d_val );
	int * d_col; HANDLE_CUDA_ERROR( cudaMalloc( &d_col,   matCSR.nnz         * sizeof(int) ) ); TEST_POINTER( d_col );
	int * d_row; HANDLE_CUDA_ERROR( cudaMalloc( &d_row, ( matCSR.nrows + 1 ) * sizeof(int) ) ); TEST_POINTER( d_row );
	FPT * d_vec; HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,   matCSR.nrows       * sizeof(FPT) ) ); TEST_POINTER( d_vec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matCSR.nrows       * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val, matCSR.val,   matCSR.nnz        * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col, matCSR.col,   matCSR.nnz        * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_row, matCSR.row, ( matCSR.nrows + 1 )* sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec, vec,          matCSR.nrows      * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// get space buffer for cusparseCsrmvEx
	HANDLE_CUSPARSE_ERROR( cusparseCsrmvEx_bufferSize( cusparseH, cusparseAM, cusparseO, matCSR.nrows, matCSR.nrows, matCSR.nnz, &one, cudaDT, cusparseMD, \
                                                        d_val, cudaDT, d_row, d_col, d_vec, cudaDT, &zero, cudaDT, d_res, cudaDT, cudaDT, &cudaSpaceBufferSize ) );
	void * cudaSpaceBuffer; HANDLE_CUDA_ERROR( cudaMalloc( &cudaSpaceBuffer, cudaSpaceBufferSize ) );
	// create events for time measuring
	cudaEvent_t cet1;  HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
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
	FPT * res = (FPT *) malloc( matCSR.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matCSR.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col ) );
	HANDLE_CUDA_ERROR( cudaFree( d_row ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gpuCusparseSpmvCSR" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( (double) matCSR.nnz * 2.0 ) / sr.et;
	getErrors( matCSR.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN lenAX; UIN lenBRP; FPT * ax; UIN * brp; } str_matAXCv0;



static __host__ UIN getBrickPointerAXCv0( const str_matCSR matCSR, str_matAXCv0 * matAXC )
{
	UIN rowID = 0, brickNum = 0;
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		brickNum                 = ( matCSR.rl[rowID] + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
		matAXC->brp[rowID + 1]   = matAXC->brp[rowID]  + ( 2 * brickNum * HBRICK_SIZE );
	}
	return( matAXC->brp[matAXC->nrows] );
}



static __host__ void getAxAXCv0_0( const UIN numThreads, const str_matCSR matCSR, const FPT * vec, str_matAXCv0 * matAXC )
{
	UIN rowID, posAXC, counter, posCSR;
	#pragma omp parallel for default( shared ) private( rowID, posAXC, counter, posCSR ) num_threads( numThreads ) schedule( runtime ) if( _OPENMP )
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		posAXC  = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			matAXC->ax[posAXC]               = matCSR.val[posCSR];
			matAXC->ax[posAXC + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			if ( counter == (HBRICK_SIZE - 1) )
			{
				posAXC  = posAXC + 1 + HBRICK_SIZE;
				counter = 0;
			}
			else
			{
				posAXC++;
				counter++;
			}
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXCv0( const UIN numThreads, const str_matCSR matCSR, const FPT * vec, str_matAXCv0 * matAXC )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXCv0" );
	// get AXCv0 parameters
	matAXC->nrows  = matCSR.nrows;
	matAXC->nnz    = matCSR.nnz;
	matAXC->lenBRP = matCSR.nrows + 1;
	matAXC->brp    = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXCv0
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = getBrickPointerAXCv0( matCSR, matAXC );
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
		getAxAXCv0_0( numThreads, matCSR, vec, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXCv0 memory footprint
	fd.mfp =          (double) ( matAXC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXC->lenBRP * sizeof(UIN) ); // brp ( stores the starting address of a row )
	// AXCv0 occupancy ( beta )
	fd.beta = ( (double) matAXC->nnz / (double) (matAXC->lenAX >> 1) );
	// AXCv0 conversion time
	fd.ct = tc;
	return( fd );
}



#ifndef FULL_MASK
	#define FULL_MASK 0xffffffff
#endif



static __global__ void gk_AXCv0( const UIN nrows, const FPT * ax, const UIN * brp, FPT * y )
{
	const UIN tidGRID  = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID  = tidGRID >> 5;
	const UIN tidWARP  = tidGRID & 31;
	const UIN offset   = brp[widGRID];
	const UIN upperLim = brp[widGRID+1];
	      UIN pos;
	      FPT val;
	      FPT aux = (FPT) 0;
	//if (widGRID < nrows)
	//{
		for ( pos = offset + tidWARP; pos < upperLim; pos = pos + 2 * HBRICK_SIZE )
		{
			val = ax[pos];
			val = val * __shfl_down_sync( FULL_MASK, val, 16 );
			val = val + __shfl_down_sync( FULL_MASK, val,  8 );
			val = val + __shfl_down_sync( FULL_MASK, val,  4 );
			val = val + __shfl_down_sync( FULL_MASK, val,  2 );
			val = val + __shfl_down_sync( FULL_MASK, val,  1 );
			if (tidWARP == 0) aux = aux + val;
		}
		if (tidWARP == 0) y[widGRID] = aux;
	//}
	return;
}



static __host__ str_res testGpuSpmvAXCv0( const UIN cudaBlockSize, const str_matAXCv0 matAXC, const FPT * ref )
{
	//
	const UIN cudaBlockNum = ( matAXC.nrows * 32 + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN      wrpsGen = cudaBlockNum * wrpsPerBlock;
	const UIN          dif = wrpsGen - matAXC.nrows;
	const UIN     devLenAX = matAXC.lenAX  + dif*32;
	const UIN    devLenBRP = matAXC.lenBRP + dif;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_brp; HANDLE_CUDA_ERROR( cudaMalloc( &d_brp, devLenBRP * sizeof(UIN) ) ); TEST_POINTER( d_brp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0,          devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_brp, 0,          devLenBRP * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_brp, matAXC.brp, matAXC.lenBRP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0,          matAXC.nrows  * sizeof(FPT) ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv0 <<<cudaBlockNum, cudaBlockSize>>> (  matAXC.nrows, d_ax, d_brp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_brp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv0" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN lenAX; UIN lenBRP; UIN lenRWP; FPT * ax; UIN * brp; UIN * rwp; } str_matAXCv1;



static __host__ UIN getLenRWPAXCv1( const UIN numThreads, const str_matCSR matCSR )
{
	UIN i, brickNum, lenRWP = 0;
	#pragma omp parallel for default( shared ) private( i, brickNum ) reduction( +:lenRWP ) num_threads( numThreads ) schedule( runtime ) if( _OPENMP)
	for ( i = 0; i < matCSR.nrows; i++ )
	{
		brickNum = ( matCSR.rl[i] + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
		lenRWP   = lenRWP + brickNum;
	}
	return( lenRWP );
}



static __host__ UIN getBrickPointerAXCv1( const str_matCSR matCSR, str_matAXCv1 * matAXC )
{
	UIN rowID = 0,brickNum,  brickCounter = 0, pos = 0;
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		brickNum               = ( matCSR.rl[rowID] + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
		matAXC->brp[rowID + 1] = matAXC->brp[rowID]  + ( 2 * brickNum * HBRICK_SIZE );
		for ( brickCounter = 0; brickCounter < brickNum; brickCounter++ )
		{
			matAXC->rwp[pos] = rowID;
			pos++;
		}
	}
	return( matAXC->brp[matAXC->nrows] );
}



static __host__ void getAxAXCv1_0( const UIN numThreads, const str_matCSR matCSR, const FPT * vec, str_matAXCv1 * matAXC )
{
	UIN rowID, posAXC, counter, posCSR;
	#pragma omp parallel for default( shared ) private( rowID, posAXC, counter, posCSR ) num_threads( numThreads ) schedule( runtime ) if( _OPENMP )
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		posAXC  = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			matAXC->ax[posAXC]               = matCSR.val[posCSR];
			matAXC->ax[posAXC + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			if ( counter == (HBRICK_SIZE - 1) )
			{
				posAXC  = posAXC + 1 + HBRICK_SIZE;
				counter = 0;
			}
			else
			{
				posAXC++;
				counter++;
			}
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXCv1( const UIN numThreads, const str_matCSR matCSR, const FPT * vec, str_matAXCv1 * matAXC )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXCv1" );
	// get AXCv1 parameters
	matAXC->nrows  = matCSR.nrows;
	matAXC->nnz    = matCSR.nnz;
	matAXC->lenBRP = matCSR.nrows + 1;
	matAXC->brp    = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXCv1
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenRWP = getLenRWPAXCv1( numThreads, matCSR );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	matAXC->rwp = (UIN *) calloc( matAXC->lenRWP, sizeof(UIN) ); TEST_POINTER( matAXC->rwp  );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = getBrickPointerAXCv1( matCSR, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	matAXC->ax = (FPT *) calloc( matAXC->lenAX, sizeof(FPT) ); TEST_POINTER( matAXC->ax );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxAXCv1_0( numThreads, matCSR, vec, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXC memory footprint
	fd.mfp =          (double) ( matAXC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXC->lenRWP * sizeof(UIN) ); // rwp ( has 1 element per brick tjat indicates the row that brick belongs to )
	// AXC occupancy ( beta )
	fd.beta = ( (double) matAXC->nnz / (double) (matAXC->lenAX >> 1) );
	// AXC conversion time
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



static __global__ void gk_AXCv1_0( const UIN ul, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	      UIN pos       = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = pos & 31;
	      UIN widGRID;
	      UIN rowID;
	      FPT val;
	for ( ; pos < ul; pos = pos + blockNum * blockSize )
	{
		val     = ax[pos];
		val     = val * __shfl_down_sync( FULL_MASK, val, 16 );
		val     = val + __shfl_down_sync( FULL_MASK, val,  8 );
		val     = val + __shfl_down_sync( FULL_MASK, val,  4 );
		val     = val + __shfl_down_sync( FULL_MASK, val,  2 );
		val     = val + __shfl_down_sync( FULL_MASK, val,  1 );
		widGRID = pos >> 5;
		rowID   = rwp[widGRID];
		#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
			if (tidWARP == 0) customAtomicAdd( &y[rowID], val );
		#else
			if (tidWARP == 0)       atomicAdd( &y[rowID], val );
		#endif
	}
	return;
}



static __host__ str_res testGpuSpmvAXCv1_0( const UIN cudaBlockSize, const str_matAXCv1 matAXC, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN cudaBlockNum = ( matAXC.lenAX + cudaBlockSize - 1 ) / cudaBlockSize;
	const UIN     upperLim = ( ( matAXC.lenAX + 31 ) / 32 ) * 32;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  matAXC.lenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, matAXC.lenRWP * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXC.rwp, matAXC.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv1_0 <<<cudaBlockNum, cudaBlockSize>>> (  upperLim, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv1_0" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXCv1_1( const UIN ul, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride    = 2 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos       = widGRID * 64 + tidWARP;
	      UIN rowID1, rowID2, rowID;
	      FPT val1, val2, val;
	for ( ; pos < ul; pos = pos + stride )
	{
		rowID1  = rwp[pos>>5];
		val1    = ax[pos];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+32)>>5];
		val2    = ax[pos+32];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXCv1_1( const UIN cudaBlockSize, const str_matAXCv1 matAXC, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXC.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 1 ) / 2;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 64;
	const UIN    devLenRWP =   warpNum * 2;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenRWP     * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0,          devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0,          devLenRWP     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXC.rwp, matAXC.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv1_1 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv1_1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXCv1_2( const UIN ul, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride    = 4 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos       = widGRID * 128 + tidWARP;
	      UIN rowID1, rowID2, rowID;
	      FPT val1, val2, val;
	for ( ; pos < ul; pos = pos + stride )
	{
		rowID1  = rwp[pos>>5];
		val1    = ax[pos];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+32)>>5];
		val2    = ax[pos+32];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+64)>>5];
		val1    = ax[pos+64];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+96)>>5];
		val2    = ax[pos+96];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}

	}
	return;
}



static __host__ str_res testGpuSpmvAXCv1_2( const UIN cudaBlockSize, const str_matAXCv1 matAXC, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXC.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 3 ) / 4;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 128;
	const UIN    devLenRWP =   warpNum * 4;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenRWP     * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0,          devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0,          devLenRWP     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXC.rwp, matAXC.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv1_2 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv1_2" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXCv1_3( const UIN ul, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride    = 6 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos       = widGRID * 192 + tidWARP;
	      UIN rowID1, rowID2, rowID;
	      FPT val1, val2, val;
	for ( ; pos < ul; pos = pos + stride )
	{
		rowID1  = rwp[pos>>5];
		val1    = ax[pos];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+32)>>5];
		val2    = ax[pos+32];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+64)>>5];
		val1    = ax[pos+64];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+96)>>5];
		val2    = ax[pos+96];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+128)>>5];
		val1    = ax[pos+128];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+160)>>5];
		val2    = ax[pos+160];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXCv1_3( const UIN cudaBlockSize, const str_matAXCv1 matAXC, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXC.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 5 ) / 6;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 192;
	const UIN    devLenRWP =   warpNum * 6;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenRWP     * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0,          devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0,          devLenRWP     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXC.rwp, matAXC.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv1_3 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv1_3" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXCv1_4( const UIN ul, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride    = 8 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos       = widGRID * 256 + tidWARP;
	      UIN rowID1, rowID2, rowID;
	      FPT val1, val2, val;
	for ( ; pos < ul; pos = pos + stride )
	{
		rowID1  = rwp[pos>>5];
		val1    = ax[pos];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+32)>>5];
		val2    = ax[pos+32];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+64)>>5];
		val1    = ax[pos+64];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+96)>>5];
		val2    = ax[pos+96];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+128)>>5];
		val1    = ax[pos+128];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+160)>>5];
		val2    = ax[pos+160];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		rowID1  = rwp[(pos+192)>>5];
		val1    = ax[pos+192];
		val1    = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		rowID2  = rwp[(pos+224)>>5];
		val2    = ax[pos+224];
		val2    = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2 = val;
		val1 = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1 = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		val1 = __shfl_down_sync( FULL_MASK, val, 15 );
		if ( (tidWARP == 0) || (tidWARP == HBRICK_SIZE) )
		{
			if (tidWARP == 0) rowID = rowID1;
			else              rowID = rowID2;
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXCv1_4( const UIN cudaBlockSize, const str_matAXCv1 matAXC, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXC.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 7 ) / 8;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 256;
	const UIN    devLenRWP =   warpNum * 8;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenRWP     * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXC.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0,          devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0,          devLenRWP     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXC.ax,  matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXC.rwp, matAXC.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXCv1_4 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXCv1_4" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	getErrors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN brickNum; UIN lenAX; UIN lenHDR; FPT * ax; UIN * hdr0; } str_matAXBv0;



static __host__ void getAxHdrAXBv0( const str_matCSR matCSR, const FPT * vec, str_matAXBv0 * matAXB )
{
	UIN rowID, rowLen, flag, offset, posCSR, posAX = 0, posHDR = 0, posHDRbrick, posLeft;
	for ( rowID = 0; rowID < matCSR.nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		flag   = 1;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
		{
			if ( (flag==1) || (((posHDR>>5)<<5)==posHDR) )
			{
				posHDRbrick = posHDR & 31;
				posLeft     = 2 * HBRICK_SIZE - posHDRbrick;
				if ( rowLen < posLeft )
				{
					offset = rowLen - 1;
				}
				else
				{
					offset = posLeft - 1;
					rowLen = rowLen - posLeft;
				}
				matAXB->hdr0[posHDR] = (rowID<<5) | offset;
				flag = 0;
			}
			matAXB->ax[posAX]               = matCSR.val[posCSR];
			matAXB->ax[posAX + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			posAX++;
			posHDR++;
			if ( ((posAX>>4)<<4) == posAX ) posAX = posAX + HBRICK_SIZE;
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXBv0( const str_matCSR matCSR, const FPT * vec, str_matAXBv0 * matAXB )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXBv0" );
	// get AXBv0 parameters
	matAXB->nrows    =   matCSR.nrows;
	matAXB->nnz      =   matCSR.nnz;
	matAXB->brickNum = ( matAXB->nnz + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
	matAXB->lenAX    =   matAXB->brickNum * 2 * HBRICK_SIZE;
	matAXB->lenHDR   =   matAXB->brickNum     * HBRICK_SIZE;
	matAXB->ax       = (FPT *) calloc( matAXB->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXB->ax   );
	matAXB->hdr0     = (UIN *) calloc( matAXB->lenHDR, sizeof(UIN) ); TEST_POINTER( matAXB->hdr0 );
	// get matAXB
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxHdrAXBv0( matCSR, vec, matAXB );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXBv0 memory footprint
	fd.mfp =          (double) ( matAXB->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXB->lenHDR * sizeof(UIN) ); // hdr0
	// AXBv0 occupancy ( beta )
	fd.beta = ( (double) matAXB->nnz / (double) (matAXB->lenAX >> 1) );
	// AXBv0 conversion time
	fd.ct = tc;
	return( fd );
}



static __global__ void gk_AXBv0_0( const UIN ul, const FPT * ax, const UIN * hdr0, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride1   = 2 * blockNum * blockSize;
	const UIN stride2   =     blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos1      = widGRID * 64 + tidWARP;
	      UIN pos2      = widGRID * 32 + tidWARP;
	      UIN ro, rowID, offset;
	      FPT val1, val2, val;
	for ( ; pos1 < ul; pos1 = pos1 + stride1, pos2 = pos2 + stride2 )
	{
		val1   = ax[pos1];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+32];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXBv0_0( const UIN cudaBlockSize, const str_matAXBv0 matAXB, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXB.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 1) / 2;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 64;
	const UIN    devLenHDR =   warpNum * 32;
	// allocate memory on GPU
	FPT * d_ax;   HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,   devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax   );
	UIN * d_hdr0; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr0, devLenHDR     * sizeof(UIN) ) ); TEST_POINTER( d_hdr0 );
	FPT * d_res;  HANDLE_CUDA_ERROR( cudaMalloc( &d_res,  matAXB.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res  );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,   0,           devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr0, 0,           devLenHDR     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,   matAXB.ax,   matAXB.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr0, matAXB.hdr0, matAXB.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXB.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXBv0_0 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_hdr0, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXB.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXB.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr0 ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res  ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXBv0_0" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXB.nnz ) ) / sr.et;
	getErrors( matAXB.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXBv0_1( const UIN ul, const FPT * ax, const UIN * hdr0, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride1   = 4 * blockNum * blockSize;
	const UIN stride2   = 2 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos1      = widGRID * 128 + tidWARP;
	      UIN pos2      = widGRID *  64 + tidWARP;
	      UIN ro, rowID, offset;
	      FPT val1, val2, val;
	for ( ; pos1 < ul; pos1 = pos1 + stride1, pos2 = pos2 + stride2 )
	{
		val1   = ax[pos1];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+32];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+64];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+96];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+32];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXBv0_1( const UIN cudaBlockSize, const str_matAXBv0 matAXB, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXB.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 3) / 4;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 128;
	const UIN    devLenHDR =   warpNum * 64;
	// allocate memory on GPU
	FPT * d_ax;   HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,   devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax   );
	UIN * d_hdr0; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr0, devLenHDR     * sizeof(UIN) ) ); TEST_POINTER( d_hdr0 );
	FPT * d_res;  HANDLE_CUDA_ERROR( cudaMalloc( &d_res,  matAXB.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res  );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,   0,           devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr0, 0,           devLenHDR     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,   matAXB.ax,   matAXB.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr0, matAXB.hdr0, matAXB.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXB.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXBv0_1 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_hdr0, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXB.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXB.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr0 ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res  ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXBv0_1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXB.nnz ) ) / sr.et;
	getErrors( matAXB.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXBv0_2( const UIN ul, const FPT * ax, const UIN * hdr0, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride1   = 6 * blockNum * blockSize;
	const UIN stride2   = 3 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos1      = widGRID * 192 + tidWARP;
	      UIN pos2      = widGRID *  96 + tidWARP;
	      UIN ro, rowID, offset;
	      FPT val1, val2, val;
	for ( ; pos1 < ul; pos1 = pos1 + stride1, pos2 = pos2 + stride2 )
	{
		val1   = ax[pos1];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+32];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+64];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+96];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+32];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+128];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+160];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+64];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXBv0_2( const UIN cudaBlockSize, const str_matAXBv0 matAXB, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXB.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 5) / 6;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 192;
	const UIN    devLenHDR =   warpNum * 96;
	// allocate memory on GPU
	FPT * d_ax;   HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,   devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax   );
	UIN * d_hdr0; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr0, devLenHDR     * sizeof(UIN) ) ); TEST_POINTER( d_hdr0 );
	FPT * d_res;  HANDLE_CUDA_ERROR( cudaMalloc( &d_res,  matAXB.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res  );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,   0,           devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr0, 0,           devLenHDR     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,   matAXB.ax,   matAXB.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr0, matAXB.hdr0, matAXB.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXB.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXBv0_2 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_hdr0, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXB.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXB.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr0 ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res  ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXBv0_2" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXB.nnz ) ) / sr.et;
	getErrors( matAXB.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXBv0_3( const UIN ul, const FPT * ax, const UIN * hdr0, FPT * y )
{
	const UIN blockNum  = gridDim.x;
	const UIN blockSize = blockDim.x;
	const UIN stride1   = 8 * blockNum * blockSize;
	const UIN stride2   = 4 * blockNum * blockSize;
	const UIN tidGRID   = blockIdx.x * blockSize + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	      UIN pos1      = widGRID * 256 + tidWARP;
	      UIN pos2      = widGRID * 128 + tidWARP;
	      UIN ro, rowID, offset;
	      FPT val1, val2, val;
	for ( ; pos1 < ul; pos1 = pos1 + stride1, pos2 = pos2 + stride2 )
	{
		val1   = ax[pos1];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+32];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+64];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+96];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+32];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+128];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+160];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+64];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
		val1   = ax[pos1+192];
		val1   = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
		val2   = ax[pos1+224];
		val2   = val2 *   __shfl_up_sync( FULL_MASK, val2, HBRICK_SIZE );
		if (tidWARP < HBRICK_SIZE) val = val1;
		else                       val = val2;
		val2   = val;
		val1   = __shfl_up_sync( FULL_MASK, val,  1 ); if (tidWARP >=  1) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  2 ); if (tidWARP >=  2) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  4 ); if (tidWARP >=  4) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val,  8 ); if (tidWARP >=  8) val = val + val1;
		val1   = __shfl_up_sync( FULL_MASK, val, 16 ); if (tidWARP >= 16) val = val + val1;
		ro     = hdr0[pos2+96];
		rowID  = ro >> 5;
		offset = ro & 31;
		val1   = __shfl_down_sync( FULL_MASK, val, offset );
		if (ro != 0)
		{
			val  = val1 - val + val2;
			#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
				customAtomicAdd( &y[rowID], val );
			#else
				      atomicAdd( &y[rowID], val );
			#endif
		}
	}
	return;
}



static __host__ str_res testGpuSpmvAXBv0_3( const UIN cudaBlockSize, const str_matAXBv0 matAXB, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickNum = ( matAXB.lenAX + 31 ) / 32;
	const UIN      warpNum = ( brickNum + 7) / 8;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpNum + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 256;
	const UIN    devLenHDR =   warpNum * 128;
	// allocate memory on GPU
	FPT * d_ax;   HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,   devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax   );
	UIN * d_hdr0; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr0, devLenHDR     * sizeof(UIN) ) ); TEST_POINTER( d_hdr0 );
	FPT * d_res;  HANDLE_CUDA_ERROR( cudaMalloc( &d_res,  matAXB.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res  );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,   0,           devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr0, 0,           devLenHDR     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,   matAXB.ax,   matAXB.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr0, matAXB.hdr0, matAXB.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXB.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXBv0_3 <<<cudaBlockNum, cudaBlockSize>>> (  devLenAX, d_ax, d_hdr0, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXB.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXB.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr0 ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res  ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXBv0_3" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXB.nnz ) ) / sr.et;
	getErrors( matAXB.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN brickNum; UIN lenAX; UIN lenHDR; UIN blockLog; FPT * ax; UIN * hdr1; } str_matAXBv1;



static __host__ void getAxHdrAXBv1( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, str_matAXBv1 * matAXB )
{
	UIN rowID, rowLen, flag, offset, posCSR, posAX = 0, posHDR = 0, posHDRbrick, posLeft;
	for ( rowID = 0; rowID < matCSR.nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		flag   = 1;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
		{
			if ( (flag==1) || (((posHDR>>matAXB->blockLog)<<matAXB->blockLog)==posHDR) )
			{
				posHDRbrick = posHDR & (cudaBlockSize - 1);
				posLeft     = cudaBlockSize - posHDRbrick;
				if ( rowLen < posLeft )
				{
					offset = rowLen - 1;
				}
				else
				{
					offset = posLeft - 1;
					rowLen = rowLen - posLeft;
				}
				matAXB->hdr1[posHDR] = (rowID<<matAXB->blockLog) | offset;
				flag = 0;
			}
			matAXB->ax[posAX]               = matCSR.val[posCSR];
			matAXB->ax[posAX + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			posAX++;
			posHDR++;
			if ( ((posAX>>4)<<4) == posAX ) posAX = posAX + HBRICK_SIZE;
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXBv1( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, str_matAXBv1 * matAXB )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXBv1" );
	// get AXBv1 parameters
	matAXB->nrows    =   matCSR.nrows;
	matAXB->nnz      =   matCSR.nnz;
	matAXB->brickNum = ( matAXB->nnz + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
	matAXB->lenAX    =   matAXB->brickNum * 2 * HBRICK_SIZE;
	matAXB->lenHDR   = ( ( ( matAXB->lenAX / 2 ) + ( 2 * HBRICK_SIZE ) - 1 ) / ( 2 * HBRICK_SIZE ) ) * ( 2 * HBRICK_SIZE );
	//matAXB->lenHDR   =   matAXB->brickNum     * HBRICK_SIZE;
	matAXB->ax       = (FPT *) calloc( matAXB->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXB->ax   );
	matAXB->hdr1     = (UIN *) calloc( matAXB->lenHDR, sizeof(UIN) ); TEST_POINTER( matAXB->hdr1 );
	// cuda block log factor
	UIN i;
	matAXB->blockLog = 0;
	for ( i = 5; i < 11; i++ )
	{
		if ((cudaBlockSize>>i)==1)
		{
			matAXB->blockLog = i;
			break;
		}
	}
	// get matAXBv1
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxHdrAXBv1( cudaBlockSize, matCSR, vec, matAXB );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXBv1 memory footprint
	fd.mfp =          (double) ( matAXB->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXB->lenHDR * sizeof(UIN) ); // hdr1
	// AXBv1 occupancy ( beta )
	fd.beta = ( (double) matAXB->nnz / (double) (matAXB->lenAX >> 1) );
	// AXBv1 conversion time
	fd.ct = tc;
	return( fd );
}



static __global__ void gk_AXBv1_0( const UIN blockLog, const FPT * ax, const UIN * hdr1, FPT * res )
{
	const UIN   bidGRID = blockIdx.x;
	const UIN blockSize = blockDim.x;
	const UIN  tidBLOCK = threadIdx.x;
	const UIN   tidWARP = tidBLOCK &  31;
	const UIN  widBLOCK = tidBLOCK >> 5;
	const UIN   offsetB = bidGRID * 2 * blockSize;
	const UIN   offsetW = widBLOCK * 64;
	      UIN      lane = offsetB + offsetW + tidWARP;
	      UIN        ro = hdr1[bidGRID * blockSize + tidBLOCK], row, off;
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
	val1 = ax[lane];
	val1 = val1 * __shfl_down_sync( FULL_MASK, val1, HBRICK_SIZE );
	val2 = ax[lane+32];
	val2 = val2 * __shfl_up_sync(   FULL_MASK, val2, HBRICK_SIZE );
	if (tidWARP < HBRICK_SIZE) val = val1;
	else                       val = val2;
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
		row = ro>>blockLog;
		#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
			customAtomicAdd( &res[row], val );
		#else
			      atomicAdd( &res[row], val );
		#endif
	}

	return;
}



static __host__ str_res testGpuSpmvAXBv1_0( const UIN cudaBlockSize, const str_matAXBv1 matAXB, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN     brickReq = ( matAXB.lenAX + 31 ) / 32;
	const UIN      warpReq = ( brickReq + 1 ) / 2;
	const UIN wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN cudaBlockNum = ( warpReq + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN      warpNum = cudaBlockNum * wrpsPerBlock;
	const UIN     devLenAX =   warpNum * 64;
	const UIN    devLenHDR =   warpNum * 32;
	// allocate memory on GPU
	FPT * d_ax;   HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,   devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax   );
	UIN * d_hdr1; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr1, devLenHDR     * sizeof(UIN) ) ); TEST_POINTER( d_hdr1 );
	FPT * d_res;  HANDLE_CUDA_ERROR( cudaMalloc( &d_res,  matAXB.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res  );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,   0,           devLenAX      * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr1, 0,           devLenHDR     * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,   matAXB.ax,   matAXB.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr1, matAXB.hdr1, matAXB.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXB.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXBv1_0 <<<cudaBlockNum, cudaBlockSize, (cudaBlockSize * sizeof(FPT))>>> (  matAXB.blockLog, d_ax, d_hdr1, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXB.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXB.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr1 ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res  ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk_AXBv1_0" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXB.nnz ) ) / sr.et;
	getErrors( matAXB.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN ind; UIN val; } str_pair;



typedef struct { UIN nrows; UIN nnz; UIN chunkNum; UIN lenVC; UIN * permi; UIN * nmc; UIN * chp; FPT * val; UIN * col; } str_matK1v0;



static __host__ int orderFunction( const void * ele1, const void * ele2 )
{
	return (  ( (str_pair *) ele2 )->val - ( (str_pair *) ele1 )->val  );
}



static __host__ void getPermiK1( const str_matCSR matCSR, str_matK1v0 * matK1 )
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



static __host__ UIN getNmcChpK1( const str_matCSR matCSR, str_matK1v0 * matK1 )
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



static __host__ void getValColK1( const str_matCSR matCSR, str_matK1v0 * matK1 )
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



static __host__ str_formatData getFormatDataK1v0( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, str_matK1v0 * matK1 )
{
	// format's name
	str_formatData fd;
	strcpy( fd.name, "K1v0" );
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
		getPermiK1( matCSR, matK1 );
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
		matK1->lenVC = getNmcChpK1( matCSR, matK1 );
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
		getValColK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
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



static __global__ void gk_K1v0_0( const int NROWS, const FPT * val, const UIN * col, const UIN * nmc, const UIN * chp, const UIN * permi, const FPT * x, FPT * y )
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



static __host__ str_res testGpuSpmvK1v0_0( const UIN cudaBlockSize, const str_matK1v0 matK1, const FPT * vec, const FPT * ref )
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
		gk_K1v0_0 <<<cudaBlockNum, cudaBlockSize>>> (  matK1.nrows, d_val, d_col, d_nmc, d_chp, d_permi, d_vec, d_res );
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
	strcpy( sr.name, "gk_K1v0_0" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matK1.nnz ) ) / sr.et;
	getErrors( matK1.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN tileHW; UIN tileH; UIN tileN; UIN lenAX; UIN lenRWP; UIN lenCLP; FPT * ax; UIN * rwp; UIN * clp; } str_matAXTv0;



static __host__ void getLenAxAXTv0( const UIN numThreads, const str_matCSR matCSR, str_matAXTv0 * matAXT )
{
	const UIN thw = matAXT->tileHW;
	const UIN th  = matAXT->tileH;
	      UIN cn = 0, tc = 0, tn = 0, lax = 0, rowID = 0;
	#pragma omp parallel for default(shared) private(rowID,cn) reduction(+:tc) num_threads(numThreads) if(_OPENMP)
	for ( rowID = 0; rowID < matCSR.nrows; rowID++ )
	{
		cn = ( matCSR.rl[rowID] + th - 1 ) / th;
		tc = tc + cn;
	}
	tn   = ( tc + thw - 1 ) / thw;
	lax  = 2 * tn * th * thw;
	matAXT->tileN  = tn;
	matAXT->lenAX  = lax;
	matAXT->lenRWP = tn * thw;
	return;
}



static __host__ void getAxRwpAXTv0( const UIN numThreads, const UIN log, const str_matCSR matCSR, const FPT * vec, str_matAXTv0 * matAXT )
{
	const UIN thw = matAXT->tileHW;
	const UIN th  = matAXT->tileH;
	      UIN rowID = 0, colID = 0, posAX = 0, eleCounter = 0, posCSR = 0;
	for ( rowID = 0; rowID < matAXT->nrows; rowID++ )
	{
		if (matCSR.rl[rowID] > 0)
		{
			matAXT->rwp[colID] = rowID;
			matAXT->clp[rowID] = (colID>>log) * 2 * thw * th + (colID&(thw-1));
			posAX              = matAXT->clp[rowID];
			eleCounter         = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				if (eleCounter == th)
				{
					colID++;
					matAXT->rwp[colID] = rowID;
					posAX              = (colID>>log) * 2 * thw * th + (colID&(thw-1));
					eleCounter         = 0;
				}
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				posAX                 = posAX + 2 * thw;
				eleCounter++;
			}
			colID++;
		}
	}
	return;
}



static __host__ str_formatData getFormatDataAXTv0( const UIN numThreads, const UIN tileHalfWidth, const UIN tileHeight, const str_matCSR matCSR, const FPT * vec, str_matAXTv0 * matAXT )
{
	// format's name
	str_formatData fd;
	char buffer[128];
	strcpy( buffer, "AXTv0H" );
	char H[3];  sprintf( H,  "%d", tileHeight );
	strcat( buffer, H );
	strcpy( fd.name, buffer );
	// get AXBv0 parameters
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->lenCLP = matCSR.nrows + 1;
	matAXT->tileHW = tileHalfWidth;
	matAXT->tileH  = tileHeight;
	UIN log = 0, i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileHW) >> i) == 1 ) log = i;
	// get matAXB
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getLenAxAXTv0( numThreads, matCSR, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXT->ax  = (FPT *) calloc( matAXT->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXT->ax  );
	matAXT->rwp = (UIN *) calloc( matAXT->lenRWP, sizeof(UIN) ); TEST_POINTER( matAXT->rwp );
	matAXT->clp = (UIN *) calloc( matAXT->lenCLP, sizeof(UIN) ); TEST_POINTER( matAXT->clp );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxRwpAXTv0( numThreads, log, matCSR, vec, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXTv0 memory footprint
	fd.mfp =          (double) ( matAXT->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXT->lenRWP * sizeof(UIN) ); // rwp
	// AXTv0 occupancy ( beta )
	fd.beta = ( (double) matAXT->nnz / (double) (matAXT->lenAX >> 1) );
	// AXTv0 conversion time
	fd.ct = tc;
	return( fd );
}



static __global__ void gk_AXTv0_0( const UIN ts, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID   = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	const UIN stride    = 2 * thw;
	      UIN posAX     = widGRID * ts  + tidWARP;
	      UIN posRWP    = widGRID * thw + tidWARP;
	      UIN rowID     = rwp[posRWP];
	      UIN ul        = ( widGRID + 1 ) * ts;
	      FPT val, red = 0.0;
	for ( ; posAX < ul; posAX = posAX + stride )
	{
		val = ax[posAX];
		red = red + val * __shfl_down_sync( FULL_MASK, val, thw );
	}
	if (tidWARP < thw)
	{
		#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
			customAtomicAdd( &y[rowID], red );
		#else
			      atomicAdd( &y[rowID], red );
		#endif
	}
	return;
}



static __host__ str_res testGpuSpmvAXTv0_0( const UIN cudaBlockSize, const str_matAXTv0 matAXT, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN tileHalfWidth = matAXT.tileHW;
	const UIN      tileSize = matAXT.tileH * 2 * tileHalfWidth;
	const UIN  wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN  cudaBlockNum = ( matAXT.tileN + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN      devLenAX = cudaBlockNum * wrpsPerBlock * tileSize;
	const UIN     devLenRWP = cudaBlockNum * wrpsPerBlock * tileHalfWidth;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,      devLenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp,     devLenRWP * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXT.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0, devLenRWP * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXT.rwp, matAXT.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXT.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXTv0_0 <<<cudaBlockNum, cudaBlockSize>>> ( tileSize, tileHalfWidth, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXT.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXT.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	char buffer[128];
	strcpy( buffer, "gk_AXTv0_0H" );
	char H[3]; sprintf( H, "%d", matAXT.tileH );
	strcat( buffer, H );
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXT.nnz ) ) / sr.et;
	getErrors( matAXT.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



static __global__ void gk_AXTv0_1( const UIN ts, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID   = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP   = tidGRID & 31;
	const UIN widGRID   = tidGRID >> 5;
	const UIN stride    = 2 * thw;
	      UIN posAX     = 2 * widGRID * ts  + tidWARP;
	      UIN ul        = 2 * widGRID * ts  + ts;
	      UIN posRWP    = 2 * widGRID * thw + tidWARP;
	      UIN rowID     = rwp[posRWP];
	      FPT val1, val2, red1 = 0.0, red2 = 0.0;

	for ( ; posAX < ul; posAX = posAX + stride )
	{
		val1 = ax[posAX];
		val1 = val1 * __shfl_down_sync( FULL_MASK, val1, thw );
		red1 = red1 + val1;
		val2 = ax[posAX+ts];
		val2 = val2 *   __shfl_up_sync( FULL_MASK, val2, thw );
		red2 = red2 + val2;
	}

	#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
		if (tidWARP < thw) customAtomicAdd( &y[rowID], red1 );
		else               customAtomicAdd( &y[rowID], red2 );
	#else
		if (tidWARP < thw)       atomicAdd( &y[rowID], red1 );
		else                     atomicAdd( &y[rowID], red2 );
	#endif

	return;
}



static __host__ str_res testGpuSpmvAXTv0_1( const UIN cudaBlockSize, const str_matAXTv0 matAXT, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN tileHalfWidth = matAXT.tileHW;
	const UIN      tileSize = matAXT.tileH * 2 * tileHalfWidth;
	const UIN       wrpsReq = ( matAXT.tileN + 1 ) / 2;
	const UIN  wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN  cudaBlockNum = ( wrpsReq + wrpsPerBlock - 1 ) / wrpsPerBlock;
	const UIN      devLenAX = cudaBlockNum * wrpsPerBlock * 2 * tileSize;
	const UIN     devLenRWP = cudaBlockNum * wrpsPerBlock * 2 * matAXT.tileHW;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenRWP     * sizeof(UIN) ) ); TEST_POINTER( d_rwp );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXT.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0, devLenRWP * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXT.rwp, matAXT.lenRWP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXT.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXTv0_1 <<<cudaBlockNum, cudaBlockSize>>> ( tileSize, tileHalfWidth, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXT.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXT.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	char buffer[128];
	strcpy( buffer, "gk_AXTv0_1H" );
	char H[3]; sprintf( H, "%d", matAXT.tileH );
	strcat( buffer, H );
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXT.nnz ) ) / sr.et;
	getErrors( matAXT.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenHDR; FPT * ax; UIN * hdr; } str_matAXTv1;



static __host__ void getLenAxAXTv1( const str_matCSR matCSR, str_matAXTv1 * matAXT )
{
	const UIN thw = matAXT->tileHW;
	const UIN th  = matAXT->tileH;
	      UIN tc = 0, tn = 0, lhdr = 0;
	tc    = ( matAXT->nnz + th - 1 ) / th;
	tn    = ( tc + thw - 1 ) / thw;
	lhdr  = tn * th * thw;
	matAXT->tileN  = tn;
	matAXT->lenAX  = 2 * lhdr;
	matAXT->lenHDR = lhdr;
	return;
}



static __host__ void getAxRwpAXTv1( const str_matCSR matCSR, const FPT * vec, str_matAXTv1 * matAXT )
{
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN logTH = matAXT->logTH;
	const UIN   ths = thw * th;
	      UIN rowID = 0, rowLen = 0, flag = 0, posCSR = 0, offsetT = 0, offsetR = 0, offsetC = 0, posAX = 0, posHDR = 0, q1 = 0, q2 = 0, len = 0;


	for ( rowID = 0; rowID < matAXT->nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen > 0)
		{
			flag = 1;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				     offsetT = ( ((posCSR + ths) / ths) - 1 );
				     offsetR = posCSR % th;
				     offsetC = ( ( posCSR + th ) / th ) - 1 - (offsetT * thw);
				       posAX = offsetT * (2 * ths) + offsetR * 2 * thw + offsetC;
				      posHDR = offsetT * ths       + offsetR     * thw + offsetC;
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( (flag == 1) || (offsetR == 0) )
				{
					q1 = th - 1 - (posCSR%th);
					q2 = rowLen - 1;
					if ( q1 < q2 )
					{
						len = q1;
						rowLen = rowLen - (th - (posCSR%th));
					}
					else len = q2;
					matAXT->hdr[posHDR] = (rowID << logTH) | len;
					flag = 0;
				}
			}
		}
	}
	return;
}



static __host__ void updateAxAXTv1( const UIN numThreads, const str_matCSR matCSR, const FPT * vec, str_matAXTv1 * matAXT )
{
	const UIN thw = matAXT->tileHW;
	const UIN th  = matAXT->tileH;
	const UIN ths = thw * th;
	      UIN posCSR = 0, offsetT = 0, offsetR = 0, offsetC = 0, posAX = 0;
	for ( posCSR = 0; posCSR < matAXT->nnz; posCSR++ )
	{
		              offsetT = ( ( ( posCSR + ths ) / ths ) - 1 );
		              offsetR = (posCSR % th) * 2 * th;
		              offsetC = ( ( posCSR + th ) / th ) - 1 - (offsetT * th);
		                posAX = offsetT * (2 * ths) + offsetR + offsetC + thw;
		matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
	}
	return;
}



static __host__ str_formatData getFormatDataAXTv1( const UIN numThreads, const UIN tileHalfWidth, const UIN tileHeight, const str_matCSR matCSR, const FPT * vec, str_matAXTv1 * matAXT )
{
	// format's name
	str_formatData fd;
	char buffer[128];
	strcpy( buffer, "AXTv1H" );
	char H[3];  sprintf( H,  "%d", tileHeight );
	strcat( buffer, H );
	strcpy( fd.name, buffer );
	// get AXBv0 parameters
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->tileHW = tileHalfWidth;
	matAXT->tileH  = tileHeight;
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileH) >> i) == 1 ) matAXT->logTH = i;
	// get matAXB
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getLenAxAXTv1( matCSR, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXT->ax  = (FPT *) calloc( matAXT->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXT->ax  );
	matAXT->hdr = (UIN *) calloc( matAXT->lenHDR, sizeof(UIN) ); TEST_POINTER( matAXT->hdr );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxRwpAXTv1( matCSR, vec, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXTv0 memory footprint
	fd.mfp =          (double) ( matAXT->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXT->lenHDR * sizeof(UIN) ); // hdr
	// AXTv0 occupancy ( beta )
	fd.beta = ( (double) matAXT->nnz / (double) (matAXT->lenAX >> 1) );
	// AXTv0 conversion time
	fd.ct = tc;
	return( fd );
}



static __global__ void gk_AXTv1_0( const UIN thw, const UIN th, const UIN log, const UIN ts, const FPT * ax, const UIN * hdr, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP = tidGRID & 31;
	const UIN widGRID = tidGRID >> 5;
	const UIN  stride = 2 * thw;
	      UIN   posAX = widGRID * ts + tidWARP;
	const UIN    jump = widGRID * (thw*th);
	      UIN  posHDR = jump + tidWARP;
	      UIN posHDR2 = __shfl_up_sync( FULL_MASK, posHDR, thw );
	      UIN i1 = 0, i2 = 0, segNum = 0, thdr = 0, rowID = 0, rowLen = 0, upperLim = 0, nextPos = 0;
	      FPT  red = 0;
	if (tidWARP >= thw) posHDR = posHDR2;
	for ( i1 = 0, i2 = posHDR; i1 < th; i1++, i2 = i2 + thw )
		if (hdr[i2] != 0) segNum++;
	for ( i1 = 0; i1 < segNum; i1++ )
	{
		    thdr = hdr[posHDR];
		   rowID = thdr >> log;
		  rowLen = thdr & (th-1);
		upperLim = posAX + rowLen * stride;
		 nextPos = nextPos + 1 + rowLen;
		     red = 0;
		for ( ; posAX <= upperLim; posAX = posAX + stride )
			red = red + ax[posAX] * ax[posAX+thw];
		#if (FP_TYPE == FP_DOUBLE) && (__CUDA_ARCH__ < 600)
			if (tidWARP < thw) customAtomicAdd( &y[rowID], red );
		#else
			if (tidWARP < thw)       atomicAdd( &y[rowID], red );
		#endif
		  posHDR = jump + nextPos * thw + tidWARP;
		 posHDR2 = __shfl_up_sync( FULL_MASK, posHDR, thw );
		if (tidWARP >= thw) posHDR = posHDR2;
	}
	return;
}



static __host__ str_res testGpuSpmvAXTv1_0( const UIN cudaBlockSize, const str_matAXTv1 matAXT, const FPT * ref )
{
	// 1 V100 card ---> 80 SMs ---> 5,120 warps ---> 163,840 threads
	//                   1 SM  --->    64 warps --->   2,048 threads
	//                                  1 warp  --->      32 threads
	const UIN        tileHW = matAXT.tileHW;
	const UIN         tileH = matAXT.tileH;
	const UIN         logTH = matAXT.logTH;
	const UIN         tileS = 2 * tileHW * tileH;
	const UIN  wrpsPerBlock = ( cudaBlockSize + 31 ) / 32;
	const UIN  cudaBlockNum = ( matAXT.tileN + wrpsPerBlock - 1 ) / wrpsPerBlock;
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  matAXT.lenAX  * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
	UIN * d_hdr; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr, matAXT.lenHDR * sizeof(UIN) ) ); TEST_POINTER( d_hdr );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXT.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, matAXT.lenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_hdr, 0, matAXT.lenHDR * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr, matAXT.hdr, matAXT.lenHDR * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXT.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk_AXTv1_0 <<<cudaBlockNum, cudaBlockSize>>> ( tileHW, tileH, logTH, tileS, d_ax, d_hdr, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXT.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXT.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
	HANDLE_CUDA_ERROR( cudaFree( d_hdr ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	char buffer[128];
	strcpy( buffer, "gk_AXTv1_0H" );
	char H[3]; sprintf( H, "%d", matAXT.tileH );
	strcat( buffer, H );
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXT.nnz ) ) / sr.et;
	getErrors( matAXT.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}























typedef struct{ UIN nrows; UIN nnz; char mode[8]; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenSEC; UIN lenCON; FPT * ax; UIN * sec; UIN * con; } str_matAXTC;



static __host__ void getArraysLenAXTC( const str_matCSR matCSR, str_matAXTC * matAXTC )
{
	const UIN  nrows = matAXTC->nrows;
	const UIN    thw = matAXTC->tileHW;
	const UIN     th = matAXTC->tileH;
	const UIN    ths = thw * th;
	const UIN grpLen = (th == 1) ? (thw) : (th) ;
	char mode[8];
	strcpy( mode, matAXTC->mode );
	      UIN rowID = 0, rowStartPos = 0, rowOffT, rowOffR, rowOffC, pos, rowLen, positions, columns, totalColumns = 0, totalTiles;
	for ( ; rowID < nrows; rowID++ )
	{
		            rowOffT = ( (rowStartPos + ths)/ths ) - 1;
		            rowOffR =    rowStartPos % th;
		            rowOffC = ( (rowStartPos + th)/th ) - 1 - (rowOffT * thw);
		                pos = rowOffT * (2 * ths) + rowOffR * (2 * thw) + rowOffC;
		matAXTC->con[rowID] = pos;
		             rowLen = matCSR.rl[rowID];
		          positions = ( strcmp( mode, "NOComp" ) == 0 ) ? ( ( ( rowLen + grpLen - 1 ) / grpLen ) * grpLen ) : ( rowLen ) ;
		            columns = ( positions + th - 1 ) / th;
		       totalColumns = totalColumns + columns;
		        rowStartPos = rowStartPos + positions;
	}
	     totalTiles = ( totalColumns + thw - 1 ) / thw;
	 matAXTC->tileN = totalTiles;
	 matAXTC->lenAX = totalTiles * 2 * ths;
	if      ( (strcmp(mode, "NOComp")==0) && (th==1) ) matAXTC->lenSEC = totalTiles;
	else if ( (strcmp(mode, "NOComp")==0) && (th!=1) ) matAXTC->lenSEC = totalTiles * thw;
	else                                               matAXTC->lenSEC = totalTiles * ths;
	return;
}





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



















static __host__ void getAxSecAXTC( const UIN ompNumThreads, const str_matCSR matCSR, const FPT * vec, str_matAXTC * matAXTC )
{
	const UIN nrows = matAXTC->nrows;
	const UIN   thw = matAXTC->tileHW;
	const UIN    th = matAXTC->tileH;
	const UIN logTH = matAXTC->logTH;
	const UIN   ths = thw * th;
	      UIN rowID = 0, rowLen = 0, flag = 0, posCSR = 0, offsetT = 0, offsetR = 0, offsetC = 0, posAX = 0, posSEC = 0, q1 = 0, q2 = 0, len = 0;



/*
	printf( "ompNumThreads:%d\n", ompNumThreads );
	UIN tid = 1579;
	#pragma omp parallel for default(shared) private(rowID, tid) num_threads(ompNumThreads) if(_OMP_)
	for ( rowID = 0; rowID < 40; rowID++ )
	{
		tid = omp_get_thread_num();
		printf( "tid:%d, rowID:%d\n", tid, rowID );
	}
*/




/*
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen > 0)
		{
			flag = 1;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				     offsetT = ( ((posCSR + ths) / ths) - 1 );
				     offsetR = posCSR % th;
				     offsetC = ( ( posCSR + th ) / th ) - 1 - (offsetT * thw);
				       posAX = offsetT * (2 * ths) + offsetR * 2 * thw + offsetC;
				      posSEC = offsetT * ths       + offsetR     * thw + offsetC;
				matAXTC->ax[posAX]     = matCSR.val[posCSR];
				matAXTC->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( (flag == 1) || (offsetR == 0) )
				{
					q1 = th - 1 - (posCSR%th);
					q2 = rowLen - 1;
					if ( q1 < q2 )
					{
						len = q1;
						rowLen = rowLen - (th - (posCSR%th));
					}
					else len = q2;
					matAXTC->sec[posSEC] = (rowID << logTH) | len;
					flag = 0;
				}
			}
		}
	}
*/
	return;
}



static __host__ str_formatData getFormatDataAXTC( const UIN ompNumThreads, const UIN tileHalfWidth, const UIN tileHeight, const char * mode, const str_matCSR matCSR, const FPT * vec, str_matAXTC * matAXTC )
{
	// set AXTC parameters
	matAXTC->nrows  = matCSR.nrows;
	matAXTC->nnz    = matCSR.nnz;
	matAXTC->tileHW = tileHalfWidth;
	matAXTC->tileH  = tileHeight;
	strcpy( matAXTC->mode, mode );
	matAXTC->lenCON = matCSR.nrows;
	   matAXTC->con = (UIN *) calloc( matAXTC->lenCON, sizeof(UIN) ); TEST_POINTER( matAXTC->con );
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXTC->tileH) >> i) == 1 ) matAXTC->logTH = i;
	// get AXTC arrays' length
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysLenAXTC( matCSR, matAXTC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;

	// get arrays ax[] and sec[]
	matAXTC->ax  = (FPT *) calloc( matAXTC->lenAX,  sizeof(FPT) ); TEST_POINTER( matAXTC->ax  );
	matAXTC->sec = (UIN *) calloc( matAXTC->lenSEC, sizeof(UIN) ); TEST_POINTER( matAXTC->sec );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getAxSecAXTC( ompNumThreads, matCSR, vec, matAXTC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;



	// AXTC specific name
	str_formatData fd;
	char buffer[128];
	strcpy( buffer, "AXTC_HW" );
	char HW[3]; sprintf( HW, "%d", tileHalfWidth );
	strcat( buffer, HW );
	strcat( buffer, "_H" );
	char H[3];  sprintf( H,  "%d", tileHeight );
	strcat( buffer, H );
	strcpy( fd.name, buffer );
	// AXTC memory footprint
	fd.mfp =          (double) ( matAXTC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXTC->lenSEC * sizeof(UIN) ); // sec
	// AXTC occupancy ( beta )
	fd.beta = ( (double) matAXTC->nnz / (double) (matAXTC->lenAX >> 1) );
	// AXTC conversion time
	fd.ct = tc;

	printf( "fd.name:         %14s\n",    fd.name );
	printf( "fd.mfp:          %14.8lf\n", fd.mfp );
	printf( "fd.beta:         %14.8lf\n", fd.beta );
	printf( "fd.ct:           %14.8lf\n", fd.ct );

	printf( "matAXTC->nrows:  %10d\n",    matAXTC->nrows  );
	printf( "matAXTC->nnz:    %10d\n",    matAXTC->nnz    );
	printf( "matAXTC->mode:   %10s\n",    matAXTC->mode   );
	printf( "matAXTC->tileHW: %10d\n",    matAXTC->tileHW );
	printf( "matAXTC->tileH:  %10d\n",    matAXTC->tileH  );
	printf( "matAXTC->logTH:  %10d\n",    matAXTC->logTH  );
	printf( "matAXTC->tileN:  %10d\n",    matAXTC->tileN  );
	printf( "matAXTC->lenAX:  %10d\n",    matAXTC->lenAX  );
	printf( "matAXTC->lenSEC: %10d\n",    matAXTC->lenSEC );
	printf( "matAXTC->lenCON: %10d\n",    matAXTC->lenCON );
/*
for ( i = 0; i < matAXTC->lenCON; i++ )
	printf( "con[%2d]:%3d\n", i, matAXTC->con[i] );
for ( i = 0; i < matAXTC->lenCON; i++ )
	printf( "con[%2d]:%3d\n", i, matAXTC->con[i] );
*/

	return( fd );
}




















#endif



