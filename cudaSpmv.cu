// ┌────────────────────────────────┐
// │program: cudaSpmv.cu            │
// │author: Edoardo Coronado        │
// │date: 21-08-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#ifndef __CUDA_SPMV_HEADER__
#define __CUDA_SPMV_HEADER__



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



#ifndef CHUNK_SIZE
	#define CHUNK_SIZE 32
#endif



typedef struct { UIN cbs; char matFileName[48]; UIN ompMT; } str_inputArgs;



static str_inputArgs checkArgs( const UIN argc, char ** argv )
{
	if ( argc < 3 )
	{
		fflush(stdout);
		printf( "\n\tMissing input arguments.\n" );
		printf( "\n\tUsage:\n\n\t\t%s <cudaBlockSize> <matFileName>\n\n", argv[0] );
		printf( "\t\t\t<cudaBlockSize>:  number of threads per cuda block.\n" );
		printf( "\t\t\t<matFileName>:    file's name that contains the matrix in CSR format [string].\n" );
		fflush(stdout);
		exit( EXIT_FAILURE );
	}
	str_inputArgs sia;
	sia.cbs    = atoi( argv[1] );
	strcpy( sia.matFileName, argv[2] );
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
	FILE * fh = fopen( "HASH.txt", "r+" );
	char hash[128];
	if ( fscanf( fh, "%s", &(hash) ) != 1 ) ABORT;
	fclose( fh );
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
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
	printf( "gitHash:            %s\n", hash );
	printf( "date:               %d-%d-%d (yyyy-mm-dd)\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday );
	printf( "time:               %d:%d:%d (hh:mm:ss)\n", tm.tm_hour, tm.tm_min, tm.tm_sec );
	printf( "matFileName:        %s\n", sia.matFileName );
	#ifdef _OMP_
	printf( "ompMaxThreads:      %d\n", sia.ompMT );
	printf( "omp_schedule:       %s\n", omp_schedule );
	#endif
	printf( "FPT:                %s\n", fptMsg );
	printf( "sizeof(FPT):        %zu bytes\n", sizeof(FPT) );
	printf( "cudaBlockSize:      %d\n",  sia.cbs  );
	printf( "NUM_ITE:            %d\n", (UIN) NUM_ITE );
	printf( "CHUNK_SIZE:         %d\n", (UIN) CHUNK_SIZE ); fflush(stdout);
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



typedef struct { UIN nrows; UIN nnz; UIN rmin; FPT rave; UIN rmax; FPT rsd; UIN bw; FPT * val; UIN * row; UIN * rowStart; UIN * rowEnd; UIN * col; UIN * rl; } str_matCSR;



static str_matCSR matrixReading( const char * matFileName )
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



static void printMatrixStats( const char * matFileName, str_matCSR * matCSR )
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
		rmin          = (rmin<rl) ? rmin : rl;
		rmax          = (rmax>rl) ? rmax : rl;
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



typedef struct { char name[48]; double mfp; double beta; double ct; } str_formatData;



#ifndef GT
	#define GT( t ) { clock_gettime( CLOCK_MONOTONIC, &t ); }
#endif



static double measure_time( const struct timespec t2, const struct timespec t1 )
{
	double t = (double) ( t2.tv_sec - t1.tv_sec ) + ( (double) ( t2.tv_nsec - t1.tv_nsec ) ) * 1e-9;
	return( t );
}



static str_formatData getFormatDataCSR( str_matCSR * matCSR )
{
	// define local variables
	UIN i, ii;
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	matCSR->rowStart = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowStart );
	matCSR->rowEnd   = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowEnd   );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		for ( ii = 0; ii < matCSR->nrows; ii++ )
		{
			matCSR->rowStart[ii] = matCSR->row[ii];
			matCSR->rowEnd[ii]   = matCSR->row[ii+1];
		}
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	// format's name
	str_formatData fd;
	strcpy( fd.name, "fcsr" );
	// CSR memory footprint
	fd.mfp =          (double) (   matCSR->nnz         * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) (   matCSR->nnz         * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( ( matCSR->nrows + 1 ) * sizeof(UIN) ); // row
	fd.mfp = fd.mfp + (double) (   matCSR->nrows       * sizeof(FPT) ); // vec
	// CSR occupancy ( beta )
	fd.beta = ( (double) matCSR->nnz / (double) matCSR->nnz );
	// CSR conversion time (conversion time for MKL functions)
	fd.ct = tt / (double) NUM_ITE;
	return( fd );
}



static void init_vec( const UIN ompNT, const UIN len, FPT * vec )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( i = 0 ; i < len; i++ )
		vec[i] = (FPT) i;
	return;
}



static void fill_array( const UIN ompNT, const UIN len, FPT val, FPT * array )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( i = 0; i < len; i++ )
		array[i] = val;
	return;
}



static void ncsr( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, FPT * res )
{
	UIN i, j;
	FPT aux;
	#pragma omp parallel for default(shared) private(i,j,aux) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
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



typedef struct { double aErr; double rErr; UIN pos; } str_err;



static void get_errors( const UIN len, const FPT * ar, const FPT * ac, str_err * sErr )
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



typedef struct { char name[48]; double et; double ot; double flops; str_err sErr; } str_res;



static str_res test_ncsr( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	FPT * res = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		ncsr( ompNT, matCSR, vec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "ncsr" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
	get_errors( matCSR.nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



static __global__ void gcsr( const UIN nrows, const FPT * val, const UIN * col, const UIN * row, const FPT * x, FPT * y )
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



static __host__ str_res test_gcsr( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, const FPT * ref )
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
		gcsr <<<cudaBlockNum, cudaBlockSize>>> ( nrows, d_val, d_col, d_row, d_vec, d_res );
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
	strcpy( sr.name, "gcsr" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
	get_errors( matCSR.nrows, ref, res, &(sr.sErr) );
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



static __host__ str_res test_gcucsr( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
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
	strcpy( sr.name, "gcucsr" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( (double) matCSR.nnz * 2.0 ) / sr.et;
	get_errors( matCSR.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN ind; UIN val; } str_pair;



typedef struct { UIN nrows; UIN nnz; UIN chunkNum; UIN lenVC; UIN * permi; UIN * nmc; UIN * chp; FPT * val; UIN * col; } str_matK1;



static int orderFunction( const void * ele1, const void * ele2 )
{
	return (  ( (str_pair *) ele2 )->val - ( (str_pair *) ele1 )->val  );
}



static void getArrayPermiK1( const str_matCSR matCSR, str_matK1 * matK1 )
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



static UIN getArraysNmcChpK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	UIN i, p, n, l = 0, chunkNum = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	for ( i = 0 ; i < chunkNum; i++ )
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



static void getArraysValColK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	const UIN chunkNum = matK1->chunkNum;
	UIN chunkID, rid, row, posCSR, rowOff, posK1;
	for ( chunkID = 0; chunkID < chunkNum; chunkID++ )
	{
		for ( rid = 0; rid < CHUNK_SIZE; rid++ )
		{
			row = chunkID * CHUNK_SIZE + rid;
			if ( row == matCSR.nrows ) return;
			row = matK1->permi[row];
			for ( posCSR = matCSR.row[row], rowOff = 0; posCSR < matCSR.row[row + 1]; posCSR++, rowOff++ )
			{
				posK1             = matK1->chp[chunkID] + rowOff * CHUNK_SIZE + rid;
				matK1->val[posK1] = matCSR.val[posCSR];
				matK1->col[posK1] = matCSR.col[posCSR];
			}
		}
	}
	return;
}



static str_formatData getFormatDataK1( const UIN blockSize, const str_matCSR matCSR, const FPT * vec, str_matK1 * matK1 )
{
	// get K1 parameters
	matK1->nrows     = matCSR.nrows;
	matK1->nnz       = matCSR.nnz;
	matK1->chunkNum  = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	matK1->permi     = (UIN *) calloc( ( matK1->chunkNum + 1 ) * CHUNK_SIZE, sizeof(UIN) ); TEST_POINTER( matK1->permi );
	matK1->nmc       = (UIN *) calloc(   matK1->chunkNum,                    sizeof(UIN) ); TEST_POINTER( matK1->nmc   );
	matK1->chp       = (UIN *) calloc(   matK1->chunkNum,                    sizeof(UIN) ); TEST_POINTER( matK1->chp   );
	UIN i;
	for ( i = 0; i < ( matK1->chunkNum + 1 ) * CHUNK_SIZE; i++ )
		matK1->permi[i] = 0;
	// get matK1
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayPermiK1( matCSR, matK1 );
		GT( t2 );
		ti = measure_time( t2, t1 );
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
		ti = measure_time( t2, t1 );
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
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "fk1" );
	// K1 memory footprint
	fd.mfp =          (double) ( matK1->chunkNum * sizeof(UIN) ); // nmc
	fd.mfp = fd.mfp + (double) ( matK1->chunkNum * sizeof(UIN) ); // chp
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(UIN) ); // permi
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(FPT) ); // vec
	// K1 occupancy ( beta )
	fd.beta = ( (double) matK1->nnz / (double) (matK1->lenVC) );
	// K1 conversion time
	fd.ct = tc;
	return( fd );
}




static __global__ void gk1( const int NROWS, const FPT * val, const UIN * col, const UIN * nmc, const UIN * chp, const UIN * permi, const FPT * x, FPT * y )
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



static __host__ str_res test_gk1( const UIN cudaBlockSize, const str_matK1 matK1, const FPT * vec, const FPT * ref )
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
		gk1 <<<cudaBlockNum, cudaBlockSize>>> (  matK1.nrows, d_val, d_col, d_nmc, d_chp, d_permi, d_vec, d_res );
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
	strcpy( sr.name, "gk1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matK1.nnz ) ) / sr.et;
	get_errors( matK1.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



typedef struct { UIN nrows; UIN nnz; UIN hbs; UIN lenAX; UIN lenBRP; UIN lenMAPX; FPT * ax; UIN * brp; UIN * mapx; } str_matAXC;



static UIN get_brpAXC( const str_matCSR matCSR, str_matAXC * matAXC )
{
	const UIN hbs   = matAXC->hbs;
	const UIN nrows = matAXC->nrows;
	      UIN rowID, brickNum;
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		brickNum               = ( matCSR.rl[rowID] + hbs - 1 ) / hbs;
		matAXC->brp[rowID + 1] = matAXC->brp[rowID]  + ( 2 * brickNum * hbs );
	}
	return( matAXC->brp[matAXC->nrows] );
}



static void get_axAXC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	const UIN hbs   = matAXC->hbs;
	const UIN nrows = matAXC->nrows;
	      UIN rowID, posAX, counter, posCSR;
	#pragma omp parallel for default(shared) private(rowID,posAX,counter,posCSR) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		posAX   = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			matAXC->ax[posAX]       = matCSR.val[posCSR];
			matAXC->ax[posAX + hbs] = vec[matCSR.col[posCSR]];
			if ( counter == (hbs - 1) )
			{
				posAX  = posAX + 1 + hbs;
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



static void get_mapxAXC( const UIN ompNT, const str_matCSR matCSR, str_matAXC * matAXC )
{
	const UIN nrows = matAXC->nrows;
	      UIN rowID, pos1, pos2, pos, eleID;
	#pragma omp parallel for default(shared) private(rowID,pos1,pos2,pos,eleID) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		pos1 = matCSR.row[rowID];
		pos2 = matCSR.row[rowID+1];
		pos  = matAXC->brp[rowID]>>1;
		for ( eleID = pos1; eleID < pos2; eleID++ )
		{
			matAXC->mapx[pos] = matCSR.col[eleID];
			pos++;
		}
	}
	return;
}



static str_formatData getFormatDataAXC( const UIN ompNT, const UIN hbs, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	// get AXC parameters
	matAXC->nrows  = matCSR.nrows;
	matAXC->nnz    = matCSR.nnz;
	matAXC->hbs    = hbs;
	matAXC->lenBRP = matCSR.nrows + 1;
	matAXC->brp    = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXC
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = get_brpAXC( matCSR, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXC->ax      = (FPT *) calloc( matAXC->lenAX,   sizeof(FPT) ); TEST_POINTER( matAXC->ax );
	matAXC->lenMAPX = (matAXC->lenAX >> 1) + 8;
	matAXC->mapx    = (UIN *) calloc( matAXC->lenMAPX, sizeof(UIN) ); TEST_POINTER( matAXC-> mapx );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		get_mapxAXC( ompNT, matCSR, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		get_axAXC( ompNT, matCSR, vec, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "faxc" );
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



static __global__ void gaxc( const FPT * ax, const UIN * brp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID = tidGRID >> 5;
	const UIN tidWARP = tidGRID & 31;
	const UIN p1      = brp[widGRID]   + tidWARP;
	const UIN p2      = brp[widGRID+1] + tidWARP;
	      UIN pAX;
	      FPT val = 0.0, red = 0.0;
	for ( pAX = p1; pAX < p2; pAX = pAX + 64 )
	{
		val = ax[pAX] * ax[pAX+32];
		val = val + __shfl_down_sync( FULL_MASK, val, 16 );
		val = val + __shfl_down_sync( FULL_MASK, val,  8 );
		val = val + __shfl_down_sync( FULL_MASK, val,  4 );
		val = val + __shfl_down_sync( FULL_MASK, val,  2 );
		val = val + __shfl_down_sync( FULL_MASK, val,  1 );
		red = red + val;
	}
	if (tidWARP == 0) y[widGRID] = red;
	return;
}



static __host__ str_res test_gaxc( const UIN cudaBlockSize, const str_matAXC matAXC, const FPT * ref )
{
	// 
	UIN cudaBlockNum = matAXC.nrows * 32;
	    cudaBlockNum = ( cudaBlockNum + cudaBlockSize - 1 ) / cudaBlockSize;
	// allocate memory on GPU
	FPT * d_ax;    HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,    matAXC.lenAX                * sizeof(FPT) ) ); TEST_POINTER( d_ax    );
	UIN * d_brp;   HANDLE_CUDA_ERROR( cudaMalloc( &d_brp,   matAXC.lenBRP               * sizeof(UIN) ) ); TEST_POINTER( d_brp   );
	FPT * d_res;   HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matAXC.nrows                * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,    matAXC.ax,    matAXC.lenAX                  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_brp,   matAXC.brp,   matAXC.lenBRP                 * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gaxc <<<cudaBlockNum, cudaBlockSize>>> ( d_ax, d_brp, d_res );
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
	HANDLE_CUDA_ERROR( cudaFree( d_ax    ) );
	HANDLE_CUDA_ERROR( cudaFree( d_brp   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res   ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gaxc" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	get_errors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}



/*
typedef struct{ UIN nrows; UIN nnz; char mode[8]; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenSEC; UIN lenCON; UIN log; UIN bs; FPT * ax; UIN * sec; UIN * con; } str_matAXT;



static void getArraysLenAXT( const str_matCSR matCSR, str_matAXT * matAXT )
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
		         positions = ( strcmp( mode, "UNC" ) == 0 ) ? ( ( ( rowLen + grpLen - 1 ) / grpLen ) * grpLen ) : ( rowLen ) ;
		           columns = ( positions + th - 1 ) / th;
		      totalColumns = totalColumns + columns;
		       rowStartPos = rowStartPos + positions;
	}
	     totalTiles = ( totalColumns + thw - 1 ) / thw;
	 matAXT->tileN = totalTiles;
	 matAXT->lenAX = totalTiles * 2 * ths;
	if      ( (strcmp(mode, "UNC")==0) && (th==1) ) matAXT->lenSEC = totalTiles;
	else if ( (strcmp(mode, "UNC")==0) && (th!=1) ) matAXT->lenSEC = totalTiles * thw;
	else                                            matAXT->lenSEC = totalTiles * ths;
	return;
}



static void getArraysLenAXT_UNC_H1( const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW;
	const UIN ts    = 2 * ths;
	      UIN rid, tiles, totalTiles = 0;
	 matAXT->con[0] = 0;
	for ( rid = 0; rid < nrows; rid++ )
	{
		tiles              = ( matCSR.rl[rid] + ths - 1 ) / ths;
		totalTiles         = totalTiles + tiles;
		matAXT->con[rid+1] = matAXT->con[rid] + tiles * ts;
	}
	matAXT->tileN  = totalTiles;
	matAXT->lenAX  = totalTiles * ts;
	matAXT->lenSEC = totalTiles;
	return;
}



static void getArraysLenAXT_COM_H1( const UIN ompNT, const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW;
	      UIN rid, totalElements = 0, totalTiles;
	#pragma omp parallel for default(shared) private(rid) reduction(+:totalElements) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
		totalElements = totalElements + matCSR.rl[rid];
	totalTiles     = ( totalElements + ths - 1 ) / ths;
	matAXT->tileN  =     totalTiles;
	matAXT->lenAX  = 2 * totalTiles * ths;
	matAXT->lenSEC =     totalTiles * ths;
	return;
}



static void getArraysLenAXT_COM( const UIN ompNT, const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW * matAXT->tileH;
	      UIN rid, totalElements = 0, totalTiles;
	#pragma omp parallel for default(shared) private(rid) reduction(+:totalElements) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
		totalElements = totalElements + matCSR.rl[rid];
	totalTiles     = ( totalElements + ths - 1 ) / ths;
	matAXT->tileN  =     totalTiles;
	matAXT->lenAX  = 2 * totalTiles * ths;
	matAXT->lenSEC =     totalTiles * ths;
	return;
}



static void getArraysAxSecAXT_UNC_H1( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
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



static void getArraysAxSecAXT_UNC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
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



static void getArraysAxSecAXT_COM_H1( const UIN bs, const UIN log, const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	      UIN rowID, rowLen, eleCtr, posCSR, bid, bco, tid, tco, posAX, posSEC, posBLK, q1, q2, offset, blk;
	#pragma omp parallel for default(shared) private(rowID,rowLen,eleCtr,posCSR,bid,bco,tid,tco,posAX,posSEC,posBLK,q1,q2,offset) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			eleCtr = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				bid                   = ((posCSR+bs)/bs)-1;
				bco                   =   posCSR%bs;
				tid                   = ((posCSR+thw)/thw)-1;
				tco                   =  posCSR%thw;
				posAX                 = tid * 2 * thw + tco;
				posSEC                = tid     * thw + tco;
				posBLK                = bid     * bs  + bco;
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( (eleCtr==0) || (bco==0))
				{
					q1     = rowLen - eleCtr - 1;
					q2     = bs - 1 - bco;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[posSEC] = rowID<<log | offset;
				}
				eleCtr++;
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_COM( const UIN ompNT, str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN th    = matAXT->tileH;
	const UIN thw   = matAXT->tileHW;
	const UIN log   = matAXT->log;
	const UIN ths   = th * thw;
	const UIN ts    =  2 * ths;
	      UIN rid, rl, ec, pCSR, tid, fid, cid, pAX, pSEC, q1, q2, offset;
	#pragma omp parallel for default(shared) private(rid,rl,ec,pCSR,tid,fid,cid,pAX,pSEC,q1,q2,offset) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
	{
		rl = matCSR.rl[rid];
		if (rl>0)
		{
			ec = 0;
			for ( pCSR = matCSR.row[rid]; pCSR < matCSR.row[rid+1]; pCSR++ )
			{
				tid  = ( (pCSR + ths) / ths ) - 1;
				fid  = pCSR % th;
				cid  = ( ( (pCSR - tid * ths) + th ) / th ) - 1;
				pAX  = tid * ts  + 2 * fid * thw + cid;
				pSEC = tid * ths +     fid * thw + cid;
				matAXT->ax[pAX]     = matCSR.val[pCSR];
				matAXT->ax[pAX+thw] = vec[matCSR.col[pCSR]];
				if ( (ec==0) || (fid==0) )
				{
					q1     = rl - ec - 1;
					q2     = th - 1 - fid;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[pSEC] = rid << log | offset;
				}
				ec++;
			}
		}
	}
	return;
}



static void fill_array_uin( const UIN ompNT, const UIN len, const UIN value, UIN * array )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for( i = 0; i < len; i++ )
	array[i] = value;
	return;
}



static str_formatData getFormatDataAXT( const UIN ompNT, const UIN bs, const UIN thw, const UIN th, const char * mode, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	// set AXT parameters
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->bs     = bs;
	matAXT->tileHW = thw;
	matAXT->tileH  = th;
	strcpy( matAXT->mode, mode );
	matAXT->lenCON = matCSR.nrows;
	   matAXT->con = (UIN *) calloc( matAXT->lenCON + 1, sizeof(UIN) ); TEST_POINTER( matAXT->con );
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileH) >> i) == 1 ) matAXT->logTH = i;
	// get AXT arrays' length
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	if (strcmp(mode,"UNC")==0)
	{
		if (th == 1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_UNC_H1( matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT( matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	else
	{
		if (th == 1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_COM_H1( ompNT, matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_COM( ompNT, matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// get arrays ax[] and sec[]
	matAXT->ax  = (FPT *) _mm_malloc( matAXT->lenAX  * sizeof(FPT), 64 );  TEST_POINTER( matAXT->ax  ); fill_array    ( ompNT, matAXT->lenAX,  0.0, matAXT->ax  );
	matAXT->sec = (UIN *) _mm_malloc( matAXT->lenSEC * sizeof(UIN), 64 );  TEST_POINTER( matAXT->sec ); fill_array_uin( ompNT, matAXT->lenSEC,   0, matAXT->sec );
	tt = 0.0;
	char buffer[48];
	if (strcmp(mode,"UNC")==0)
	{
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_UNC_H1( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char THW[5]; sprintf( THW, "%d", thw );
			strcpy( buffer, "f_axt_unc_h1_hw" );
			strcat( buffer, THW );
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_UNC( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char TH[5];  sprintf( TH,  "%d", th  );
			char THW[5]; sprintf( THW, "%d", thw );
			strcpy( buffer, "f_axt_unc_h" );
			strcat( buffer, TH );
			strcat( buffer, "_hw" );
			strcat( buffer, THW );
		}
	}
	else
	{
		if (th==1)
		{
			for ( i = 1; i < 10; i++ )
			{
				if ((bs>>i) == 1)
				{
					matAXT->log = i;
					break;
				}
			}
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM_H1( bs, matAXT->log, ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char THW[5]; sprintf( THW, "%d", thw );
			char BS[5];  sprintf( BS, "%d", bs );
			strcpy( buffer, "f_axt_com_h1_hw" );
			strcat( buffer, THW );
			strcat( buffer, "_bs" );
			strcat( buffer, BS );
		}
		else
		{
			for ( i = 1; i < 10; i++ )
			{
				if ((th>>i) == 1)
				{
					matAXT->log = i;
					break;
				}
			}
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char TH[5];  sprintf( TH,  "%d", th  );
			char THW[5]; sprintf( THW, "%d", thw );
			strcpy( buffer, "f_axt_com_h" );
			strcat( buffer, TH );
			strcat( buffer, "_hw" );
			strcat( buffer, THW );
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXTC specific name
	str_formatData fd;
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



static void int_axt_unc_h1( const UIN ompNT, const UIN tn, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN stride = 2 * thw;
	      UIN tid, posAX, rowID;
	      FPT red;
	  __m512d vtMat, vtVec, vtPro;
	#pragma omp parallel for default(shared) private(tid,posAX,vtMat,vtVec,vtPro,red,rowID) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( tid = 0; tid < tn; tid++ )
	{
		posAX    = tid * stride;
		vtMat    = _mm512_load_pd( &ax[posAX]       );
		vtVec    = _mm512_load_pd( &ax[posAX + thw] );
		vtPro    = _mm512_mul_pd( vtMat, vtVec );
		red      = _mm512_reduce_add_pd( vtPro );
		rowID    = rwp[tid];
		#pragma omp atomic
		y[rowID] = y[rowID] + red;
	}
	return;
}



static str_res test_int_axt_unc_h1( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN tileN  = matAXT.tileN;
	const UIN tileHW = matAXT.tileHW;
	const UIN nrows  = matAXT.nrows;
	const UIN nnz    = matAXT.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_unc_h1( ompNT, tileN, tileHW, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	char THW[5]; sprintf( THW, "%d", tileHW );
	char buffer[48];
	strcpy( buffer, "p_axt_unc_h1_hw" );
	strcat( buffer, THW );
	// store results
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	get_errors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}


static void int_axt_unc( const UIN ompNT, const UIN tn, const UIN th, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN stride = 2 * thw;
	const UIN ts     = th * stride;
	      UIN tid, posAX, posRWP, rowID, i;
	      FPT tmp[thw];
	      FPT red;
	  __m512d vtMat, vtVec, vtPro, vtSum;
	  __m512i vtRid;
	#pragma omp parallel for default(shared) private(tid,vtSum,posAX,vtMat,vtVec,vtPro,tmp,i,posRWP,rowID) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( tid = 0; tid < tn; tid++ )
	{
		vtSum = _mm512_setzero_pd();
		for ( posAX = tid * ts; posAX < (tid + 1) * ts; posAX = posAX + stride )
		{
			vtMat = _mm512_load_pd( &ax[posAX]       );
			vtVec = _mm512_load_pd( &ax[posAX + thw] );
			vtPro = _mm512_mul_pd( vtMat, vtVec );
			vtSum = _mm512_add_pd( vtSum, vtPro );
		}
		_mm512_store_pd( tmp, vtSum );
		for ( i = 0, posRWP = tid * thw; i < thw; i++, posRWP++ )
		{
			rowID    = rwp[posRWP];
			y[rowID] = y[rowID] + tmp[i];
		}
	}
	return;
}



static str_res test_int_axt_unc( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN tileNum = matAXT.tileN;
	const UIN tileH   = matAXT.tileH;
	const UIN tileHW  = matAXT.tileHW;
	const UIN nrows   = matAXT.nrows;
	const UIN nnz     = matAXT.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	FPT * res = (FPT *) _mm_malloc( nrows * sizeof(FPT), 64 ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_unc( ompNT, tileNum, tileH, tileHW, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	char TH[5];      sprintf( TH,   "%d", tileH  );
	char THW[5];     sprintf( THW,  "%d", tileHW );
	char buffer[48]; strcpy( buffer, "p_axt_unc_h" );
	strcat( buffer, TH );
	strcat( buffer, "_hw" );
	strcat( buffer, THW );
	// store results
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	get_errors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



static __m512d inc_scan8( __m512d vtVal )
{
	__m512i  vtIdx = _mm512_set_epi64( 6, 6, 4, 4, 2, 2, 0, 0 );
	__m512d  vtAux = _mm512_permutexvar_pd( vtIdx, vtVal );
	__mmask8 cmask = 0xAA;
	__m512d  vtRed = _mm512_mask_add_pd( vtVal, cmask, vtVal, vtAux );
	         vtIdx = _mm512_set_epi64( 5, 6, 5, 4, 1, 2, 1, 0 );
	         vtAux = _mm512_permutexvar_pd( vtIdx, vtRed );
	         cmask = 0x88;
	         vtRed = _mm512_mask_add_pd( vtRed, cmask, vtRed, vtAux );
	         vtIdx = _mm512_set_epi64( 3, 6, 5, 4, 3, 2, 1, 0 );
	         vtAux = _mm512_permutexvar_pd( vtIdx, vtRed );
	         cmask = 0x80;
	         vtRed = _mm512_mask_add_pd( vtRed, cmask, vtRed, vtAux );
	         vtIdx = _mm512_set_epi64( 7, 6, 3, 4, 3, 2, 1, 0 );
	         vtAux = _mm512_permutexvar_pd( vtIdx, vtRed );
	         cmask = 0x20;
	         vtRed = _mm512_mask_add_pd( vtRed, cmask, vtRed, vtAux );
	         vtIdx = _mm512_set_epi64( 7, 5, 5, 3, 3, 1, 1, 0 );
	         vtAux = _mm512_permutexvar_pd( vtIdx, vtRed );
	         cmask = 0x54;
	         vtRed = _mm512_mask_add_pd( vtRed, cmask, vtRed, vtAux );
	return( vtRed );
}



static void int_axt_com_h1_bs64( const UIN ompNT, const UIN bn, const UIN lenRWP, const FPT * ax, const UIN * rwp, FPT * y )
{
	      UIN bid, blkOff, tid, posAX, posBLK, ep, ro, r, o;
	  __m512d vtMat, vtVec, vtVal, vtRed, vtAcu;
	      FPT v;
	      FPT blk1[64];
	      FPT blk2[64];
	      FPT blk3[8];
	#pragma omp parallel for default(shared) private(bid,blkOff,tid,posAX,posBLK,ep,ro,r,o,vtMat,vtVec,vtVal,vtRed,vtAcu,v,blk1,blk2,blk3) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( bid = 0; bid < bn; bid++ )
	{
		blkOff = bid * 128;
		for ( tid = 0; tid < 8; tid++ )
		{
			posAX     = blkOff + tid * 16;
			vtMat     = _mm512_load_pd( &ax[posAX]     );
			vtVec     = _mm512_load_pd( &ax[posAX + 8] );
			vtVal     = _mm512_mul_pd( vtMat, vtVec );
			posBLK    = tid *  8;
			_mm512_store_pd( &blk1[posBLK], vtVal );
			vtRed     = inc_scan8( vtVal );
			_mm512_store_pd( &blk2[posBLK], vtRed );
			blk3[tid] = blk2[posBLK+7];
		}
		vtVal = _mm512_load_pd( blk3 );
		vtRed = inc_scan8( vtVal );
		_mm512_store_pd( blk3, vtRed );
		for ( tid = 1; tid < 8; tid++ )
		{
			posBLK = tid * 8;
			vtAcu  = _mm512_set1_pd( blk3[tid-1]   );
			vtVal  = _mm512_load_pd( &blk2[posBLK] );
			vtRed  = _mm512_add_pd( vtAcu, vtVal );
			_mm512_store_pd( &blk2[posBLK], vtRed );
		}
		blkOff = bid * 64;
		ep     = blkOff + 64;
		ep     = (ep>lenRWP) ? lenRWP : ep ;
		#pragma omp parallel for default(shared) private(posBLK,tid,ro,r,o,v) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
		for ( posBLK = blkOff, tid = 0; posBLK < ep; posBLK++, tid++ )
		{
			ro = rwp[posBLK];
			if (ro!=0)
			{
				r = ro >> 6;
				o = ro & 63;
				v = blk2[tid+o] - blk2[tid] + blk1[tid];
				#pragma omp atomic
				y[r] = y[r] + v;
			}
		}
	}
	return;
}



static void int_axt_com_h1_bs512( const UIN ompNT, const UIN bn, const UIN lenRWP, const FPT * ax, const UIN * rwp, FPT * y )
{
	      UIN bid, blkOff, tid, posAX, posBLK, ep, ro, r, o;
	  __m512d vtMat, vtVec, vtVal, vtRed, vtAcu;
	      FPT v;
	      FPT blk1[512];
	      FPT blk2[512];
	      FPT blk3[64];
	      FPT blk4[8];
	#pragma omp parallel for default(shared) private(bid,blkOff,tid,posAX,posBLK,ep,ro,r,o,vtMat,vtVec,vtVal,vtRed,vtAcu,v,blk1,blk2,blk3,blk4) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( bid = 0; bid < bn; bid++ )
	{
		blkOff = bid * 1024;
		for ( tid = 0; tid < 64; tid++ )
		{
			posAX     = blkOff + tid * 16;
			vtMat     = _mm512_load_pd( &ax[posAX]     );
			vtVec     = _mm512_load_pd( &ax[posAX + 8] );
			vtVal     = _mm512_mul_pd( vtMat, vtVec );
			posBLK    = tid *  8;
			_mm512_store_pd( &blk1[posBLK], vtVal );
			vtRed     = inc_scan8( vtVal );
			_mm512_store_pd( &blk2[posBLK], vtRed );
			blk3[tid] = blk2[posBLK+7];
		}
		for ( tid = 0; tid < 8; tid++ )
		{
			posBLK  = tid *  8;
			vtVal   = _mm512_load_pd( &blk3[posBLK] );
			vtRed   = inc_scan8( vtVal );
			_mm512_store_pd( &blk3[posBLK], vtRed );
			blk4[tid] = blk3[posBLK+7];
		}
		vtVal = _mm512_load_pd( blk4 );
		vtRed = inc_scan8( vtVal );
		_mm512_store_pd( blk4, vtRed );
		for ( tid = 1; tid < 8; tid++ )
		{
			posBLK = tid * 8;
			vtAcu  = _mm512_set1_pd( blk4[tid-1]   );
			vtVal  = _mm512_load_pd( &blk3[posBLK] );
			vtRed  = _mm512_add_pd( vtAcu, vtVal );
			_mm512_store_pd( &blk3[posBLK], vtRed );
		}
		for ( tid = 1; tid < 64; tid++ )
		{
			posBLK = tid * 8;
			vtAcu  = _mm512_set1_pd( blk3[tid-1]   );
			vtVal  = _mm512_load_pd( &blk2[posBLK] );
			vtRed  = _mm512_add_pd( vtAcu, vtVal );
			_mm512_store_pd( &blk2[posBLK], vtRed );
		}
		blkOff = bid * 512;
		ep     = blkOff + 512;
		ep     = (ep>lenRWP) ? lenRWP : ep ;
		#pragma omp parallel for default(shared) private(posBLK,tid,ro,r,o,v) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
		for ( posBLK = blkOff, tid = 0; posBLK < ep; posBLK++, tid++ )
		{
			ro = rwp[posBLK];
			if (ro!=0)
			{
				r = ro >> 9;
				o = ro & 511;
				v = blk2[tid+o] - blk2[tid] + blk1[tid];
				#pragma omp atomic
				y[r] = y[r] + v;
			}
		}
	}
	return;
}





static str_res test_int_axt_com_h1_bs64( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN nrows  = matAXT.nrows;
	const UIN nnz    = matAXT.nnz;
	const UIN lenAX  = matAXT.lenAX;
	const UIN lenSEC = matAXT.lenSEC;
	const UIN log    = matAXT.log;
	const UIN tn     = matAXT.tileN;
	const UIN thw    = matAXT.tileHW;
	const UIN bs     = matAXT.bs;
	const UIN ts     = 2 * thw;
	const UIN bn     = ( (tn * thw) + bs - 1 ) / bs;
	const UIN tpb    = bs / thw;
	FPT * res = (FPT *) _mm_malloc( nrows   * sizeof(FPT), 64 ); TEST_POINTER( res );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	UIN i;
	const UIN newLenAX  = 2 * bn * bs; FPT * ax  = (FPT *) _mm_malloc( newLenAX  * sizeof(FPT), 64 ); TEST_POINTER( ax  );
	const UIN newLenSEC =     bn * bs; UIN * sec = (UIN *)     calloc( newLenSEC, sizeof(UIN) );      TEST_POINTER( sec );
	fill_array( ompNT, newLenAX, 0, ax );
	cblas_dcopy( lenAX, matAXT.ax, 1, ax, 1 );
	memcpy( sec, matAXT.sec, lenSEC * sizeof(UIN) );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_com_h1_bs64( ompNT, bn, lenSEC, ax, sec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	_mm_free( ax );
	free( sec );
	// store results
	char THW[5]; sprintf( THW, "%d", thw );
	char BS[5];  sprintf( BS,  "%d", bs  );
	char buffer[48];
	strcpy( buffer, "p_axt_com_h1_hw" );
	strcat( buffer, THW );
	strcat( buffer, "_bs" );
	strcat( buffer, BS );
	// store results
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	get_errors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



static str_res test_int_axt_com_h1_bs512( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN nrows  = matAXT.nrows;
	const UIN nnz    = matAXT.nnz;
	const UIN lenAX  = matAXT.lenAX;
	const UIN lenSEC = matAXT.lenSEC;
	const UIN log    = matAXT.log;
	const UIN tn     = matAXT.tileN;
	const UIN thw    = matAXT.tileHW;
	const UIN bs     = matAXT.bs;
	const UIN ts     = 2 * thw;
	const UIN bn     = ( (tn * thw) + bs - 1 ) / bs;
	const UIN tpb    = bs / thw;
	FPT * res = (FPT *) _mm_malloc( nrows   * sizeof(FPT), 64 ); TEST_POINTER( res );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	UIN i;
	const UIN newLenAX  = 2 * bn * bs; FPT * ax  = (FPT *) _mm_malloc( newLenAX  * sizeof(FPT), 64 ); TEST_POINTER( ax  );
	const UIN newLenSEC =     bn * bs; UIN * sec = (UIN *)     calloc( newLenSEC, sizeof(UIN) );      TEST_POINTER( sec );
	fill_array( ompNT, newLenAX, 0, ax );
	cblas_dcopy( lenAX, matAXT.ax, 1, ax, 1 );
	memcpy( sec, matAXT.sec, lenSEC * sizeof(UIN) );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_com_h1_bs512( ompNT, bn, lenSEC, ax, sec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	_mm_free( ax );
	free( sec );
	// store results
	char THW[5]; sprintf( THW, "%d", thw );
	char BS[5];  sprintf( BS,  "%d", bs  );
	char buffer[48];
	strcpy( buffer, "p_axt_com_h1_hw" );
	strcat( buffer, THW );
	strcat( buffer, "_bs" );
	strcat( buffer, BS );
	// store results
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	get_errors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



static void int_axt_com( const UIN ompNT, const UIN tn, const UIN th, const UIN log, const UIN thw, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN ths = th * thw;
	const UIN ts  =  2 * ths;
	      UIN tid, off, f, pAX, pBLK, ro, r, o;
	  __m512d vtMat, vtVec, vtVal, vtAcu;
	      FPT blk1[th*thw];
	      FPT blk2[th*thw];
	      FPT v;
	#pragma omp parallel for default(shared) private(tid,off,f,pAX,pBLK,ro,r,o,vtMat,vtVec,vtVal,vtAcu,v,blk1,blk2) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( tid = 0; tid < tn; tid++ )
	{
		off = tid * ts;
		vtAcu = _mm512_setzero_pd();
		for ( f = 0; f < th; f++ )
		{
			pAX   = off + f * 16;
			vtMat = _mm512_load_pd( &ax[pAX]     );
			vtVec = _mm512_load_pd( &ax[pAX + 8] );
			vtVal = _mm512_mul_pd( vtMat, vtVec );
			pBLK  = f * 8;
			_mm512_store_pd( &blk1[pBLK], vtVal );
			vtAcu = _mm512_add_pd( vtAcu, vtVal );
			_mm512_store_pd( &blk2[pBLK], vtAcu );
		}
		off = tid * ths;
		#pragma omp parallel for default(shared) private(f,ro,r,o,v) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
		for ( f = 0; f < ths; f++ )
		{
			ro = rwp[off + f];
			if (ro != 0)
			{
				r    = ro >> log;
				o    = ro & (th-1);
				v    = blk2[f+(o*thw)] - blk2[f] + blk1[f];
				#pragma omp atomic
				y[r] = y[r] + v;
			}
		}
	}
	return;
}



static str_res test_int_axt_com( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN nrows  = matAXT.nrows;
	const UIN nnz    = matAXT.nnz;
	const UIN tn     = matAXT.tileN;
	const UIN th     = matAXT.tileH;
	const UIN thw    = matAXT.tileHW;
	const UIN log    = matAXT.log;
	FPT * res = (FPT *) _mm_malloc( nrows   * sizeof(FPT), 64 ); TEST_POINTER( res );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_com( ompNT, tn, th, log, thw, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	// store results
	char TH[5];  sprintf( TH,  "%d", th );
	char THW[5]; sprintf( THW, "%d", thw );
	char buffer[48];
	strcpy( buffer, "p_axt_com_h" );
	strcat( buffer, TH );
	strcat( buffer, "_hw" );
	strcat( buffer, THW );
	// store results
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	get_errors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



*/

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
		printf( "cudaDeviceSelected:                               %d <-------------------\n", cudaDeviceID );
	}

	// CSR format  ------------------------------------------------------------------------------------------------------------------
	// read matrix in CSR format
	str_matCSR matCSR = matrixReading( sia.matFileName );
	// print matrix's statistics
	printMatrixStats( sia.matFileName, &matCSR );

	// get memory footprint, occupancy (beta) and conversion time
	str_formatData fd01 = getFormatDataCSR( &matCSR );

	// CSR format  ------------------------------------------------------------------------------------------------------------------
	// init vectors to perform SpMV multiplication and check errors (spM * vr = yr)
	FPT * vr = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( vr );
	init_vec( sia.ompMT, matCSR.nrows, vr );
	FPT * yr = (FPT *) calloc( matCSR.nrows,  sizeof(FPT) ); TEST_POINTER( yr );
	ncsr( sia.ompMT, matCSR, vr, yr );
	// test CSR kernels
	str_res sr01 = test_ncsr( sia.ompMT, matCSR, vr, yr );
	str_res sr02 = test_gcsr( sia.cbs, matCSR, vr, yr );
	str_res sr03 = test_gcucsr( matCSR, vr, yr );
	// CSR format  ------------------------------------------------------------------------------------------------------------------

	// K1 format  -------------------------------------------------------------------------------------------------------------------
	str_matK1 matK1; str_formatData fd02 = getFormatDataK1( CHUNK_SIZE, matCSR, vr, &matK1 );
	str_res sr04 = test_gk1( sia.cbs, matK1, vr, yr );
	// K1 format  -------------------------------------------------------------------------------------------------------------------

	// AXC format  ------------------------------------------------------------------------------------------------------------------
	str_matAXC matAXC; str_formatData fd03 = getFormatDataAXC( sia.ompMT, 32, matCSR, vr, &matAXC );
	str_res sr05 = test_gaxc( sia.cbs, matAXC, yr );
	// AXC format  ------------------------------------------------------------------------------------------------------------------

/*
	// AXT format  ------------------------------------------------------------------------------------------------------------------
	str_matAXT matAXT1;  str_formatData fd04 = getFormatDataAXT( sia.ompMT,   64, 8,   1, "UNC", matCSR, vr, &matAXT1 );
	str_matAXT matAXT2;  str_formatData fd05 = getFormatDataAXT( sia.ompMT,   64, 8,   4, "UNC", matCSR, vr, &matAXT2 );
	str_matAXT matAXT3;  str_formatData fd06 = getFormatDataAXT( sia.ompMT,   64, 8,   8, "UNC", matCSR, vr, &matAXT3 );
	str_matAXT matAXT4;  str_formatData fd07 = getFormatDataAXT( sia.ompMT,   64, 8,  12, "UNC", matCSR, vr, &matAXT4 );
	str_matAXT matAXT5;  str_formatData fd08 = getFormatDataAXT( sia.ompMT,   64, 8,  16, "UNC", matCSR, vr, &matAXT5 );
	str_matAXT matAXT6;  str_formatData fd09 = getFormatDataAXT( sia.ompMT,   64, 8,  20, "UNC", matCSR, vr, &matAXT6 );
	str_matAXT matAXT7;  str_formatData fd10 = getFormatDataAXT( sia.ompMT,   64, 8,  24, "UNC", matCSR, vr, &matAXT7 );
	str_matAXT matAXT8;  str_formatData fd11 = getFormatDataAXT( sia.ompMT,   64, 8,  28, "UNC", matCSR, vr, &matAXT8 );
	str_matAXT matAXT9;  str_formatData fd12 = getFormatDataAXT( sia.ompMT,   64, 8,  32, "UNC", matCSR, vr, &matAXT9 );
	str_matAXT matAXT10; str_formatData fd13 = getFormatDataAXT( sia.ompMT,   64, 8,  36, "UNC", matCSR, vr, &matAXT10 );
	str_matAXT matAXT11; str_formatData fd14 = getFormatDataAXT( sia.ompMT,   64, 8,  40, "UNC", matCSR, vr, &matAXT11 );
	str_matAXT matAXT12; str_formatData fd15 = getFormatDataAXT( sia.ompMT,   64, 8,   1, "COM", matCSR, vr, &matAXT12 );
	str_matAXT matAXT13; str_formatData fd16 = getFormatDataAXT( sia.ompMT,  512, 8,   1, "COM", matCSR, vr, &matAXT13 );
	str_matAXT matAXT14; str_formatData fd17 = getFormatDataAXT( sia.ompMT,   64, 8,   8, "COM", matCSR, vr, &matAXT14 );
	str_matAXT matAXT15; str_formatData fd18 = getFormatDataAXT( sia.ompMT,   64, 8,  16, "COM", matCSR, vr, &matAXT15 );
	str_matAXT matAXT16; str_formatData fd19 = getFormatDataAXT( sia.ompMT,   64, 8,  32, "COM", matCSR, vr, &matAXT16 );
	str_matAXT matAXT17; str_formatData fd20 = getFormatDataAXT( sia.ompMT,   64, 8,  64, "COM", matCSR, vr, &matAXT17 );
	str_matAXT matAXT18; str_formatData fd21 = getFormatDataAXT( sia.ompMT,   64, 8, 128, "COM", matCSR, vr, &matAXT18 );
	str_matAXT matAXT19; str_formatData fd22 = getFormatDataAXT( sia.ompMT,   64, 8, 256, "COM", matCSR, vr, &matAXT19 );
	str_matAXT matAXT20; str_formatData fd23 = getFormatDataAXT( sia.ompMT,   64, 8, 512, "COM", matCSR, vr, &matAXT20 );
	str_res sr06 = test_int_axt_unc_h1( sia.ompMT, matAXT1, yr );
	str_res sr07 = test_int_axt_unc( sia.ompMT, matAXT2, yr );
	str_res sr08 = test_int_axt_unc( sia.ompMT, matAXT3, yr );
	str_res sr09 = test_int_axt_unc( sia.ompMT, matAXT4, yr );
	str_res sr10 = test_int_axt_unc( sia.ompMT, matAXT5, yr );
	str_res sr11 = test_int_axt_unc( sia.ompMT, matAXT6, yr );
	str_res sr12 = test_int_axt_unc( sia.ompMT, matAXT7, yr );
	str_res sr13 = test_int_axt_unc( sia.ompMT, matAXT8, yr );
	str_res sr14 = test_int_axt_unc( sia.ompMT, matAXT9, yr );
	str_res sr15 = test_int_axt_unc( sia.ompMT, matAXT10, yr );
	str_res sr16 = test_int_axt_unc( sia.ompMT, matAXT11, yr );
	str_res sr17 = test_int_axt_com_h1_bs64( sia.ompMT, matAXT12, yr );
	str_res sr18 = test_int_axt_com_h1_bs512( sia.ompMT, matAXT13, yr );
	str_res sr19 = test_int_axt_com( sia.ompMT, matAXT14, yr );
	str_res sr20 = test_int_axt_com( sia.ompMT, matAXT15, yr );
	str_res sr21 = test_int_axt_com( sia.ompMT, matAXT16, yr );
	str_res sr22 = test_int_axt_com( sia.ompMT, matAXT17, yr );
	str_res sr23 = test_int_axt_com( sia.ompMT, matAXT18, yr );
	str_res sr24 = test_int_axt_com( sia.ompMT, matAXT19, yr );
	str_res sr25 = test_int_axt_com( sia.ompMT, matAXT20, yr );
	// AXT format  ------------------------------------------------------------------------------------------------------------------
*/
	HDL; printf( "formats' data\n" ); HDL;
	printf( "%25s %20s %10s %20s\n", "format", "memory [Mbytes]", "occupancy", "convTime [s]" );
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd01.name, ( fd01.mfp * 1e-6 ), fd01.beta, fd01.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd02.name, ( fd02.mfp * 1e-6 ), fd02.beta, fd02.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd03.name, ( fd03.mfp * 1e-6 ), fd03.beta, fd03.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd04.name, ( fd04.mfp * 1e-6 ), fd04.beta, fd04.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd05.name, ( fd05.mfp * 1e-6 ), fd05.beta, fd05.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd06.name, ( fd06.mfp * 1e-6 ), fd06.beta, fd06.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd07.name, ( fd07.mfp * 1e-6 ), fd07.beta, fd07.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd08.name, ( fd08.mfp * 1e-6 ), fd08.beta, fd08.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd09.name, ( fd09.mfp * 1e-6 ), fd09.beta, fd09.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd10.name, ( fd10.mfp * 1e-6 ), fd10.beta, fd10.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd11.name, ( fd11.mfp * 1e-6 ), fd11.beta, fd11.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd12.name, ( fd12.mfp * 1e-6 ), fd12.beta, fd12.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd13.name, ( fd13.mfp * 1e-6 ), fd13.beta, fd13.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd14.name, ( fd14.mfp * 1e-6 ), fd14.beta, fd14.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd15.name, ( fd15.mfp * 1e-6 ), fd15.beta, fd15.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd16.name, ( fd16.mfp * 1e-6 ), fd16.beta, fd16.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd17.name, ( fd17.mfp * 1e-6 ), fd17.beta, fd17.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd18.name, ( fd18.mfp * 1e-6 ), fd18.beta, fd18.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd19.name, ( fd19.mfp * 1e-6 ), fd19.beta, fd19.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd20.name, ( fd20.mfp * 1e-6 ), fd20.beta, fd20.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd21.name, ( fd21.mfp * 1e-6 ), fd21.beta, fd21.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd22.name, ( fd22.mfp * 1e-6 ), fd22.beta, fd22.ct ); fflush(stdout);
//	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd23.name, ( fd23.mfp * 1e-6 ), fd23.beta, fd23.ct ); fflush(stdout);

	HDL; printf( "SpMV kernels' results\n" ); HDL;
	printf( "%25s %15s %8s %15s %13s %13s %10s\n", "kernel", "exeTime [s]", "Gflops", "ordTime [s]", "errAbs", "errRel", "rowInd" );
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr01.name, sr01.et, ( sr01.flops * 1e-9 ), sr01.ot, sr01.sErr.aErr, sr01.sErr.rErr, sr01.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr02.name, sr02.et, ( sr02.flops * 1e-9 ), sr02.ot, sr02.sErr.aErr, sr02.sErr.rErr, sr02.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr03.name, sr03.et, ( sr03.flops * 1e-9 ), sr03.ot, sr03.sErr.aErr, sr03.sErr.rErr, sr03.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr04.name, sr04.et, ( sr04.flops * 1e-9 ), sr04.ot, sr04.sErr.aErr, sr04.sErr.rErr, sr04.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr05.name, sr05.et, ( sr05.flops * 1e-9 ), sr05.ot, sr05.sErr.aErr, sr05.sErr.rErr, sr05.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr06.name, sr06.et, ( sr06.flops * 1e-9 ), sr06.ot, sr06.sErr.aErr, sr06.sErr.rErr, sr06.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr07.name, sr07.et, ( sr07.flops * 1e-9 ), sr07.ot, sr07.sErr.aErr, sr07.sErr.rErr, sr07.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr08.name, sr08.et, ( sr08.flops * 1e-9 ), sr08.ot, sr08.sErr.aErr, sr08.sErr.rErr, sr08.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr09.name, sr09.et, ( sr09.flops * 1e-9 ), sr09.ot, sr09.sErr.aErr, sr09.sErr.rErr, sr09.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr10.name, sr10.et, ( sr10.flops * 1e-9 ), sr10.ot, sr10.sErr.aErr, sr10.sErr.rErr, sr10.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr11.name, sr11.et, ( sr11.flops * 1e-9 ), sr11.ot, sr11.sErr.aErr, sr11.sErr.rErr, sr11.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr12.name, sr12.et, ( sr12.flops * 1e-9 ), sr12.ot, sr12.sErr.aErr, sr12.sErr.rErr, sr12.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr13.name, sr13.et, ( sr13.flops * 1e-9 ), sr13.ot, sr13.sErr.aErr, sr13.sErr.rErr, sr13.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr14.name, sr14.et, ( sr14.flops * 1e-9 ), sr14.ot, sr14.sErr.aErr, sr14.sErr.rErr, sr14.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr15.name, sr15.et, ( sr15.flops * 1e-9 ), sr15.ot, sr15.sErr.aErr, sr15.sErr.rErr, sr15.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr16.name, sr16.et, ( sr16.flops * 1e-9 ), sr16.ot, sr16.sErr.aErr, sr16.sErr.rErr, sr16.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr17.name, sr17.et, ( sr17.flops * 1e-9 ), sr17.ot, sr17.sErr.aErr, sr17.sErr.rErr, sr17.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr18.name, sr18.et, ( sr18.flops * 1e-9 ), sr18.ot, sr18.sErr.aErr, sr18.sErr.rErr, sr18.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr19.name, sr19.et, ( sr19.flops * 1e-9 ), sr19.ot, sr19.sErr.aErr, sr19.sErr.rErr, sr19.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr20.name, sr20.et, ( sr20.flops * 1e-9 ), sr20.ot, sr20.sErr.aErr, sr20.sErr.rErr, sr20.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr21.name, sr21.et, ( sr21.flops * 1e-9 ), sr21.ot, sr21.sErr.aErr, sr21.sErr.rErr, sr21.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr22.name, sr22.et, ( sr22.flops * 1e-9 ), sr22.ot, sr22.sErr.aErr, sr22.sErr.rErr, sr22.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr23.name, sr23.et, ( sr23.flops * 1e-9 ), sr23.ot, sr23.sErr.aErr, sr23.sErr.rErr, sr23.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr24.name, sr24.et, ( sr24.flops * 1e-9 ), sr24.ot, sr24.sErr.aErr, sr24.sErr.rErr, sr24.sErr.pos ); fflush(stdout);
//	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr25.name, sr25.et, ( sr25.flops * 1e-9 ), sr25.ot, sr25.sErr.aErr, sr25.sErr.rErr, sr25.sErr.pos ); fflush(stdout);

	return( EXIT_SUCCESS );
}


