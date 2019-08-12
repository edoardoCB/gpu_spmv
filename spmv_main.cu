// ┌────────────────────────────────┐
// │program: main_spmv.cu           │
// │author: Edoardo Coronado        │
// │date: 05-06-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#include "spmv_header.h"



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

	// CSR format  ---------------------------------------------------------------------------------------------
	// read matrix in CSR format
	str_matCSR matCSR = matrixReading( sia.matFileName );
	// print matrix's statistics
	printMatrixStats( sia.matFileName, &matCSR );
	// get memory footprint, occupancy (beta) and conversion time
	str_formatData fd01 = getFormatDataCSR( matCSR );
	// init vectors to perform SpMV multiplication and check errors (spM * vr = yr)
	FPT * vr = (FPT *) malloc( matCSR.nrows * sizeof(FPT) ); TEST_POINTER( vr );
	initVec( matCSR.nrows, vr );
	FPT * yr = (FPT *) malloc( matCSR.nrows * sizeof(FPT) ); TEST_POINTER( yr );
	cpuSpmvCSR( matCSR, vr, yr );
	// CSR format  ---------------------------------------------------------------------------------------------

	// AXC format  ---------------------------------------------------------------------------------------------
	str_matAXCv0 matAXCv0; str_formatData fd02 = getFormatDataAXCv0( sia.ompMaxThreads, matCSR, vr, &matAXCv0 );
	str_matAXCv1 matAXCv1; str_formatData fd03 = getFormatDataAXCv1( sia.ompMaxThreads, matCSR, vr, &matAXCv1 );
	// AXC format  ---------------------------------------------------------------------------------------------

	// AXB format  ---------------------------------------------------------------------------------------------
	str_matAXBv0 matAXBv0; str_formatData fd04 = getFormatDataAXBv0( matCSR, vr, &matAXBv0 );
	str_matAXBv1 matAXBv1; str_formatData fd05 = getFormatDataAXBv1( sia.cudaBlockSize, matCSR, vr, &matAXBv1 );
	// AXB format  ---------------------------------------------------------------------------------------------

	// K1 format  ----------------------------------------------------------------------------------------------
	str_matK1v0 matK1v0;   str_formatData fd06 = getFormatDataK1v0( sia.cudaBlockSize, matCSR, vr, &matK1v0 );
	// K1 format  ----------------------------------------------------------------------------------------------

	// AXT format  ---------------------------------------------------------------------------------------------
	const UIN tileHW = 16;
	str_matAXTv0 matAXTv0H4;  str_formatData fd07 = getFormatDataAXTv0( sia.ompMaxThreads, tileHW,  4, matCSR, vr, &matAXTv0H4 );
	str_matAXTv0 matAXTv0H6;  str_formatData fd08 = getFormatDataAXTv0( sia.ompMaxThreads, tileHW,  6, matCSR, vr, &matAXTv0H6 );
	str_matAXTv0 matAXTv0H8;  str_formatData fd09 = getFormatDataAXTv0( sia.ompMaxThreads, tileHW,  8, matCSR, vr, &matAXTv0H8 );
	UIN avg = matCSR.nnz / matCSR.nrows, h = 0;
	if  (avg <=  4)                  h = 4;
	if ((avg >   4) && (avg <=  32)) h = avg;
	if ((avg >  32) && (avg <= 256)) h = 32;
	if  (avg > 256)                  h = 4;
	str_matAXTv1 matAXTv1H4;  str_formatData fd10 = getFormatDataAXTv1( sia.ompMaxThreads, tileHW, 4, matCSR,  vr, &matAXTv1H4 );
	str_matAXTv1 matAXTv1HH;  str_formatData fd11 = getFormatDataAXTv1( sia.ompMaxThreads, tileHW, h, matCSR,  vr, &matAXTv1HH );
	str_matAXTv1 matAXTv1H16; str_formatData fd12 = getFormatDataAXTv1( sia.ompMaxThreads, tileHW, 16, matCSR, vr, &matAXTv1H16 );
	str_matAXTv1 matAXTv1H32; str_formatData fd13 = getFormatDataAXTv1( sia.ompMaxThreads, tileHW, 32, matCSR, vr, &matAXTv1H32 );
	// AXT format  ---------------------------------------------------------------------------------------------


	int nt = 555, mnt = 555, tid = 555, np = 555, ip = 555;
	#pragma omp parallel default(shared) private(nt,mnt,tid,np,ip) num_threads(sia.ompMaxThreads) if(_OMP_)
	{
		 nt = omp_get_num_threads();
		mnt = omp_get_max_threads();
		tid = omp_get_thread_num();
		 np = omp_get_num_procs();
		 ip = omp_in_parallel();
		printf( "nt:%2d, mnt:%2d, tid:%2d, np:%2d, ip:%d\n", nt, mnt, tid, np, ip );
	}


	str_matAXTC matAXTC1; str_formatData fd15 = getFormatDataAXTC( sia.ompMaxThreads, 16, 1, "NOComp", matCSR, vr, &matAXTC1 );
	//str_matAXTC matAXTC2; str_formatData fd16 = getFormatDataAXTC( sia.ompMaxThreads, 16, 4, "NOComp", matCSR, vr, &matAXTC2 );
	//str_matAXTC matAXTC3; str_formatData fd17 = getFormatDataAXTC( sia.ompMaxThreads, 16, 1, "comp",   matCSR, vr, &matAXTC3 );
	//str_matAXTC matAXTC4; str_formatData fd18 = getFormatDataAXTC( sia.ompMaxThreads, 16, 4, "comp",   matCSR, vr, &matAXTC4 );


	// print formats' data
	HDL; printf( "formats' data\n" ); HDL;
	printf( "%12s %20s %10s %20s\n", "format", "memory [Mbytes]", "occupancy", "conversion time [s]" );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd01.name, ( fd01.mfp * 1e-6 ), fd01.beta, fd01.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd02.name, ( fd02.mfp * 1e-6 ), fd02.beta, fd02.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd03.name, ( fd03.mfp * 1e-6 ), fd03.beta, fd03.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd04.name, ( fd04.mfp * 1e-6 ), fd04.beta, fd04.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd05.name, ( fd05.mfp * 1e-6 ), fd05.beta, fd05.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd06.name, ( fd06.mfp * 1e-6 ), fd06.beta, fd06.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd07.name, ( fd07.mfp * 1e-6 ), fd07.beta, fd07.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd08.name, ( fd08.mfp * 1e-6 ), fd08.beta, fd08.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd09.name, ( fd09.mfp * 1e-6 ), fd09.beta, fd09.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd10.name, ( fd10.mfp * 1e-6 ), fd10.beta, fd10.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd11.name, ( fd11.mfp * 1e-6 ), fd11.beta, fd11.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd12.name, ( fd12.mfp * 1e-6 ), fd12.beta, fd12.ct );
	printf( "%12s %20.2lf %10.2lf %20.6lf\n", fd13.name, ( fd13.mfp * 1e-6 ), fd13.beta, fd13.ct );

	// CSR kernels ---------------------------------------------------------------------------------------------
	str_res sr01 = testCpuSpmvCSR( matCSR, vr, yr );
	str_res sr02 = testGpuSpmvCSR( sia.cudaBlockSize, matCSR, vr, yr );
	str_res sr03 = testGpuCusparseSpmvCSR( matCSR, vr, yr );
	// print CSR kernels' results
	HDL; printf( "kernels' results\n" ); HDL;
	printf( "%20s %15s %8s %13s %13s %10s\n", "kernel", "time [s]", "Gflops", "aErr||.||inf", "rErr||.||inf", "rowInd" );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr01.name, sr01.et, ( sr01.flops * 1e-9 ), sr01.sErr.aErr, sr01.sErr.rErr, sr01.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr02.name, sr02.et, ( sr02.flops * 1e-9 ), sr02.sErr.aErr, sr02.sErr.rErr, sr02.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr03.name, sr03.et, ( sr03.flops * 1e-9 ), sr03.sErr.aErr, sr03.sErr.rErr, sr03.sErr.pos );
	// CSR kernels ---------------------------------------------------------------------------------------------

	// AXC kernels ---------------------------------------------------------------------------------------------
	str_res sr04 = testGpuSpmvAXCv0( sia.cudaBlockSize, matAXCv0, yr );
	str_res sr05 = testGpuSpmvAXCv1_0( sia.cudaBlockSize, matAXCv1, yr );
	str_res sr06 = testGpuSpmvAXCv1_1( sia.cudaBlockSize, matAXCv1, yr );
	str_res sr07 = testGpuSpmvAXCv1_2( sia.cudaBlockSize, matAXCv1, yr );
	str_res sr08 = testGpuSpmvAXCv1_3( sia.cudaBlockSize, matAXCv1, yr );
	str_res sr09 = testGpuSpmvAXCv1_4( sia.cudaBlockSize, matAXCv1, yr );
	// print AXC kernels' results
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr04.name, sr04.et, ( sr04.flops * 1e-9 ), sr04.sErr.aErr, sr04.sErr.rErr, sr04.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr05.name, sr05.et, ( sr05.flops * 1e-9 ), sr05.sErr.aErr, sr05.sErr.rErr, sr05.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr06.name, sr06.et, ( sr06.flops * 1e-9 ), sr06.sErr.aErr, sr06.sErr.rErr, sr06.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr07.name, sr07.et, ( sr07.flops * 1e-9 ), sr07.sErr.aErr, sr07.sErr.rErr, sr07.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr08.name, sr08.et, ( sr08.flops * 1e-9 ), sr08.sErr.aErr, sr08.sErr.rErr, sr08.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr09.name, sr09.et, ( sr09.flops * 1e-9 ), sr09.sErr.aErr, sr09.sErr.rErr, sr09.sErr.pos );
	// AXC kernels ---------------------------------------------------------------------------------------------

	// AXB kernels ---------------------------------------------------------------------------------------------
	str_res sr10 = testGpuSpmvAXBv0_0( sia.cudaBlockSize, matAXBv0, yr );
	str_res sr11 = testGpuSpmvAXBv0_1( sia.cudaBlockSize, matAXBv0, yr );
	str_res sr12 = testGpuSpmvAXBv0_2( sia.cudaBlockSize, matAXBv0, yr );
	str_res sr13 = testGpuSpmvAXBv0_3( sia.cudaBlockSize, matAXBv0, yr );
	str_res sr14 = testGpuSpmvAXBv1_0( sia.cudaBlockSize, matAXBv1, yr );
	// print AXB kernels' results
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr10.name, sr10.et, ( sr10.flops * 1e-9 ), sr10.sErr.aErr, sr10.sErr.rErr, sr10.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr11.name, sr11.et, ( sr11.flops * 1e-9 ), sr11.sErr.aErr, sr11.sErr.rErr, sr11.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr12.name, sr12.et, ( sr12.flops * 1e-9 ), sr12.sErr.aErr, sr12.sErr.rErr, sr12.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr13.name, sr13.et, ( sr13.flops * 1e-9 ), sr13.sErr.aErr, sr13.sErr.rErr, sr13.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr14.name, sr14.et, ( sr14.flops * 1e-9 ), sr14.sErr.aErr, sr14.sErr.rErr, sr14.sErr.pos );
	// AXB kernels ---------------------------------------------------------------------------------------------

	// K1 kernel  ----------------------------------------------------------------------------------------------
	str_res sr15 = testGpuSpmvK1v0_0( sia.cudaBlockSize, matK1v0, vr, yr );
	// print K1 kernel's results
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr15.name, sr15.et, ( sr15.flops * 1e-9 ), sr15.sErr.aErr, sr15.sErr.rErr, sr15.sErr.pos );
	// K1 kernel  ----------------------------------------------------------------------------------------------

	// AXT format  ---------------------------------------------------------------------------------------------
	str_res sr16 = testGpuSpmvAXTv0_0( sia.cudaBlockSize, matAXTv0H4, yr );
	str_res sr17 = testGpuSpmvAXTv0_0( sia.cudaBlockSize, matAXTv0H6, yr );
	str_res sr18 = testGpuSpmvAXTv0_0( sia.cudaBlockSize, matAXTv0H8, yr );
	// print AXTv0_0 kernels' result
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr16.name, sr16.et, ( sr16.flops * 1e-9 ), sr16.sErr.aErr, sr16.sErr.rErr, sr16.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr17.name, sr17.et, ( sr17.flops * 1e-9 ), sr17.sErr.aErr, sr17.sErr.rErr, sr17.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr18.name, sr18.et, ( sr18.flops * 1e-9 ), sr18.sErr.aErr, sr18.sErr.rErr, sr18.sErr.pos );
	str_res sr19 = testGpuSpmvAXTv0_1( sia.cudaBlockSize, matAXTv0H4, yr );
	str_res sr20 = testGpuSpmvAXTv0_1( sia.cudaBlockSize, matAXTv0H6, yr );
	str_res sr21 = testGpuSpmvAXTv0_1( sia.cudaBlockSize, matAXTv0H8, yr );
	// print AXTv0_1 kernels' result
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr19.name, sr19.et, ( sr19.flops * 1e-9 ), sr19.sErr.aErr, sr19.sErr.rErr, sr19.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr20.name, sr20.et, ( sr20.flops * 1e-9 ), sr20.sErr.aErr, sr20.sErr.rErr, sr20.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr21.name, sr21.et, ( sr21.flops * 1e-9 ), sr21.sErr.aErr, sr21.sErr.rErr, sr21.sErr.pos );


	str_res sr22 = testGpuSpmvAXTv1_0( sia.cudaBlockSize, matAXTv1H4, yr );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr22.name, sr22.et, ( sr22.flops * 1e-9 ), sr22.sErr.aErr, sr22.sErr.rErr, sr22.sErr.pos );
	str_res sr23 = testGpuSpmvAXTv1_0( sia.cudaBlockSize, matAXTv1HH, yr );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr23.name, sr23.et, ( sr23.flops * 1e-9 ), sr23.sErr.aErr, sr23.sErr.rErr, sr23.sErr.pos );
	str_res sr24 = testGpuSpmvAXTv1_0( sia.cudaBlockSize, matAXTv1H16, yr );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr24.name, sr24.et, ( sr24.flops * 1e-9 ), sr24.sErr.aErr, sr24.sErr.rErr, sr24.sErr.pos );
	str_res sr25 = testGpuSpmvAXTv1_0( sia.cudaBlockSize, matAXTv1H32, yr );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr25.name, sr25.et, ( sr25.flops * 1e-9 ), sr25.sErr.aErr, sr25.sErr.rErr, sr25.sErr.pos );




	// print AXTv1_0 kernels' result
	//printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr22.name, sr22.et, ( sr22.flops * 1e-9 ), sr22.sErr.aErr, sr22.sErr.rErr, sr22.sErr.pos );
	//printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr23.name, sr23.et, ( sr23.flops * 1e-9 ), sr23.sErr.aErr, sr23.sErr.rErr, sr23.sErr.pos );
	//printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr24.name, sr24.et, ( sr24.flops * 1e-9 ), sr24.sErr.aErr, sr24.sErr.rErr, sr24.sErr.pos );
	//printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr25.name, sr25.et, ( sr25.flops * 1e-9 ), sr25.sErr.aErr, sr25.sErr.rErr, sr25.sErr.pos );
	// AXT kernels ---------------------------------------------------------------------------------------------

	return( EXIT_SUCCESS );
}



