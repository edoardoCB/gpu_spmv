// ┌────────────────────────────────┐
// │program: gpuSpmv_main.cu        │
// │author: Edoardo Coronado        │
// │date: 24-07-2019 (dd-mm-yyyy)   │
// ╰────────────────────────────────┘



#include "gpuSpmv_header.h"



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
	str_formatData fd01 = getFormatDataCSR( matCSR );
	// init vectors to perform SpMV multiplication and check errors (spM * vr = yr)
	FPT * vr = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( vr );
	initVec( matCSR.nrows, vr );
	FPT * yr = (FPT *) calloc( matCSR.nrows,  sizeof(FPT) ); TEST_POINTER( yr );
	cf_CSR( matCSR, vr, yr );
	// test CSR kernels
	str_res sr01 = test_cf_CSR( matCSR, vr, yr );
	str_res sr02 = test_gk_CSR( sia.cudaBlockSize, matCSR, vr, yr );
	str_res sr03 = test_gcu_CSR( matCSR, vr, yr );
	// CSR format  ------------------------------------------------------------------------------------------------------------------

	// AXC format  ------------------------------------------------------------------------------------------------------------------
	str_matAXC matAXC; str_formatData fd02 = getFormatDataAXC( sia.ompMaxThreads, matCSR, vr, &matAXC );
	str_res sr04 = test_gk_AXC( sia.cudaBlockSize, matAXC, yr );
	// AXC format  ------------------------------------------------------------------------------------------------------------------

	// K1 format   ------------------------------------------------------------------------------------------------------------------
	str_matK1 matK1; str_formatData fd03 = getFormatDataK1( sia.cudaBlockSize, matCSR, vr, &matK1 );
	str_res sr05 = test_gk_K1( sia.cudaBlockSize, matK1, vr, yr );
	// K1 format   ------------------------------------------------------------------------------------------------------------------

	// AXT format  ------------------------------------------------------------------------------------------------------------------
	str_matAXT matAXT1; str_formatData fd04 = getFormatDataAXT( sia.ompMaxThreads, sia.cudaBlockSize, 32,  1, "NOC", matCSR, vr, &matAXT1 );
	str_res sr06 = test_gk_AXT_NOC_H1( sia.cudaBlockSize, matAXT1, yr );

	str_matAXT matAXT2; str_formatData fd05 = getFormatDataAXT( sia.ompMaxThreads, sia.cudaBlockSize, 32,  4, "NOC", matCSR, vr, &matAXT2 );
	str_res sr07 = test_gk_AXT_NOC( sia.cudaBlockSize, matAXT2, yr );

	str_matAXT matAXT3; str_formatData fd06 = getFormatDataAXT( sia.ompMaxThreads, sia.cudaBlockSize, 32,  1, "COM", matCSR, vr, &matAXT3 );
	str_res sr08 = test_gk_AXT_COM_H1( sia.cudaBlockSize, matAXT3, yr );



/*
	str_matAXTC matAXTC3; str_formatData fd06 = getFormatDataAXTC( sia.ompMaxThreads, thw, 32, "NOC", matCSR, vr, &matAXTC3 );
	str_matAXTC matAXTC4; str_formatData fd07 = getFormatDataAXTC( sia.ompMaxThreads, thw,  1, "CMP", matCSR, vr, &matAXTC4 );
	str_matAXTC matAXTC5; str_formatData fd08 = getFormatDataAXTC( sia.ompMaxThreads, thw,  4, "CMP", matCSR, vr, &matAXTC5 );
	str_matAXTC matAXTC6; str_formatData fd09 = getFormatDataAXTC( sia.ompMaxThreads, thw, 32, "CMP", matCSR, vr, &matAXTC6 );
	const UIN avg = matCSR.nnz / matCSR.nrows;
	if (  (avg >4) && (avg <= 32)  )
	{
		str_matAXTC matAXTC7; str_formatData fd10 = getFormatDataAXTC( sia.ompMaxThreads, thw, avg, "NOC", matCSR, vr, &matAXTC7 );
		str_matAXTC matAXTC8; str_formatData fd11 = getFormatDataAXTC( sia.ompMaxThreads, thw, avg, "CMP", matCSR, vr, &matAXTC8 );
	}
*/
	// AXTC format  -----------------------------------------------------------------------------------------------------------------

	HDL; printf( "formats' data\n" ); HDL;
	printf( "%18s %20s %10s %20s\n", "format", "memory [Mbytes]", "occupancy", "conversion time [s]" );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd01.name, ( fd01.mfp * 1e-6 ), fd01.beta, fd01.ct );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd02.name, ( fd02.mfp * 1e-6 ), fd02.beta, fd02.ct );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd03.name, ( fd03.mfp * 1e-6 ), fd03.beta, fd03.ct );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd04.name, ( fd04.mfp * 1e-6 ), fd04.beta, fd04.ct );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd05.name, ( fd05.mfp * 1e-6 ), fd05.beta, fd05.ct );
	printf( "%18s %20.2lf %10.2lf %20.6lf\n", fd06.name, ( fd06.mfp * 1e-6 ), fd06.beta, fd06.ct );

	HDL; printf( "kernels' results\n" ); HDL;
	printf( "%20s %15s %8s %13s %13s %10s\n", "kernel", "time [s]", "Gflops", "aErr||.||inf", "rErr||.||inf", "rowInd" );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr01.name, sr01.et, ( sr01.flops * 1e-9 ), sr01.sErr.aErr, sr01.sErr.rErr, sr01.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr02.name, sr02.et, ( sr02.flops * 1e-9 ), sr02.sErr.aErr, sr02.sErr.rErr, sr02.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr03.name, sr03.et, ( sr03.flops * 1e-9 ), sr03.sErr.aErr, sr03.sErr.rErr, sr03.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr04.name, sr04.et, ( sr04.flops * 1e-9 ), sr04.sErr.aErr, sr04.sErr.rErr, sr04.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr05.name, sr05.et, ( sr05.flops * 1e-9 ), sr05.sErr.aErr, sr05.sErr.rErr, sr05.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr06.name, sr06.et, ( sr06.flops * 1e-9 ), sr06.sErr.aErr, sr06.sErr.rErr, sr06.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr07.name, sr07.et, ( sr07.flops * 1e-9 ), sr07.sErr.aErr, sr07.sErr.rErr, sr07.sErr.pos );
	printf( "%20s %15.7lf %8.3lf %11.3le %13.3le %12d\n", sr08.name, sr08.et, ( sr08.flops * 1e-9 ), sr08.sErr.aErr, sr08.sErr.rErr, sr08.sErr.pos );

	return( EXIT_SUCCESS );
}



