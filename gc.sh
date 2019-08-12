#!/bin/bash



if [ "$#" -ne 2 ]; then
	echo "                                                        "
	echo "     Usage: ./compile.sh <NUM_ITE> <OPENMP>             "
	echo "                                                        "
	echo "          <NUM_ITE>: number of iterations to run kernels"
	echo "          <OPENMP>:  0-Do NOT use OpenMP    1-Use OpenMP"
	echo "                                                        "
	exit
fi



HOST=$(hostname)
CC=nvcc
SRC=gpuSpmv_main.cu
EXE=gpuSpmv
LOG=log.txt
OPT=""
INC=""
LIB="-lcusparse"



echo "rm -f ${EXE}"
rm -f ${EXE}



if grep -q "kay" <<< "${HOST}" ; then
	echo "compiling on KAY..."
	if [ "$2" == "1" ]; then
		echo "openmp selected..."
		OPT="${OPT} -ccbin icpc"
		OPT="${OPT} -Xcompiler -qopenmp"
		OPT="${OPT} -D _OMP_"
	fi
	OPT="${OPT} -D DEVICE=1 -arch sm_70 -D _KAY_"
fi



if grep -q "indy2" <<< "${HOST}" ; then
	echo "compiling on CIRRUS..."
	if [ "$2" == "1" ]; then
		echo "openmp selected..."
		OPT="${OPT} -ccbin icpc"
		OPT="${OPT} -Xcompiler -qopenmp"
		OPT="${OPT} -D _OMP_"
	fi
	OPT="${OPT} -D DEVICE=3 -arch sm_70 -D _CIRRUS_"
fi



if grep -q "ctgpgpu2" <<< "${HOST}" ; then
	echo "compiling on CTGPGPU2..."
	if [ "$2" == "1" ]; then
		echo "openmp selected..."
		OPT="${OPT} -D _OMP_"
		LIB="${LIB} -lopenmp"
	fi
	INC="${INC} -I/usr/local/cuda-9.2/targets/x86_64-linux/include/"
	OPT="${OPT} -D DEVICE=0 -arch sm_30 -D _CTGPGPU2_"
fi



OPT="${OPT} -D FP_TYPE=FP_DOUBLE -D NUM_ITE=$1"



echo "${CC} ${INC} ${OPT} ${SRC} ${LIB} -o ${EXE}"
${CC} ${INC} ${OPT} ${SRC} ${LIB} -o ${EXE} 1>${LOG} 2>&1



if [ "$?" == "0" ]
then
echo "Compilation successfully ... :)"
	rm -f ${LOG}
else
	echo "Compilation error ... :("
	echo "Log file content:"
	echo
	cat ${LOG}
	echo
fi


