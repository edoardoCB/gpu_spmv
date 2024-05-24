#!/bin/bash
#
#            gpu_spmv
# (SpMV product on NVIDIA's GPU)
#
# (C) 2019, University of Santiago de Compostela
#
# Author: Edoardo Coronado <eecb76@hotmail.com>
#
# Program: gpu_spmv
# File: gc.sh
# code dated: 05-06-2019 (dd-mm-yyyy)
#
#	gpu_spmv is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
////////////////////////////////////////////////////////////////

echo
echo "gpu_spmv - SpMV product on NVIDIA's GPU"
echo "File: gc.sh"
echo "(C) 2019, University of Santiago de Compostela"
echo "Author: Edoardo Coronado"
echo
echo "This program comes with ABSOLUTELY NO WARRANTY."
echo "This is free software, and you are welcome to redistribute it under"
echo "certain conditions; see README.md and LICENSE.txt for details."
echo



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
SRC=cudaSpmv.cu
EXE=cudaSpmv
LOG=log.txt
OPT=""
INC=""
LIB="-lcusparse"
HASH_FILE="HASH.txt"
HASH=$(git show-ref --hash)
echo ${HASH} > ${HASH_FILE}



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



if [ "$?" == "0" ]; then
	echo "Compilation successfully ... :)"
	rm -f ${LOG}
	echo "moving files to working directory... "
	mv HASH.txt ./..
	mv ${EXE} ./..
else
	echo "Compilation error ... :("
	echo "Log file content:"
	echo
	cat ${LOG}
	echo
fi



