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
# File: jobLauncher.sh
# code dated: 12-08-2019 (dd-mm-yyyy)
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
echo "File: jobLauncher.sh"
echo "(C) 2019, University of Santiago de Compostela"
echo "Author: Edoardo Coronado"
echo
echo "This program comes with ABSOLUTELY NO WARRANTY."
echo "This is free software, and you are welcome to redistribute it under"
echo "certain conditions; see README.md and LICENSE.txt for details."
echo






if [ "$#" -ne 1 ]; then
	echo "                                             "
	echo "     Usage: ./jobLauncher.sh <EXEC>          "
	echo "                                             "
	echo "          <EXEC>: name of the executable file"
	echo "                                             "
	exit
fi



HOST=$(hostname)
ROOT_DIR=${PWD}
MAT_SRC=${ROOT_DIR}/matrices
EXE=$1
MAT_LST_0="M_0_arrowHeadSymNnz100.csr"
MAT_LST00="M00_e001.csr"
MAT_LST01="M01_scircuit.bin"
MAT_LST02="M25_Serena.bin"
MAT_LST03="M00_e001.csr M03_e002.csr M09_e003.csr M16_e004.csr M22_e005.csr"
MAT_LST04="M00_e001.csr M01_scircuit.bin M02_mac_econ_fwd500.bin M03_e002.csr M04_mc2depi.bin M05_rma10.bin M06_cop20k_A.bin M07_webbase-1M.bin M08_shipsec1.bin M09_e003.csr M10_cant.bin"
MAT_LST05="M11_pdb1HYS.bin M12_hamrle3.bin M13_consph.bin M14_g3_circuit.bin M15_thermal2.bin M16_e004.csr M17_pwtk.bin M18_kkt_power.bin M19_memchip.bin M20_in-2004.bin"
MAT_LST06="M21_fullChip.bin M22_e005.csr M23_delaunay_n23.bin M24_circuit5M.bin M25_Serena.bin"
MAT_LST07="M00_e001.csr M01_scircuit.bin M02_mac_econ_fwd500.bin M03_e002.csr M04_mc2depi.bin M05_rma10.bin M06_cop20k_A.bin M07_webbase-1M.bin M08_shipsec1.bin M09_e003.csr M10_cant.bin M11_pdb1HYS.bin M12_hamrle3.bin M13_consph.bin M14_g3_circuit.bin M15_thermal2.bin M16_e004.csr M17_pwtk.bin M18_kkt_power.bin M19_memchip.bin M20_in-2004.bin M21_fullChip.bin M22_e005.csr M23_delaunay_n23.bin M24_circuit5M.bin M25_Serena.bin"
MAT_LST08="M06_cop20k_A.bin"
MAT_LST=${MAT_LST07}
SLEEP_TIME=30
KFSL=6300
KFSU=6400
CFSL=8800
CFSU=9300



if grep -q "kay" <<< "${HOST}"; then
	EXE_LOG="executionsLog[KAY].txt"
	echo " "                      >  ${EXE_LOG}
	echo "host: [KAY]"            >> ${EXE_LOG}
	echo "date: [$(date +"%F")]"  >> ${EXE_LOG}
	echo " "                      >> ${EXE_LOG}
	KAY_DIR="${ROOT_DIR}/KAY_[$(date +"%F")]"
	OUT_DIR=${KAY_DIR}/results
	SHL_DIR=${KAY_DIR}/launchers
	STA_DIR=${KAY_DIR}/statistics
	if [ ! -d ${KAY_DIR} ]; then
		mkdir ${KAY_DIR}
		mkdir ${OUT_DIR}
		mkdir ${SHL_DIR}
		mkdir ${STA_DIR}
	else
		for MAT in ${MAT_LST}; do
			M="${MAT:0:3}"
			rm -f ${OUT_DIR}/${M}*
			rm -f ${SHL_DIR}/${M}*
			rm -f ${STA_DIR}/${M}*
		done
	fi
	JOB_TIME="0:20:00"
	JOB_ACCOUNT="hpce3ic5"
fi



if grep -q "indy2" <<< "${HOST}"; then
	EXE_LOG="executionsLog[CIRRUS].txt"
	echo " "                     >  ${EXE_LOG}
	echo "host: [CIRRUS]"        >> ${EXE_LOG}
	echo "date: [$(date +"%F")]" >> ${EXE_LOG}
	echo " "                     >> ${EXE_LOG}
	CIRRUS_DIR="${ROOT_DIR}/CIRRUS_[$(date +"%F")]"
	STA_DIR=${CIRRUS_DIR}/statistics
	LCH_DIR=${CIRRUS_DIR}/launchers
	ERR_DIR=${CIRRUS_DIR}/errors
	RES_DIR=${CIRRUS_DIR}/results
	if [ ! -d ${CIRRUS_DIR} ]; then
		mkdir ${CIRRUS_DIR}
		mkdir ${ERR_DIR}
		mkdir ${LCH_DIR}
		mkdir ${RES_DIR}
		mkdir ${STA_DIR}
	else
		for MAT in ${MAT_LST}; do
			M="${MAT:0:3}"
			rm -f ${ERR_DIR}/${M}*
			rm -f ${LCH_DIR}/${M}*
			rm -f ${RES_DIR}/${M}*
			rm -f ${STA_DIR}/${M}*
		done
	fi
	JOB_TIME="0:20:00"
	JOB_ACCOUNT="dc010-edoardo"
fi



(
	rm -f *.bin
	rm -f *.csr
	for F in ${MAT_LST}; do
		FILE=${MAT_SRC}/${F}
		ln -s ${FILE} ${ROOT_DIR}/${F}
	done
)



for MAT in ${MAT_LST}; do
	MAT_NAME="${MAT:0:${#MAT}-4}"
	if [ "${MAT}" == "M23_delaunay_n23.bin" ] || [ "${MAT}" == "M24_circuit5M.bin" ]; then
		BLOCK_SIZE="0032 0064 0128 0256 0512"
	else
		BLOCK_SIZE="0032 0064 0128 0256 0512 1024"
	fi
	if grep -q "kay" <<< "${HOST}" ; then
		echo "launching jobs with matrix: ${MAT_NAME}" >> ${EXE_LOG}
		for BS in ${BLOCK_SIZE}; do
			JOB_NAME="${MAT:0:3}_BS${BS}_KAY"
			SH_FILE="${JOB_NAME}.sh"
			echo "#!/bin/sh"                        >  ${SH_FILE}
			echo "#SBATCH --job-name=${JOB_NAME}"   >> ${SH_FILE}
			echo "#SBATCH --partition=GpuQ"         >> ${SH_FILE}
			echo "#SBATCH --account=${JOB_ACCOUNT}" >> ${SH_FILE}
			echo "#SBATCH --nodes=1"                >> ${SH_FILE}
			echo "#SBATCH --time=${JOB_TIME}"       >> ${SH_FILE}
			echo "module load intel/2017u8"         >> ${SH_FILE}
			echo "module load cuda/9.2"             >> ${SH_FILE}
			echo "export OMP_WAIT_POLICY=active"    >> ${SH_FILE}
			echo "export OMP_DYNAMIC=false"         >> ${SH_FILE}
			echo "export OMP_PROC_BIND=true"        >> ${SH_FILE}
			echo "export OMP_NUM_THREADS=40"        >> ${SH_FILE}
			echo "export MKL_NUM_THREADS=1"         >> ${SH_FILE}
			echo "./${EXE} ${BS} ${MAT}"            >> ${SH_FILE}
			chmod 755 ${SH_FILE}
			JOB=$(sbatch ${SH_FILE})
			JOB=${JOB:${#JOB}-7:${#JOB}}
			STAT=$(squeue --jobs=${JOB})
			JOB_ID=${STAT:97:6}
			STAT=${STAT:132:2}
			SLURM_FILE="slurm-${JOB_ID}.out"
			if [ "${STAT}" == "PD" ]; then
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} QUEUED   @ $(date +"%T")" >> ${EXE_LOG}
			fi
			while [ "${STAT}" == "PD" ]; do
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} PENDING  @ $(date +"%T")" >> ${EXE_LOG}
				sleep ${SLEEP_TIME}
				STAT=$(squeue --jobs=${JOB})
				STAT=${STAT:132:2}
			done
			echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} RUNNING  @ $(date +"%T")" >> ${EXE_LOG}
			while [ "${STAT}" == " R" ]; do
				sleep ${SLEEP_TIME}
				STAT=$(squeue --jobs=${JOB})
				STAT=${STAT:132:2}
			done
			echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} FINISHED @ $(date +"%T")" >> ${EXE_LOG}
			while [ ! -f ${SLURM_FILE} ]; do
				printf "%s\r" "waiting for ${SLURM_FILE}"                                      >> ${EXE_LOG}
			done
			SLURM_FSIZE=$(stat -c%s "${SLURM_FILE}")
			printf -v SIZE '%d' ${SLURM_FSIZE} 2>/dev/null
			if [ ${SIZE} -lt ${KFSL} ]; then
				sleep 7
				SLURM_FSIZE=$(stat -c%s "${SLURM_FILE}")
				printf -v SIZE '%d' ${SLURM_FSIZE} 2>/dev/null
			fi
			if [ ${SIZE} -gt ${KFSL} ] && [ ${SIZE} -lt ${KFSU} ]; then
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} CHECKED  @ $(date +"%T")  [CORRECT] ${SIZE}" >> ${EXE_LOG}
			else
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} CHECKED  @ $(date +"%T")  [WRONG]   ${SIZE}" >> ${EXE_LOG}
			fi
			mv "${MAT_NAME}.sta" ${STA_DIR}/"${MAT_NAME}_KAY.sta"
			mv ${SH_FILE} ${SHL_DIR}
			mv ${SLURM_FILE} ${OUT_DIR}/"${JOB_NAME}.txt"
		done
	fi
	if grep -q "indy2" <<< "${HOST}" ; then
		echo "launching jobs with matrix: ${MAT_NAME}" >>  ${EXE_LOG}
		for BS in ${BLOCK_SIZE}; do
			JOB_NAME="${MAT:0:3}_BS${BS}_CIRRUS"
			PBS_FILE="${JOB_NAME}.pbs"
			echo "#!/bin/bash"                       >  ${PBS_FILE}
			echo "#PBS -N ${JOB_NAME}"               >> ${PBS_FILE}
			echo "#PBS -q gpu"                       >> ${PBS_FILE}
			echo "#PBS -A ${JOB_ACCOUNT}"            >> ${PBS_FILE}
			echo "#PBS -l select=1:ncpus=40:ngpus=4" >> ${PBS_FILE}
			echo "#PBS -l walltime=${JOB_TIME}"      >> ${PBS_FILE}
			echo "#PBS -l place=scatter:excl"        >> ${PBS_FILE}
			echo "module load cuda"                  >> ${PBS_FILE}
			echo "module load gcc/6.3.0"             >> ${PBS_FILE}
			echo "module load intel-compilers-17"    >> ${PBS_FILE}
			echo "module load openmpi"               >> ${PBS_FILE}
			echo "cd \$PBS_O_WORKDIR"                >> ${PBS_FILE}
			echo "export OMP_WAIT_POLICY=active"     >> ${PBS_FILE}
			echo "export OMP_DYNAMIC=false"          >> ${PBS_FILE}
			echo "export OMP_PROC_BIND=true"         >> ${PBS_FILE}
			echo "export OMP_NUM_THREADS=40"         >> ${PBS_FILE}
			echo "export MKL_NUM_THREADS=1"          >> ${PBS_FILE}
			echo "./${EXE} ${BS} ${MAT}"             >> ${PBS_FILE}
			chmod 755 ${PBS_FILE}
			JOB=$(qsub ${PBS_FILE})
			JOB_ID="${JOB:0:7}"
			STAT=$(qstat -x ${JOB})
			STAT=${STAT:202:1}
			RFILE_NAME="${JOB_NAME}.o${JOB_ID}"
			if [ "${STAT}" == "Q" ]; then
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} QUEUED   @ $(date +"%T")" >> ${EXE_LOG}
			fi
			while [ "${STAT}" != "F" ]; do
				sleep ${SLEEP_TIME}
				STAT=$(qstat -x ${JOB})
				STAT=${STAT:202:1}
			done
			if [ "${STAT}" == "F" ]; then
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} FINISHED @ $(date +"%T")" >> ${EXE_LOG}
			fi
			while [ ! -f ${RFILE_NAME} ]; do
				printf "%s\r" "waiting for ${RFILE_NAME}"                                           >> ${EXE_LOG}
			done
			RFILE_SIZE=$(stat -c%s "${RFILE_NAME}")
			printf -v SIZE '%d' ${RFILE_SIZE} 2>/dev/null
			if [ ${SIZE} -lt ${CFSL} ]; then
				sleep 7
				RFILE_SIZE=$(stat -c%s "${RFILE_NAME}")
				printf -v SIZE '%d' ${RFILE_SIZE} 2>/dev/null
			fi
			if [ ${SIZE} -gt ${CFSL} ] && [ ${SIZE} -lt ${CFSU} ]; then
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} CHECKED  @ $(date +"%T")  [CORRECT] ${SIZE}" >> ${EXE_LOG}
			else
				echo "[MEASURING] JOB_ID: ${JOB_ID} JOB_NAME: ${JOB_NAME} CHECKED  @ $(date +"%T")  [WRONG]   ${SIZE}" >> ${EXE_LOG}
			fi
			mv "${MAT_NAME}.sta" ${STA_DIR}/"${MAT_NAME}_CIRRUS.sta"
			mv ${PBS_FILE} ${LCH_DIR}
			mv ${JOB_NAME}.e${JOB_ID} ${ERR_DIR}/"${JOB_NAME}.err"
			mv ${JOB_NAME}.o${JOB_ID} ${RES_DIR}/"${JOB_NAME}.txt"
		done
	fi
	echo " " >> ${EXE_LOG}
	rm -f ${mat}
done



if grep -q "kay" <<< "${HOST}"; then
	echo " " >> ${EXE_LOG}
	mv ${EXE_LOG} ${KAY_DIR}
fi



if grep -q "indy2" <<< "${HOST}"; then
	echo " " >> ${EXE_LOG}
	mv ${EXE_LOG} ${CIRRUS_DIR}
fi


(
	rm -f *.bin
	rm -f *.csr
)



