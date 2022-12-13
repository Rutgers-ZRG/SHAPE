#!/bin/bash
for i in {1..200} ; do grep -l "Traceback" ${i}/slurm-* ; done 1> error_log 2> /dev/null
awk -F '/' '{print $1 }' error_log > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd ./$line ; rm slurm-* opt.* KPOINTS POSCAR CONTCAR OUTCAR WAVECAR ; cd .. ; done < "$input" 2> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp ../../struct/$line/BPOSCAR ./$line/POSCAR  ; done < "$input"
input="./error_dir" ; while IFS= read -r line ; do cd ./$line ; sbatch ase_fp_job.sh ; cd .. ; done < "$input"
