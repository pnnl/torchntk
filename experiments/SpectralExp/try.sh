#!/bin/bash
#SBATCH -p dlt
#SBATCH -A task0_pmml
#SBATCH -J try_reg
#SBATCH -N 1
#SBATCH -t 24:0:0
#SBATCH -n 16
#SBATCH --gres=gpu:8
#SBATCH -o try.out.%j
#SBATCH -e try.err.%j

python change_reg.py --eta 0.02 --steps 4000 --lda 0.006 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.0005 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.01 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.005 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.004 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.003 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 4000 --lda 0.002 --gamma1 0.1 --gamma2 0.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.006 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.0005 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.01 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.005 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.004 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.003 --gamma1 0.1 --gamma2 1.5
python change_reg.py --eta 0.02 --steps 5000 --lda 0.002 --gamma1 0.1 --gamma2 1.5
