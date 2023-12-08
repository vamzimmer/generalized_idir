#!/bin/bash
#SBATCH --job-name=idir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --output=idir.%j.out
#SBATCH --error=idir.%j.err
#SBATCH --time=4-00:00:00 
#SBATCH --mem=20G 

export mypython=/u/home/hammerni/.conda/envs/idir/bin/python

mod=localAM-3l-256
exp=KH-N10-AE-$mod-Omega50

echo GPU=$CUDA_VISIBLE_DEVICES

$mypython run-general.py --exp=$exp --mode=train --user=KH