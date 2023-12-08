#!/bin/bash
#SBATCH --job-name=idir120
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --output=idir120.%j.out
#SBATCH --error=idir120.%j.err
#SBATCH --time=4-00:00:00 
#SBATCH --mem=20G
#SBATCH --nodelist=prometheus

export mypython=/u/home/hammerni/.conda/envs/idir/bin/python

mod=localAM-3l-256
exp=KH-N120-AE-$mod-Omega50-rescale

echo GPU=$CUDA_VISIBLE_DEVICES

$mypython run-general.py --exp=$exp --mode=train --user=KHprometheus