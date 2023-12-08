#!/bin/bash
gpu=0

if [[ $gpu -eq 0 ]]
then
    cpu=1-32
elif [[ $gpu -eq 1 ]]
then
    cpu=33-64
elif [[ $gpu -eq 2 ]]
then
    cpu=65-96
elif [[ $gpu -eq 3 ]]
then
    cpu=97-128
elif [[ $gpu -eq 4 ]]
then
    cpu=129-160
elif [[ $gpu -eq 5 ]]
then
    cpu=161-192
elif [[ $gpu -eq 6 ]]
then
    cpu=193-224
elif [[ $gpu -eq 7 ]]
then
    cpu=225-256
fi
echo $gpu $cpu


mod=localAM-3l-256
exp=KH-N10-AE-$mod-Omega50-rescale

echo $exp $gpu
export CUDA_VISIBLE_DEVICES=$gpu 
taskset -c $cpu ~/.conda/envs/idir/bin/python run-general.py --exp=$exp --mode=train --user=KH
