#!/bin/bash

# mod=localCrossAttentionAMv2

# for exp in KH-N10-AE-$mod-3l-256-Omega50-rescale; do
# echo $exp $gpu
# CUDA_VISIBLE_DEVICES=$gpu python run-patches.py --exp=$exp --mode=train --user=KH
# done

gpu=0
#mod=localQAM-3l-256
#mod=localAttentionAM-3l-256
mod=localAM-5l-512
#mod=localCrossAttentionAM-5l-512
#mod=SIREN+-3l-256

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


for exp in KH-N120-AE-$mod-Omega50-rescale; do
echo $exp $gpu
export CUDA_VISIBLE_DEVICES=$gpu 
taskset -c $cpu python run-patches.py --exp=$exp --mode=train --user=KH
done