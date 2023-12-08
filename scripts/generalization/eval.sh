#!/bin/bash

gpu=$1

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

#for mod in localAM-3l-256 localAM-3l-512 localAM-5l-512 localAM-5l-256; do
#for mod in localQAM-3l-256 localAM-3l-256-AE10 localAM-3l-256-AE8-unfreeze; do 
#exp=KH-N10-AE-$mod-Omega50-rescale
#for mod in localAM-3l-256 localAM-5l-256 localQAM-3l-256 globalAM-3l-256 localAM-3l-512 localAM-5l-512; do
# localAM-3l-256 363523
# localAM-5l-256 
for mod in localAM-3l-256 localAM-3l-512 localAM-5l-256 localAM-5l-512  ; do
exp=KH-N120-AE-$mod-Omega50-rescale
echo $exp $gpu $cpu
export CUDA_VISIBLE_DEVICES=$gpu 
python run-patches.py --exp=$exp --mode=test --user=KH
python run-patches.py --exp=$exp --mode=eval --user=KH
done