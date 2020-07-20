#!/bin/bash 

#ConDo target.fa ncpu

ConDodir=$HOME/programs/ConDo
ConDobin=$ConDodir/bin
weight_dir=$ConDodir/save
target=${1%.*}

if [ $# -eq 1 ]
then
    nprocessor=1
else
    nprocessor=$2
fi

$ConDobin/run_jackhmmer.sh $target $nprocessor
$ConDobin/run_ccmpred.sh $target $nprocessor
$ConDobin/gen_features.sh $target $nprocessor
$ConDobin/feature.py $target 
$ConDobin/pred_ss.py $weight_dir
$ConDobin/pred_sa.py $weight_dir
$ConDobin/pred_dom.py $weight_dir

$ConDobin/gen_results.py $target

