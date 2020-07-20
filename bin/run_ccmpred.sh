#!/bin/bash

programsdir=/lustre/protein/programs
contactdir=$programsdir/contact
ccmpredbindir=$contactdir/CCMpred/bin

target=$1

if [ $# -eq 1 ]
then
    nprocessor=1
else
    nprocessor=$2
fi

nalign=`cat $target.aln|wc -l`
echo "Nalign:" $nalign

if [ $nalign -gt 5 ]
then
    echo "Running CCMpred..."
    $ccmpredbindir/ccmpred $target.aln $target.ccmpred -t $nprocessor
else
    rm -f $target.ccmpred
    touch $target.ccmpred
fi


