#!/bin/bash

target=$1
if [ $# -eq 1 ]
then
    ncpu=1
else
    ncpu=$2
fi

DBPATH=""
uniprot20=$DBPATH/hh-suite/uniprot20/uniprot20_2016_02
hhpath=/lustre/protein/programs/hh-suite
hhbindir=$hhpath/bin
log="gen_features.log"

$hhbindir/hhblits -i $target.fa -d $uniprot20 -oa3m $target.a3m -cpu $ncpu 2>&1 | tee -a $log

$hhpath/scripts/addss_new.pl $target.a3m 2>&1 | tee -a $log
$hhpath/bin/hhmake -i $target.a3m | tee -a $log

if [ ! -e $target.chk ]; then
    echo ">>$target.chk is not exist has problem in a3m, hhr, ..." 
    echo "check it"
    exit
fi

