#!/bin/bash 

# set directory and file path
programsdir=/lustre/protein/programs
contactdir=$programsdir/contact
hhbdir=$programsdir/hh-suite
domdir=
#export HHLIB=/share/database/build/bin/hhpred
database=/lustre/protein/database/uniref/20160504/uniref90.fasta

target=${1%.*}

if [ $# -eq 1 ]
then
    nprocessor=1
else
    nprocessor=$2
fi

jackhmmerbin=$contactdir/hmmer/bin
$jackhmmerbin/jackhmmer -N 4 --cpu $nprocessor -o $target.dat -A $target.align $target.fasta $database 
$jackhmmerbin/esl-reformat -o $target.a2m a2m $target.align
$hhbdir/scripts/reformat.pl -r $target.a2m $target.hmm.fas
$domdir/jackhammer_aln.py $target
$domdir/jackhammer_si.py $target

rm -f $target.align $target.a2m $target.hmm.fas 

