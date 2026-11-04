#!/bin/bash
cd "$(dirname "$0")/../.."
explore/E3/runall.sh "B0,B1a,a=track-w=1,a=track60k-w=1-st=20000,o=1-a=none,a=track" "3 4"
explore/E3/runall.sh "B0,B1a,a=track-w=1,a=track60k-w=1-st=20000" "0 1 2" "k4 alt" "1 64"
explore/E3/runall.sh "a=omask-w=1-st=20000,a=track-w=1-on=map,p=b0,p=b1,p=fc2,p=cond,p=lora16-lr=1e-3,a=map,a=local,a=dist20k,w=0.75,lr=3e-4,m=1,p=head-lr=1e-3"
echo CHAIN_DONE
