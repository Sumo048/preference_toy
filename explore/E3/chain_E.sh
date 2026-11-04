#!/bin/bash
cd "$(dirname "$0")/../.."
while kill -0 $(cat /tmp/claude-0/chain.pid) 2>/dev/null; do sleep 20; done
explore/E3/runall.sh "a=track-w=1-p=lora4-lr=1e-3,a=track-w=1-p=b2,a=track-w=1-p=out-lr=1e-3,a=track-w=1-st=1000,a=track-w=1-p=head-lr=1e-3,a=track-w=0.75,a=track-w=1-st=1000-lr=3e-4"
echo CHAIN_E_DONE
