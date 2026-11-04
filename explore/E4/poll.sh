#!/bin/bash
# poll: sleep 90 s x 6, then report progress
cd "$(dirname "$0")/../.."
for i in 1 2 3 4 5 6; do grep -q QUEUE_DONE explore/E4/queue.log && break; sleep 90; done
echo "runs $(ls explore/E4/results/*/*/*.json | wc -l)/420  units $(grep -c '^unit' explore/E4/queue.log)/140  $(date +%H:%M)"
grep '^unit' explore/E4/queue.log | grep -v 'exit 0'
grep -il traceback explore/E4/logs/* 2>/dev/null
grep -q QUEUE_DONE explore/E4/queue.log && echo QUEUE_DONE
