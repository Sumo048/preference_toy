#!/bin/bash
# Commit and push finished cells (results JSON, graft weights, unit logs). Safe to run any time.
cd "$(dirname "$0")/../.."
n=$(ls explore/E5/results/graft_K10c_*.json 2>/dev/null | wc -l)
git add explore/E5/results explore/E5/grafts 2>/dev/null; git add -f explore/E5/logs/*.log 2>/dev/null
git diff --cached --quiet && { echo "nothing new ($n cells)"; exit 0; }
git commit -qm "E5: progress, $n/70 cells

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012G16imA2mYeZdBjho15mTX"
for i in 1 2 3 4; do git push -q origin art/confirm-E5 && break; sleep $((2**i)); done
git push -q origin art/confirm-E5:claude/confirm-e1-gated-graft-xohqq9
echo "pushed $n cells"
