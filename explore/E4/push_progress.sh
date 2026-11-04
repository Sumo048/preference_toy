#!/bin/bash
# commit finished results + interim table and push to art/confirm-E4 (and the session branch)
cd "$(dirname "$0")/../.."
python explore/E4/analyze.py > explore/E4/table_interim.md 2>&1
git add explore/E4/results explore/E4/logs explore/E4/queue.log explore/E4/table_interim.md docs/ 2>/dev/null
git commit -qm "E4: progress, $(ls explore/E4/results/*/*/*.json | wc -l)/420 runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012G16ihTFWEweZ8JGQZDARU" || exit 0
for i in 1 2 3 4; do git push -q origin art/confirm-E4 && break; sleep $((2**i)); done
git push -q origin art/confirm-E4:claude/confirm-e4-label-free-zipbi9 || true
git log --oneline -1
