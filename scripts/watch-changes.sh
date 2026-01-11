#!/bin/bash
# Watch for changes and summarize with AI
# Usage: ./scripts/watch-changes.sh [workspace]

WORKSPACE=${1:-""}
if [ -n "$WORKSPACE" ]; then
    cd ~/Downloads/deep-research-v0-$WORKSPACE
fi

echo "Watching for changes in $(pwd)..."
echo "Press Ctrl+C to stop"
echo ""

LAST_HASH=""

while true; do
    # Get current state hash
    CURRENT_HASH=$(git diff 2>/dev/null | md5)

    if [ "$CURRENT_HASH" != "$LAST_HASH" ] && [ -n "$(git diff 2>/dev/null)" ]; then
        echo "=== Changes detected $(date +%H:%M:%S) ==="

        # Get diff stats
        git diff --stat

        # Get actual diff (truncated)
        DIFF=$(git diff | head -200)

        echo ""
        echo "--- Summary request ready ---"
        echo "Run /summarize in Claude or pipe diff to AI:"
        echo "git diff | claude --print 'Summarize these code changes in 3-5 bullet points'"
        echo ""

        LAST_HASH=$CURRENT_HASH
    fi

    sleep 5
done
