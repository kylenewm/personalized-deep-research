#!/bin/bash
# Run this from a workspace directory to merge changes back to base

set -e

BASE_DIR="/Users/kylenewman/Downloads/deep-research-v0"
CURRENT_DIR="$(pwd)"

# Check we're in a workspace
if [[ ! "$CURRENT_DIR" =~ deep-research-v0-t[1-5]$ ]]; then
    echo "Error: Run this from a workspace directory (deep-research-v0-t1 through t5)"
    exit 1
fi

WS_NAME=$(basename "$CURRENT_DIR")

echo "Merging $WS_NAME → base"
echo ""

# Find changed files (excluding symlinked files)
CHANGED=$(git status --porcelain 2>/dev/null | grep -v "STATE.md\|LOG.md\|CLAUDE.md\|ARCHITECTURE.md\|INVARIANTS.md\|WORKFLOW.md\|\.claude" | awk '{print $2}')

if [ -z "$CHANGED" ]; then
    echo "No changes to merge."
    exit 0
fi

echo "Changed files:"
echo "$CHANGED"
echo ""

read -p "Merge these to base? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Copy changed files to base
for file in $CHANGED; do
    if [ -f "$CURRENT_DIR/$file" ]; then
        mkdir -p "$BASE_DIR/$(dirname "$file")"
        cp "$CURRENT_DIR/$file" "$BASE_DIR/$file"
        echo "Copied: $file"
    fi
done

echo ""
echo "Merged to base. Now in base directory:"
echo "  cd $BASE_DIR"
echo "  git diff"
echo "  git add -A && git commit -m 'Merge from $WS_NAME'"
