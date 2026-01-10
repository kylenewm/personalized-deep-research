#!/bin/bash
# Sets up 5 isolated workspaces with shared state files

set -e

BASE_DIR="/Users/kylenewman/Downloads/deep-research-v0"
PARENT_DIR="/Users/kylenewman/Downloads"

# Shared files (symlinked to base)
SHARED_FILES=("STATE.md" "LOG.md" "CLAUDE.md" "ARCHITECTURE.md" "INVARIANTS.md" "WORKFLOW.md")

echo "Setting up 5 workspaces..."
echo ""

for i in 1 2 3 4 5; do
    WS_DIR="$PARENT_DIR/deep-research-v0-t$i"

    if [ -d "$WS_DIR" ]; then
        echo "[t$i] Already exists, updating symlinks..."
    else
        echo "[t$i] Creating workspace..."
        cp -r "$BASE_DIR" "$WS_DIR"
    fi

    # Replace shared files with symlinks
    for file in "${SHARED_FILES[@]}"; do
        if [ -f "$BASE_DIR/$file" ]; then
            rm -f "$WS_DIR/$file"
            ln -s "$BASE_DIR/$file" "$WS_DIR/$file"
        fi
    done

    # Also symlink .claude directory for shared commands/settings
    rm -rf "$WS_DIR/.claude"
    ln -s "$BASE_DIR/.claude" "$WS_DIR/.claude"

    echo "[t$i] Ready: $WS_DIR"
done

echo ""
echo "Done! Workspaces created:"
echo ""
for i in 1 2 3 4 5; do
    echo "  Terminal $i: cd $PARENT_DIR/deep-research-v0-t$i && claude"
done
echo ""
echo "Shared files are symlinked - edits sync automatically."
echo "Code is isolated - no conflicts during work."
