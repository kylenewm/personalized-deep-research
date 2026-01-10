# Slash Commands

## Core Workflow

| Command | What it does |
|---------|--------------|
| `/test` | Run pytest |
| `/done` | Verify before marking complete (tests + lint + requirements check) |
| `/review` | Spawn subagent to review changes with fresh eyes |
| `/ship` | Full pipeline: verify → commit → push → PR |

## Utilities

| Command | What it does |
|---------|--------------|
| `/commit` | Stage + commit with co-author |
| `/pr` | Push + create GitHub PR |
| `/verify` | Tests + ruff + requirement check |
| `/sandbox` | Test Pipeline v2 (no API costs) |
| `/save` | Update STATE.md + LOG.md |

## Recommended Flow

```
work → /test → /done → /review → /ship
        ↑                          |
        └──── fix if issues ───────┘
```

## Subagent Commands

- `/review` - spawns fresh agent to review diff, catches blind spots
- `/done` - can spawn verification agent before marking complete

## Test Commands Work

Type `/test` - if pytest runs, commands are working.
