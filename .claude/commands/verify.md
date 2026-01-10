# Verify Work

Spawn a verification check before considering work complete.

## What to Verify

1. **Tests pass**: Run `pytest tests/unit/ -v`
2. **No regressions**: Check that existing functionality still works
3. **Code quality**: Run `ruff check src/`
4. **Matches requirements**: Re-read the original request, confirm it's addressed

## Process

1. Run all verification checks
2. Report any issues found
3. Only mark work complete if ALL checks pass

## If Issues Found

- Fix them before marking complete
- Re-run verification after fixes
