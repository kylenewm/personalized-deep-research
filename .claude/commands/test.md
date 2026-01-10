# Run Tests

Run the test suite and report results.

## Commands

```bash
# Activate venv first, or use full path
source venv/bin/activate && pytest tests/unit/ -v

# Or directly:
./venv/bin/pytest tests/unit/ -v

# Integration tests
./venv/bin/pytest tests/integration/ -v
```

## After Running

- If tests pass: report summary
- If tests fail: analyze failures, suggest fixes
- Do NOT mark any todo as complete until tests pass
