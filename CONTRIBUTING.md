# Contributing to OralAID

Thanks for helping. Bug reports, model evaluations, UI work and documentation are all welcome.

## Ground rules

- **Never commit patient data.** No intraoral photos, datasets, CSVs, databases or model weights. `.gitignore` blocks the common file types; check `git status` before every commit anyway.
- **Be honest about the model.** Report metrics as measured, say what data they came from, and don't invent numbers or testimonials.
- **No emojis in code or logs.** Use the `logging` module, not `print`, in application code.

## Development setup

See [Run it locally](README.md#run-it-locally). Then:

```bash
# Backend checks
cd oralaid_app
ruff check .
pytest -q

# Frontend checks
cd frontend
npm run typecheck
npm test
npm run build
```

CI runs the same commands, plus the security scans and a Docker smoke test, on every pull request.

## Pull requests

1. Branch from `main`. `prod` is the deployed branch.
2. Keep changes focused, and add or update tests.
3. Explain what changed and how you verified it, with screenshots for UI changes.
4. Make sure CI is green.

## Reporting security issues

Please don't open a public issue. Follow [SECURITY.md](SECURITY.md).
