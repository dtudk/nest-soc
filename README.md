# nest
Development code for the NEST (Next-gen electrochemical system tools) project

## Development setup

```bash
pip install -r requirements_dev.txt
```

### Quality checks

- Lint: `ruff check .`
- Tests: `pytest`
- Build: `python -m build`

These same commands run automatically in the GitHub Actions workflow located at `.github/workflows/ci.yml` on pushes to `main` and `feat/dev_env`, as well as pull requests targeting `main`.
