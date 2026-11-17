# Publishing to PyPI

The PyPI name is `evalloop-agents` (the name `evalloop` belongs to a different project).
The import name stays `evalloop`.

One time:

1. Create an account at https://pypi.org and turn on two-factor authentication.
2. Account settings, API tokens, "Add API token" (scope: entire account for the first upload).

Every release:

```bash
pip install build twine
rm -rf dist build                      # PowerShell: Remove-Item -Recurse -Force dist, build
python -m build
twine check dist/*
twine upload dist/*                    # username: __token__   password: your pypi-... token
```

Then tag the release on GitHub:

```bash
git tag v0.6.0
git push origin v0.6.0
```

Bump the version in both `pyproject.toml` and `src/evalloop/__init__.py` before each release.
PyPI never lets you upload the same version twice.
