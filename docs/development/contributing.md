# Contributing

If you want to contribute to BEAM ABM, open an issue first for any non-trivial
change so the scope is clear before you start work.

Submit changes through a pull request.

Before opening a pull request, make sure the tests and pre-commit checks pass:

```bash
make test
make pre-commit
```

If your change touches the documentation, also make sure the docs site serves
locally:

```bash
make docs
```
