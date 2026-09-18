# Causal Impact Measurement

Compare naive estimates with difference-in-differences, synthetic control and propensity stratification on data with known planted effects.

## Implementation and validation

Three executable method implementations and their regression tests. No cloud account required.

Automated checks: **9 tests**. The GitHub Actions run linked above the file browser is the current CI result. Local checks and external integrations are separate claims.

## Reproduce locally

Use Python 3.12. Run from this repository’s root in a fresh virtual environment.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

For repositories using `src/`, set the import path before running commands:

```powershell
# PowerShell
$env:PYTHONPATH="src"
```
```sh
# macOS/Linux
export PYTHONPATH=src
```

```sh
python -m pytest tests -q
```

## Data and interpretation

Synthetic cases with known effects. Real-world causal interpretation depends on parallel trends, donor fit or observed-confounding assumptions; these demonstrations do not prove those assumptions in new data.

## Inspect the work

- [`tests/`](tests/) — executable checks and examples.
- [`docs/`](docs/) — methodology, integration specifications and the historical design.
- [Portfolio](https://raveesh-rajg.github.io/) — project directory.

## Completion boundary

Passing local tests establishes the checks listed in this repository. It does not establish cloud deployment, real-data quality, production security, or native BI rendering unless an explicit verification record says so.
