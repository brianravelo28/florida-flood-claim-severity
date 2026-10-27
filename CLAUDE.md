# Notes for Claude Code

- **Use current stable versions, and verify them.** Do not write package or Python versions from memory. Check with
  `pip index versions <package>` and confirm wheels exist for the target Python (`pip download --no-deps --only-binary=:all:`).
  Pins live in `requirements.txt`, `requirements-render.txt` and `requirements-dev.txt`; CI (`.github/workflows/tests.yml`) and
  Render (the `PYTHON_VERSION` setting in Render's dashboard, which overrides `render.yaml`) must use Python versions those pins support. After any bump, rerun the
  pipeline and compare outputs with the previous ones; a version change should not change the numbers.
- **Run `python -m pytest` before committing.** The suite uses small fixtures and the committed `deploy_data/`; it needs no
  downloaded data.
- **Numbers in the README, `docs/METHODOLOGY.md` and the app text come from the pipeline's JSON outputs.** Change them by
  rerunning the pipeline in the order the README lists, then updating the text. Do not type a number that a file computes.
- **Do not edit pipeline code to fix a documentation problem.** Code changes go in their own commit.
- **Dashboard conventions:** every Plotly chart has the hint line underneath it (`chart_hint()` in `app.py`); Dash dropdowns with
  fewer than 7 options use `searchable=False`.
