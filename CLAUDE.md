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
- **Dashboard conventions:** every Plotly chart has a collapsed "About this chart" section whose bullets include its interaction tips (`about()` in `app.py`; the legend tip only on a chart that has a legend), and bullets in these sections are caveats or tips, never a restatement of what a chart or legend already shows; Dash dropdowns with
  fewer than 7 options use `searchable=False`.
- **Wording describes what the project does.** In the app, README and Methodology, state what the dashboard shows and how to read it.
  Do not announce features it lacks or that were cut ("does not compare eras", "no model"); a caveat is phrased as what the data
  or figure covers ("the map shows counties with 30 or more claims").
