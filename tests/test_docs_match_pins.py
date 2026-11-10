"""The version numbers the docs quote are the ones the project pins and runs."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODOLOGY = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def pins(name):
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return dict(line.split("==") for line in lines if "==" in line and not line.startswith("#"))


def test_the_methodology_software_line_names_the_pinned_versions():
    software = re.search(r"\*\*Software\.\*\*(.*)", METHODOLOGY).group(1)
    pinned = pins("requirements.txt")
    for label, package in {"pandas": "pandas", "NumPy": "numpy", "PyArrow": "pyarrow", "Requests": "requests", "Plotly": "plotly",
                           "Dash": "dash"}.items():
        assert f"{label} {pinned[package]}" in software, f"{label} {pinned[package]} is pinned but the Software line says otherwise"


def test_the_methodology_deployment_notes_name_the_pinned_gunicorn():
    assert f"gunicorn {pins('requirements-render.txt')['gunicorn']}" in METHODOLOGY


def test_the_hosted_requirements_use_the_same_versions_as_the_full_requirements():
    full, hosted = pins("requirements.txt"), pins("requirements-render.txt")
    assert {k: v for k, v in hosted.items() if k in full} == {k: full[k] for k in hosted if k in full}


def test_the_python_versions_in_the_docs_match_the_workflow_and_the_render_settings():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    matrix = re.findall(r'"(3\.\d+)"', workflow)
    assert matrix and f"Python {matrix[0]} and {matrix[-1]}" in METHODOLOGY and f"Python {matrix[0]} and {matrix[-1]}" in README
    assert f"Requires Python {matrix[0]} or newer" in README
    hosted = re.search(r"PYTHON_VERSION\s*\n\s*value:\s*(\d+\.\d+)", (ROOT / "docs" / "render.yaml").read_text(encoding="utf-8")).group(1)
    assert f"the hosted dashboard runs on {hosted}" in README and f"Python {hosted} with" in METHODOLOGY
