"""Turn the percent-format scripts in ``tools/nb_src`` into executed notebooks.

Keeping the notebooks' real source as plain ``.py`` files means code review and
``git diff`` show actual lines rather than JSON noise, while the committed
``.ipynb`` still carries its outputs for the marker.

Cell grammar
------------
``# %%`` starts a code cell, ``# %% [markdown]`` starts a markdown cell whose
lines are un-commented. Anything before the first marker is ignored.

Usage
-----
    python tools/build_notebooks.py              # every script
    python tools/build_notebooks.py project1_population
    python tools/build_notebooks.py --no-exec    # parse only, do not run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "tools" / "nb_src"
OUT_DIR = ROOT / "notebooks"
MARKER = "# %%"
TIMEOUT_SECONDS = 900


@dataclass
class Cell:
    """One parsed cell: its kind and its raw lines."""

    kind: str
    lines: list[str]

    def text(self) -> str:
        """Trim blank lines, and un-comment markdown."""
        body = list(self.lines)
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if self.kind == "markdown":
            body = [line[2:] if line.startswith("# ") else line.lstrip("#")
                    for line in body]
        return "\n".join(body)


def split_cells(source: str) -> list[Cell]:
    """Split a percent-format script into cells."""
    cells: list[Cell] = []
    for line in source.splitlines():
        if line.startswith(MARKER):
            cells.append(Cell("markdown" if "[markdown]" in line else "code", []))
        elif cells:
            cells[-1].lines.append(line)
    if not cells:
        raise ValueError(f"no {MARKER!r} markers found -- is this a percent-format file?")
    return [c for c in cells if c.text()]


def to_notebook(source: str) -> nbformat.NotebookNode:
    """Assemble a v4 notebook from a percent-format script."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    for cell in split_cells(source):
        make = (nbformat.v4.new_markdown_cell if cell.kind == "markdown"
                else nbformat.v4.new_code_cell)
        nb.cells.append(make(cell.text()))
    return nb


def build(path: Path, execute: bool = True) -> Path:
    """Build one notebook, optionally running it top to bottom."""
    nb = to_notebook(path.read_text(encoding="utf-8"))
    if execute:
        runner = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name="python3")
        try:
            runner.preprocess(nb, {"metadata": {"path": str(OUT_DIR)}})
        except CellExecutionError as err:
            raise SystemExit(f"{path.name}: a cell failed, notebook not written\n{err}")
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / f"{path.stem}.ipynb"
    nbformat.write(nb, target)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="script stems to build; default all")
    parser.add_argument("--no-exec", action="store_true",
                        help="assemble without running the cells")
    args = parser.parse_args(argv)

    scripts = ([SRC_DIR / f"{n}.py" for n in args.names] if args.names
               else sorted(SRC_DIR.glob("*.py")))
    missing = [s for s in scripts if not s.exists()]
    if missing:
        raise SystemExit(f"no such script(s): {[str(m) for m in missing]}")

    for script in scripts:
        written = build(script, execute=not args.no_exec)
        print(f"built {written.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
