#!/usr/bin/env python
#
# file: tools/gen_test_catalog.py
#
# revision history:
#  20260611 (am): initial version
#
# Generates docs/reference/test-catalog.md: one entry for every test
# in the suite, grouped by subsystem and test file, with a
# human-readable explanation extracted from (in priority order) the
# test's docstring, its first inline comment, or its name.
#
# Run it after adding tests so the catalog never drifts:
#   python tools/gen_test_catalog.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import ast
import os
import sys
from pathlib import Path

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# repository layout
#
REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
OUT = REPO / "docs" / "reference" / "test-catalog.md"

# one paragraph of context per test directory (the "why this area
# exists" that individual test names cannot carry)
#
DIR_INTROS = {
    "recorder": (
        "The recorder is the write path: `HilbertTape`, spans, events, "
        "and the content-addressed artifact store. These tests protect "
        "the append-only discipline (INV-002), atomic sealing, and the "
        "guarantee that every initiated span terminates explicitly "
        "(INV-007)."
    ),
    "trace": (
        "`HilbertTrace` is the public read API. These tests guarantee "
        "that whatever the recorder wrote, the reader resolves back "
        "exactly — spans, outcomes, parameters, circuits, calibration "
        "history — without the caller knowing about storage details."
    ),
    "integrations": (
        "The proxies (Qiskit Estimator/Sampler, backend.run, "
        "PennyLane) must be perfectly transparent to the wrapped "
        "framework (1:1 execution parity, INV-001) while recording "
        "faithfully. This area also covers calibration-snapshot "
        "capture across all three backend-access conventions found in "
        "the wild, drift refresh, and shot/precision evidence."
    ),
    "analysis": (
        "Each analyzer is tested against constructed ground-truth "
        "traces: plant a known condition, assert the verdict and the "
        "quantitative evidence (variance, SNR, KL, fidelity) with "
        "their confidence intervals. Includes regression tests for "
        "hardware-format (ISA) circuits and active-qubit calibration "
        "scoping."
    ),
    "active": (
        "Active Mode is the opt-in interventional path "
        "(expressibility probing). These tests check the probe "
        "records correctly, labels the trace as active, and that "
        "analyzers needing interventional data refuse passive traces."
    ),
    "compliance": (
        "Architecture-level checks that the documented invariants "
        "(INV-001 and friends) hold end to end — the contract the "
        "paper and the docs promise."
    ),
    "reader": (
        "Verification: `trace.verify()` must pass on honest traces "
        "and fail loudly on any tampering — the property the blinded "
        "validation protocol depends on."
    ),
    "e2e": (
        "Full journeys: record a realistic workload, seal, reopen, "
        "analyze — the integration surface a real user touches."
    ),
    "tools": (
        "The blinded-corpus protocol tool: leakage auditing, "
        "verbatim blinding with random IDs, SHA-256 answer-key "
        "commitments, and confusion-matrix scoring with Wilson "
        "intervals."
    ),
}

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def prettify(name: str) -> str:
    """
    function: prettify

    arguments:
     name: a test function name (test_snake_case)

    return:
     a readable sentence fragment derived from the name
    """

    # strip the prefix and re-space
    #
    text = name.removeprefix("test_").replace("_", " ")
    return text[0].upper() + text[1:]
#
# end of function


def first_comment(source_lines: list, node: ast.AST) -> str:
    """
    function: first_comment

    arguments:
     source_lines: the file's source split into lines
     node:         the function node to scan

    return:
     the first inline comment inside the function body, or ''
    """

    # scan the body lines for the first comment
    #
    for line_number in range(node.lineno, node.end_lineno):
        stripped = source_lines[line_number].strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
    return ""
#
# end of function


def describe(node: ast.AST, source_lines: list) -> str:
    """
    function: describe

    arguments:
     node:         a test function node
     source_lines: the file's source lines

    return:
     the best available explanation for the test
    """

    # priority: docstring, then first comment, then the name itself
    #
    doc = ast.get_docstring(node)
    if doc:
        return " ".join(doc.split())
    comment = first_comment(source_lines, node)
    pretty = prettify(node.name)
    if comment:
        return f"{pretty} — {comment}"
    return pretty
#
# end of function


def param_note(node: ast.AST) -> str:
    """
    function: param_note

    arguments:
     node: a test function node

    return:
     ' *(parametrized)*' when the test expands to multiple cases
    """

    # look for a pytest.mark.parametrize decorator
    #
    for dec in getattr(node, "decorator_list", []):
        text = ast.unparse(dec)
        if "parametrize" in text:
            return " *(parametrized)*"
    return ""
#
# end of function


def main() -> int:
    """
    function: main

    arguments:
     none

    return:
     process exit code

    description:
     Walks tests/, extracts every test with its explanation, and
     writes the grouped markdown catalog.
    """

    # collect per-directory, per-file structures
    #
    sections = []
    total = 0
    for area in sorted(p for p in TESTS.iterdir() if p.is_dir()):
        if area.name.startswith("__"):
            continue
        files = sorted(area.glob("test_*.py"))
        if not files:
            continue
        file_blocks = []
        for path in files:
            source = path.read_text()
            lines = [""] + source.splitlines()   # 1-indexed access
            tree = ast.parse(source)
            module_doc = ast.get_docstring(tree) or ""

            # walk classes and module-level functions
            #
            blocks = []
            for node in tree.body:
                if (isinstance(node, ast.ClassDef)
                        and node.name.startswith("Test")):
                    rows = [
                        (fn.name,
                         describe(fn, lines) + param_note(fn))
                        for fn in node.body
                        if isinstance(fn, (ast.FunctionDef,
                                           ast.AsyncFunctionDef))
                        and fn.name.startswith("test_")
                    ]
                    if rows:
                        blocks.append((node.name,
                                       ast.get_docstring(node) or "",
                                       rows))
                        total += len(rows)
                elif (isinstance(node, (ast.FunctionDef,
                                        ast.AsyncFunctionDef))
                      and node.name.startswith("test_")):
                    blocks.append((None, "", [(
                        node.name,
                        describe(node, lines) + param_note(node),
                    )]))
                    total += 1
            file_blocks.append((path, module_doc, blocks))
        sections.append((area.name, file_blocks))

    # render the markdown
    #
    out = [
        "# Test Catalog",
        "",
        "Every test in the suite, what it verifies, and why its area",
        "exists. Generated by `tools/gen_test_catalog.py` — regenerate",
        "after adding tests.",
        "",
        f"**Total test functions: {total}** (run them with",
        "`python -m pytest tests/ -q`).",
        "",
    ]
    for area, file_blocks in sections:
        out.append(f"## `tests/{area}/`")
        out.append("")
        if area in DIR_INTROS:
            out.append(DIR_INTROS[area])
            out.append("")
        for path, module_doc, blocks in file_blocks:
            rel = path.relative_to(REPO)
            out.append(f"### {path.name}")
            out.append("")
            if module_doc:
                first_para = module_doc.split("\n\n")[1] \
                    if "\n\n" in module_doc else module_doc
                first_para = " ".join(first_para.split())
                out.append(f"*{first_para}*")
                out.append("")
            for class_name, class_doc, rows in blocks:
                if class_name:
                    out.append(f"**{class_name}**"
                               + (f" — {' '.join(class_doc.split())}"
                                  if class_doc else ""))
                    out.append("")
                out.append("| Test | What it verifies |")
                out.append("|---|---|")
                for name, desc in rows:
                    desc = desc.replace("|", "\\|")
                    out.append(f"| `{name}` | {desc} |")
                out.append("")
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT} ({total} test functions)")

    # exit gracefully
    #
    return 0
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
