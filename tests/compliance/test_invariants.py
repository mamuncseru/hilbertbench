import ast
import pathlib
import pytest

MODEL_DIR = pathlib.Path("hilbertbench/models/v1_0")
ALLOWED_IMPORTS = {
    # stdlib
    "from __future__", "from typing", "from uuid", "from enum",
    "from datetime", "import re", "import uuid", "import enum",
    # pydantic (only allowed external dep in models)
    "from pydantic", "import pydantic",
    # intra-package relative imports
    "from .",
}


class TestINV003:
    """INV-003: models are auto-generated, never manually edited."""

    def test_all_generated_files_have_header(self):
        for py_file in MODEL_DIR.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            first_line = py_file.read_text().splitlines()[0]
            assert "AUTO-GENERATED" in first_line, (
                f"{py_file.name} is missing the AUTO-GENERATED header. "
                "Was it manually edited? Re-run generate_models.py"
            )


class TestINV004:
    """INV-004: models must only import stdlib + pydantic."""

    def test_no_forbidden_imports(self):
        forbidden = []
        for py_file in sorted(MODEL_DIR.glob("*.py")):
            src = py_file.read_text()
            for line in src.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or
                        stripped.startswith("from ")):
                    continue
                if not any(stripped.startswith(allowed)
                           for allowed in ALLOWED_IMPORTS):
                    forbidden.append(f"{py_file.name}: {stripped}")

        assert not forbidden, (
            "Models contain forbidden imports (violates INV-004):\n"
            + "\n".join(forbidden)
        )

