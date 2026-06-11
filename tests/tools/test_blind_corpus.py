"""
tests/tools/test_blind_corpus.py

Tests for the blinded-corpus protocol tool (tools/blind_corpus.py):
leakage audit, blinding round-trip, commitment verification, and
confusion-matrix scoring.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hilbertbench.recorder.tape import HilbertTape

# load the tool as a module (tools/ is not a package)
_TOOL = Path(__file__).parents[2] / "tools" / "blind_corpus.py"
_spec = importlib.util.spec_from_file_location("blind_corpus", _TOOL)
blind_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blind_corpus)


def make_run(parent: Path, tags: dict) -> Path:
    """Create a minimal sealed run directory with the given tags."""
    with HilbertTape(parent, tags=tags) as tape:
        dummy = "sha256:" + "a" * 64
        with tape.execution_span(payload_ref=dummy) as span:
            span.outcome_ref = span.attach_inline(
                "0.5", kind="execution_outcome", encoding="json"
            )
    return tape.dir_path


@pytest.fixture
def corpus(tmp_path: Path):
    """Two clean runs + a manifest; returns (corpus_dir, manifest_path)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    run_a = make_run(corpus_dir, tags={"corpus_id": "r1"})
    run_b = make_run(corpus_dir, tags={"corpus_id": "r2"})
    manifest = {
        str(run_a.relative_to(corpus_dir)): {"label": "barren_plateau"},
        str(run_b.relative_to(corpus_dir)): {"label": "healthy"},
    }
    manifest_path = corpus_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return corpus_dir, manifest_path


class TestAudit:

    def test_clean_run_passes(self, tmp_path):
        run = make_run(tmp_path, tags={"corpus_id": "x9"})
        assert blind_corpus.audit_run(run) == []

    def test_label_in_tags_is_flagged(self, tmp_path):
        run = make_run(tmp_path, tags={"planted": "barren_plateau"})
        findings = blind_corpus.audit_run(run)
        assert findings  # both key and value match patterns

    def test_label_in_dirname_is_flagged(self, tmp_path):
        run = make_run(tmp_path, tags={"corpus_id": "x1"})
        leaky = run.rename(run.parent / "noise_dominated_007")
        assert blind_corpus.audit_run(leaky)


class TestBlind:

    def test_blinding_roundtrip(self, corpus, tmp_path, capsys):
        corpus_dir, manifest_path = corpus
        out = tmp_path / "blinded"
        rc = blind_corpus.main([
            "blind", "--manifest", str(manifest_path), "--out", str(out),
        ])
        assert rc == 0

        # blinded copies + key + commitment + sheet all exist
        key = json.loads((out / "answer_key.json").read_text())
        sheet = json.loads((out / "diagnosis_sheet.json").read_text())
        assert len(key["corpus"]) == 2
        assert set(sheet) == set(key["corpus"])
        for blind_id in key["corpus"]:
            assert (out / blind_id / "trace.json").is_file()
            assert sheet[blind_id]["label"] is None

        # blinded copies are byte-identical to the originals (sealed)
        for blind_id, truth in key["corpus"].items():
            original = corpus_dir / truth["original_path"]
            same = (out / blind_id / "events.jsonl").read_bytes()
            assert same == (original / "events.jsonl").read_bytes()

    def test_leaky_corpus_is_refused(self, tmp_path):
        corpus_dir = tmp_path / "leaky"
        corpus_dir.mkdir()
        run = make_run(corpus_dir, tags={"planted": "shot_starved"})
        manifest_path = corpus_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            str(run.relative_to(corpus_dir)): {"label": "shot_starved"},
        }))
        rc = blind_corpus.main([
            "blind", "--manifest", str(manifest_path),
            "--out", str(tmp_path / "blinded"),
        ])
        assert rc == 1

    def test_invalid_label_is_refused(self, corpus, tmp_path):
        corpus_dir, manifest_path = corpus
        manifest = json.loads(manifest_path.read_text())
        first = next(iter(manifest))
        manifest[first] = {"label": "exploded"}
        manifest_path.write_text(json.dumps(manifest))
        rc = blind_corpus.main([
            "blind", "--manifest", str(manifest_path),
            "--out", str(tmp_path / "blinded"),
        ])
        assert rc == 1


class TestScore:

    def _blind(self, corpus, tmp_path):
        _, manifest_path = corpus
        out = tmp_path / "blinded"
        blind_corpus.main([
            "blind", "--manifest", str(manifest_path), "--out", str(out),
        ])
        return out

    def test_perfect_diagnosis_scores_one(self, corpus, tmp_path, capsys):
        out = self._blind(corpus, tmp_path)
        key = json.loads((out / "answer_key.json").read_text())
        sheet = {
            bid: {"label": truth["label"], "confidence": 1.0, "notes": ""}
            for bid, truth in key["corpus"].items()
        }
        diag = tmp_path / "diag.json"
        diag.write_text(json.dumps(sheet))
        result_path = tmp_path / "scores.json"

        rc = blind_corpus.main([
            "score", "--key", str(out / "answer_key.json"),
            "--commitment", str(out / "answer_key.sha256"),
            "--diagnosis", str(diag), "--out", str(result_path),
        ])
        assert rc == 0
        result = json.loads(result_path.read_text())
        assert result["accuracy"] == 1.0
        assert result["n"] == 2

    def test_wrong_diagnosis_scores_zero(self, corpus, tmp_path):
        out = self._blind(corpus, tmp_path)
        key = json.loads((out / "answer_key.json").read_text())
        wrong = {"barren_plateau": "healthy", "healthy": "barren_plateau"}
        sheet = {
            bid: {"label": wrong[truth["label"]], "confidence": 0.5}
            for bid, truth in key["corpus"].items()
        }
        diag = tmp_path / "diag.json"
        diag.write_text(json.dumps(sheet))
        result_path = tmp_path / "scores.json"

        rc = blind_corpus.main([
            "score", "--key", str(out / "answer_key.json"),
            "--commitment", str(out / "answer_key.sha256"),
            "--diagnosis", str(diag), "--out", str(result_path),
        ])
        assert rc == 0
        result = json.loads(result_path.read_text())
        assert result["accuracy"] == 0.0

    def test_tampered_key_fails_commitment(self, corpus, tmp_path):
        out = self._blind(corpus, tmp_path)
        key_path = out / "answer_key.json"
        key = json.loads(key_path.read_text())
        first = next(iter(key["corpus"]))
        # flip the label so the tamper is guaranteed to change content
        key["corpus"][first]["label"] = (
            "healthy"
            if key["corpus"][first]["label"] != "healthy"
            else "barren_plateau"
        )
        key_path.write_text(blind_corpus._canonical_json(key))

        sheet = {bid: {"label": "healthy"} for bid in key["corpus"]}
        diag = tmp_path / "diag.json"
        diag.write_text(json.dumps(sheet))

        rc = blind_corpus.main([
            "score", "--key", str(key_path),
            "--commitment", str(out / "answer_key.sha256"),
            "--diagnosis", str(diag),
        ])
        assert rc == 1

    def test_wilson_interval_sane(self):
        low, high = blind_corpus._wilson_interval(30, 36)
        assert 0.0 < low < 30 / 36 < high < 1.0
        assert blind_corpus._wilson_interval(0, 0) == (0.0, 0.0)
