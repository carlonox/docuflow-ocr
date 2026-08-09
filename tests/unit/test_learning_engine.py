"""Unit tests for the learning engine."""

from __future__ import annotations

import json

import pytest

from docuflow.learning.learning_engine import (
    LearningEngine,
    apply_similar_reference_knowledge,
    extract_numbers_from_filename,
    find_similar_references,
    levenshtein_distance,
)


# ---------------------------------------------------------------------- #
# Filename parsing
# ---------------------------------------------------------------------- #


class TestFilenameParsing:
    def test_correct_filename(self):
        expected, detected = extract_numbers_from_filename("4528913_9999999999.png")
        assert expected == "4528913"
        assert detected is None

    def test_error_filename(self):
        expected, detected = extract_numbers_from_filename(
            "4528913_detected_04528913_9999999999.png"
        )
        assert expected == "4528913"
        assert detected == "04528913"

    def test_unrecognized_filename(self):
        assert extract_numbers_from_filename("not_a_number.png") == (None, None)


# ---------------------------------------------------------------------- #
# Levenshtein
# ---------------------------------------------------------------------- #


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein_distance("12345", "12345") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("12345", "12355") == 1

    def test_insertion(self):
        assert levenshtein_distance("12345", "012345") == 1

    def test_empty(self):
        assert levenshtein_distance("", "12345") == 5
        assert levenshtein_distance("12345", "") == 5


# ---------------------------------------------------------------------- #
# Similar reference lookup
# ---------------------------------------------------------------------- #


class TestSimilarReferences:
    def test_finds_close_references(self):
        index = {
            "12345": ["references/a.png"],
            "12445": ["references/b.png"],
            "99999": ["references/c.png"],
        }
        results = find_similar_references("12345", index, max_distance=1)
        assert len(results) == 2
        assert results[0]["number"] == "12345"

    def test_prefers_manual_paths(self):
        index = {
            "12345": ["automatic/x.png", "references/manual/y.png"],
        }
        results = find_similar_references("12345", index)
        assert results[0]["priority"] == "manual"
        assert "manual" in results[0]["path"]

    def test_empty_index(self):
        assert find_similar_references("12345", {}) == []


class TestApplySimilarKnowledge:
    def test_prefix_pattern(self):
        similar = [{"number": "012345"}]
        strategies = apply_similar_reference_knowledge("12345", similar)
        assert "0" in strategies["prefix_corrections"]
        assert strategies["confidence"] > 0

    def test_digit_confusions(self):
        similar = [{"number": "12346"}]
        strategies = apply_similar_reference_knowledge("12345", similar)
        assert strategies["digit_confusion_corrections"][0] == {
            "position": 4, "expected": "5", "detected": "6",
        }

    def test_no_similar(self):
        strategies = apply_similar_reference_knowledge("12345", [])
        assert strategies["confidence"] == 0.0


# ---------------------------------------------------------------------- #
# Learning engine end-to-end (synthetic corpus)
# ---------------------------------------------------------------------- #


class TestLearningEngine:
    def test_defaults_are_created(self, tmp_path):
        engine = LearningEngine(
            fields=["field_a", "field_b"],
            config_path=str(tmp_path / "config.json"),
            stats_path=str(tmp_path / "stats.json"),
            error_patterns_path=str(tmp_path / "patterns.json"),
        )
        config = engine.get_config()
        assert "field_a" in config["fields"]
        assert config["fields"]["field_a"]["min_confidence"] == 0.6

    def test_analyze_references_mines_patterns(self, tmp_path):
        # Build a small synthetic corpus:
        # field_a: 2 correct, 1 error with a glued prefix.
        correct_dir = tmp_path / "refs" / "field_a" / "correct"
        error_dir = tmp_path / "refs" / "field_a" / "error"
        correct_dir.mkdir(parents=True)
        error_dir.mkdir(parents=True)
        for name in ("12345_1000.png", "12346_1001.png"):
            (correct_dir / name).write_bytes(b"fake")
        (error_dir / "12347_detected_012347_1002.png").write_bytes(b"fake")

        engine = LearningEngine(
            fields=["field_a"],
            config_path=str(tmp_path / "config.json"),
            stats_path=str(tmp_path / "stats.json"),
            error_patterns_path=str(tmp_path / "patterns.json"),
        )
        applied = engine.analyze_references(str(tmp_path / "refs"))
        assert applied >= 1

        patterns = engine.get_error_patterns()
        field_patterns = patterns["fields"]["field_a"]
        assert field_patterns["added_prefixes"].get("0") == 1

        # Stats recorded the success rate: 2/3 correct.
        assert engine.stats["total_analyses"] == 1
        assert 60 <= engine.stats["current_success_rate"] <= 70

    def test_correction_rules_persisted(self, tmp_path):
        engine = LearningEngine(
            fields=["field_a"],
            config_path=str(tmp_path / "config.json"),
            stats_path=str(tmp_path / "stats.json"),
            error_patterns_path=str(tmp_path / "patterns.json"),
        )
        # Inject a prefix rule directly and verify persistence round-trip.
        engine.config["fields"]["field_a"]["correction_rules"].append(
            {"type": "strip_prefix", "pattern": "0"}
        )
        engine.save_config()

        reloaded = LearningEngine(
            fields=["field_a"],
            config_path=str(tmp_path / "config.json"),
            stats_path=str(tmp_path / "stats.json"),
            error_patterns_path=str(tmp_path / "patterns.json"),
        )
        rules = reloaded.correction_rules_for("field_a")
        assert {"type": "strip_prefix", "pattern": "0"} in rules
